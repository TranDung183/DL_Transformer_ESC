import os
import sys

# 1. CẤU HÌNH MÔI TRƯỜNG
TORCH_LIB = r"C:\CODE\Torch"
if TORCH_LIB not in sys.path:
    sys.path.insert(0, TORCH_LIB)
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import librosa
from flask import Flask, render_template, request, jsonify
from transformers import ASTForAudioClassification, AutoFeatureExtractor


# --- CẤU HÌNH ĐƯỜNG DẪN ---
MODEL_RESNET_PATH = r"C:\CODE\resnet50_finetune_results\resnet50_hybrid_final.pth"
MODEL_AST_DIR = r"C:\CODE\ast_esc50_final_model"
UPLOAD_FOLDER = 'uploads'

app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. ĐỊNH NGHĨA CẤU TRÚC MÔ HÌNH HYBRID (RESNET-50)
# ==========================================
class AudioHybridModel(nn.Module):
    def __init__(self, num_classes=50):
        super(AudioHybridModel, self).__init__()
        # Khởi tạo ResNet50 đúng như trong file Train
        resnet = models.resnet50(weights=None)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        ) 
        encoder_layer = nn.TransformerEncoderLayer(d_model=2048, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        # Theo code train: x = x.unsqueeze(1)
        x = x.unsqueeze(1) 
        x = self.cnn(x)    
        x = x.mean(dim=2).permute(0, 2, 1) 
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))

# ==========================================
# 2. KHỞI TẠO VÀ NẠP MÔ HÌNH
# ==========================================
print("--- Đang nạp hệ thống mô hình đồng bộ ---")
try:
    # Nạp Hybrid Model
    model_hybrid = AudioHybridModel(50).to(device)
    model_hybrid.load_state_dict(torch.load(MODEL_RESNET_PATH, map_location=device))
    model_hybrid.eval()

    # Nạp AST Model
    model_ast = ASTForAudioClassification.from_pretrained(MODEL_AST_DIR).to(device)
    model_ast.eval()
    
    # Nạp chung 1 bộ trích xuất đặc trưng AST cho cả 2 mô hình
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_AST_DIR)
    
    print(f"✅ HỆ THỐNG SẴN SÀNG TRÊN: {device}")
except Exception as e:
    print(f"❌ Lỗi nạp mô hình: {e}")

labels = [
    "airplane", "breathing", "brushing_teeth", "can_opening", "car_horn",
    "cat", "chainsaw", "chirping_birds", "church_bells", "clapping",
    "clock_alarm", "clock_tick", "coughing", "cow", "crackling_fire",
    "crickets", "crow", "crying_baby", "dog", "door_wood_creaks",
    "door_wood_knock", "drinking_sipping", "engine", "fireworks", "footsteps",
    "frog", "glass_breaking", "hand_saw", "helicopter", "hen",
    "insects", "keyboard_typing", "laughing", "mouse_click", "pig",
    "pouring_water", "rain", "rooster", "sea_waves", "sheep",
    "siren", "sneezing", "snoring", "thunderstorm", "toilet_flush",
    "train", "vacuum_cleaner", "washing_machine", "water_drops", "wind"
]

# ==========================================
# 3. TIỀN XỬ LÝ DỮ LIỆU ĐỒNG BỘ
# ==========================================

def common_preprocess(path):
    # Cả hai mô hình của bạn giờ đây đều dùng chung quy trình này
    y, _ = librosa.load(path, sr=16000, duration=5.0)
    y = librosa.util.fix_length(y, size=16000 * 5)
    
    # Sử dụng bộ trích xuất đặc trưng AST cho cả ResNet-Hybrid
    inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt")
    return inputs['input_values'].to(device)

# ==========================================
# 4. ROUTE DỰ ĐOÁN
# ==========================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
        
    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    try:
        start_t = time.time()

        # Tiền xử lý 1 lần cho cả 2 mô hình
        input_values = common_preprocess(file_path)

        with torch.no_grad():
            # 1. Dự đoán với AST
            out_ast = model_ast(input_values).logits
            prob_ast = torch.nn.functional.softmax(out_ast, dim=1)
            
            # 2. Dự đoán với Hybrid (ResNet50 + Transformer)
            out_hybrid = model_hybrid(input_values)
            prob_hybrid = torch.nn.functional.softmax(out_hybrid, dim=1)
            
        print(f"⏱️ Dự đoán hoàn tất: {time.time() - start_t:.4f}s")
        
        return jsonify({
            "model_ast": { 
                "label": labels[torch.argmax(prob_ast).item()], 
                "conf": round(float(torch.max(prob_ast)) * 100, 2)
            },
            "model_hybrid": { 
                "label": labels[torch.argmax(prob_hybrid).item()], 
                "conf": round(float(torch.max(prob_hybrid)) * 100, 2)
            }
        })
    except Exception as e:
        print(f"🔥 Lỗi: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000)