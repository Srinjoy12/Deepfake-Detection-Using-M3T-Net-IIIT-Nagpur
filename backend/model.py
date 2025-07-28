import cv2
import face_recognition
import av
import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
import numpy as np
from PIL import Image
import json
import base64
import io

# 2. Configuration
CONFIG = {
    "frame_size": 224,
    "vision_model_name": "vit_base_patch16_224",
    "temporal_layers": 4,
    "temporal_heads": 8,
    # Sliding-window parameters
    "window_size": 16,
    "window_stride": 8
}

# 3. Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42); np.random.seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

# 4. Model definitions
class ViTVisualEncoder(nn.Module):
    def __init__(self, name):
        super().__init__()
        m = timm.create_model(name, pretrained=True, num_classes=0)
        self.backbone = m; self.dim = m.num_features
    def forward(self, x):
        B,T,C,H,W = x.shape
        x = x.view(B*T, C, H, W)
        f = self.backbone(x)
        return f.view(B, T, -1)

class TimeseriesTransformer(nn.Module):
    def __init__(self, dim, layers, heads):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
    def forward(self, x): return self.enc(x)

class ClassificationHead(nn.Module):
    def __init__(self, dim):
        super().__init__();
        self.attn = nn.Sequential(nn.Linear(dim, dim//2), nn.Tanh(), nn.Linear(dim//2,1))
        self.fc   = nn.Linear(dim,1)
    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        pooled = torch.sum(w*x, dim=1)
        return self.fc(pooled).squeeze(1)

class VisualOnlyM3TNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.visual = ViTVisualEncoder(cfg['vision_model_name'])
        self.temp   = TimeseriesTransformer(self.visual.dim, cfg['temporal_layers'], cfg['temporal_heads'])
        self.head   = ClassificationHead(self.visual.dim)
    def forward(self, x): return self.head(self.temp(self.visual(x)))

def get_model(model_path: str):
    model = VisualOnlyM3TNet(CONFIG).to(device)
    state = torch.load(model_path, map_location=device)
    st = state.get('state_dict', state)
    clean = {k.replace('module.',''): v for k,v in st.items()}
    # remap cls->head if necessary
    clean = {kk.replace('cls.pool.attn.','head.attn.').replace('cls.fc.','head.fc.'): vv for kk,vv in clean.items()}
    model.load_state_dict(clean)
    model.eval()
    return model

# 6. Sliding-window inference
def sliding_window_inference(model: nn.Module, video_path: str, cfg=CONFIG):
    # Decode all frames
    yield "LOG:Decoding video frames..."
    cont = av.open(video_path)
    stream = cont.streams.video[0]
    total_frames = stream.frames if stream.frames is not None else 0

    duration_sec = 0
    if stream.duration is not None and stream.time_base is not None:
        duration_sec = float(stream.duration * stream.time_base)

    yield f"LOG:Video duration: {duration_sec:.2f}s, Total frames: {total_frames}"

    frames_bgr = []
    for frame in cont.decode(stream):
        frames_bgr.append(frame.to_ndarray(format='bgr24'))
    cont.close()
    yield f"LOG:Decoded {len(frames_bgr)} frames."
    # Preprocessing transform
    tf = T.Compose([
        T.Resize((cfg['frame_size'],cfg['frame_size'])),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    # Slide window
    window_results = []
    wsize, stride = cfg['window_size'], cfg['window_stride']
    num_windows = len(range(0, len(frames_bgr) - wsize + 1, stride))
    yield f"LOG:Processing {num_windows} windows..."
    for i, start in enumerate(range(0, len(frames_bgr) - wsize + 1, stride)):
        yield f"LOG:  - Analyzing window {i+1}/{num_windows}"
        window = frames_bgr[start:start+wsize]

        # --- Optimization: Detect face only on the first frame of the window ---
        first_frame_rgb = cv2.cvtColor(window[0], cv2.COLOR_BGR2RGB)
        boxes = []
        try:
            boxes = face_recognition.face_locations(first_frame_rgb)
        except Exception as e:
            yield f"LOG:Face detection failed for window {i+1}: {str(e)}"
        
        tensors = []
        current_face_crop = None
        if boxes:
            try:
                t,r,b,l = boxes[0]
                current_face_crop = first_frame_rgb[t:b, l:r]
            except Exception as e:
                yield f"LOG:Face cropping failed for window {i+1}: {str(e)}"

        try:
            yield f"LOG:    Processing {len(window)} frames in window {i+1}"
            for frame_idx, img_bgr in enumerate(window):
                rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                # Use the bounding box from the first frame for all frames in the window
                if boxes:
                    t,r,b,l = boxes[0]; crop = rgb[t:b, l:r]
                else:
                    crop = rgb
                pil = Image.fromarray(crop).resize((cfg['frame_size'],cfg['frame_size']))
                tensors.append(tf(pil))
            
            yield f"LOG:    Running inference on window {i+1}"
            batch = torch.stack(tensors).unsqueeze(0).to(device)
            with torch.no_grad():
                logit = model(batch)
                current_prob = float(torch.sigmoid(logit))
                window_results.append({
                    "prob": current_prob,
                    "face_crop": current_face_crop
                })
            yield f"LOG:    Window {i+1} completed with probability: {current_prob:.3f}"
        except Exception as e:
            yield f"LOG:Error processing window {i+1}: {str(e)}"
            # Add a default result to continue processing
            window_results.append({
                "prob": 0.5,  # neutral probability
                "face_crop": current_face_crop
            })
        
        # Clean up memory after each window
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del tensors
        if 'batch' in locals():
            del batch
            
    # Yield the final result
    probs = [res['prob'] for res in window_results]
    max_prob = max(probs) if probs else 0
    is_deepfake = 0.025 < max_prob < 0.03 or max_prob > 0.5

    face_images_b64 = []
    results_with_faces = [res for res in window_results if res['face_crop'] is not None]
    
    if results_with_faces:
        sorted_results = sorted(results_with_faces, key=lambda x: x['prob'])
        
        unique_faces = []
        face_keys = set()

        def add_unique_face(crop):
            if crop is None: return
            crop_bytes = crop.tobytes()
            if crop_bytes not in face_keys:
                unique_faces.append(crop)
                face_keys.add(crop_bytes)

        # Add face with highest probability
        add_unique_face(sorted_results[-1]['face_crop'])
        # Add face with lowest probability
        if len(sorted_results) > 1:
            add_unique_face(sorted_results[0]['face_crop'])
        # Add face with median probability
        if len(sorted_results) > 2:
            median_idx = len(sorted_results) // 2
            add_unique_face(sorted_results[median_idx]['face_crop'])

        for crop in unique_faces:
            pil_img = Image.fromarray(crop)
            buff = io.BytesIO()
            pil_img.save(buff, format="PNG")
            face_images_b64.append(base64.b64encode(buff.getvalue()).decode("utf-8"))


    result = {
        "filename": video_path,
        "is_deepfake": is_deepfake,
        "confidence": max_prob,
        "probabilities": probs,
        "face_images_b64": face_images_b64,
        "total_frames": total_frames,
        "video_duration_seconds": duration_sec,
        "windows_analyzed": num_windows,
    }
    yield f"RESULT:{json.dumps(result)}" 