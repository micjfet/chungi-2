import os
from dotenv import load_dotenv

# 1. LOAD THE ENVIRONMENT FIRST
load_dotenv() 

# 2. NOW IMPORT EVERYTHING ELSE
import cv2
import numpy as np
import onnxruntime as ort
import json
import base64
import time
import threading
from fastapi import FastAPI, Body
from ultralytics import YOLO
from elevenlabs.client import ElevenLabs

# 3. IMPORT GEMINI LAST
from gemini import get_gemini_analysis

# ==========================================
# --- CONFIGURATION & MULTILINGUAL SETUP ---
# ==========================================
RUN_LIVE = True
ELEVEN_KEY = os.getenv("ELEVEN_API_KEY")

# Mapping for supported languages
LANGUAGE_CONFIGS = {
    "en": {"name": "English", "model_id": "eleven_turbo_v2_5"},
    "es": {"name": "Spanish", "model_id": "eleven_turbo_v2_5"},
    "fr": {"name": "French", "model_id": "eleven_turbo_v2_5"},
    "vi": {"name": "Vietnamese", "model_id": "eleven_turbo_v2_5"},
    "ja": {"name": "Japanese", "model_id": "eleven_turbo_v2_5"},
    "zh": {"name": "Chinese", "model_id": "eleven_turbo_v2_5"}
}

app = FastAPI()

# --- 1. INITIALIZATION ---
providers = ['CPUExecutionProvider'] 
yolo_model = YOLO("yolov8s.onnx", task='detect')
depth_session = ort.InferenceSession("midas_small.onnx", providers=providers)
el_client = ElevenLabs(api_key=ELEVEN_KEY)

# State Management
last_gemini_time = 0
cooldown_seconds = 5
current_gemini_statement = "System Initialized."
last_priority_level = 0 
pending_remote_audio = None
pending_remote_text = ""

# --- 2. VISION LOGIC ---
def get_depth_map(img_bgr):
    img_input = cv2.resize(img_bgr, (256, 256))
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img_input = (img_input - mean) / std
    img_input = img_input.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)
    onnx_input = {depth_session.get_inputs()[0].name: img_input}
    depth_output = depth_session.run(None, onnx_input)[0]
    depth_map = np.squeeze(depth_output)
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    return depth_norm

def analyze_obstacle_density(depth_norm):
    h, w = depth_norm.shape
    center_strip = depth_norm[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
    danger_score = float(np.percentile(center_strip, 90))
    if danger_score > 0.8: return "IMMEDIATE DANGER", (0, 0, 255), danger_score
    elif danger_score > 0.5: return "WARNING", (0, 165, 255), danger_score
    return "CLEAR", (0, 255, 0), danger_score

# --- 3. CORE PROCESSING ENGINE ---
def process_frame(frame, lang_code="en"):
    global last_gemini_time, current_gemini_statement, last_priority_level
    frame_h, frame_w = frame.shape[:2]
    
    lang_info = LANGUAGE_CONFIGS.get(lang_code, LANGUAGE_CONFIGS["es"])

    depth_map = get_depth_map(frame)
    depth_text, text_color, score = analyze_obstacle_density(depth_map)
    yolo_results = yolo_model.predict(source=frame, conf=0.4, verbose=False)
    
    frame_data = {
        "navigation": {"status": depth_text, "danger_score": round(score, 3)},
        "objects": [],
        "resolution": {"w": frame_w, "h": frame_h},
        "target_language": lang_info["name"]
    }

    annotated_frame = yolo_results[0].plot()
    for r in yolo_results:
        for box in r.boxes:
            label = yolo_model.names[int(box.cls[0])]
            coords = box.xyxy[0].tolist()
            center_x = (coords[0] + coords[2]) / 2
            position = "Center" if frame_w*0.33 < center_x < frame_w*0.66 else ("Left" if center_x < frame_w*0.33 else "Right")
            frame_data["objects"].append({"label": label, "position": position})

    audio_b64 = None
    interrupt_current_audio = False
    now = time.time()
    
    current_priority = 2 if score > 0.8 else (1 if score > 0.5 else 0)
    should_trigger = (now - last_gemini_time > cooldown_seconds and current_priority > 0) or \
                     (current_priority == 2 and last_priority_level < 2)

    if should_trigger:
        if current_priority == 2: interrupt_current_audio = True
        last_gemini_time = now
        last_priority_level = current_priority
        
        try:
            # We pass the target language in the JSON data to Gemini
            current_gemini_statement = get_gemini_analysis(frame, json.dumps(frame_data))
            
            audio_stream = el_client.text_to_speech.convert(
                text=current_gemini_statement,
                voice_id="pNInz6obpgDQGcFmaJgB",
                model_id=lang_info["model_id"] # Use multilingual model
            )
            audio_content = b"".join(list(audio_stream))
            audio_b64 = base64.b64encode(audio_content).decode('utf-8')
        except Exception as e:
            print(f" Gemini/Voice Error: {e}")

    if current_priority == 0: last_priority_level = 0

    # Optional: Logic to show labels on your desktop preview
    cv2.imshow("Vision Dashboard", annotated_frame)
    cv2.waitKey(1) 

    return frame_data, audio_b64, interrupt_current_audio

# --- 4. API ENDPOINTS ---
@app.post("/detect")
async def detect_broadcast(payload: dict = Body(...)):
    global pending_remote_audio, pending_remote_text
    
    img_bytes = base64.b64decode(payload['image'])
    lang_code = payload.get('lang', 'en')
    
    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    frame_data, gemini_audio, interrupt = process_frame(frame, lang_code)
    
    final_audio = gemini_audio
    final_advice = current_gemini_statement
    
    if pending_remote_audio:
        final_audio = pending_remote_audio
        final_advice = f"[REMOTE]: {pending_remote_text}"
        interrupt = True
        pending_remote_audio = None
        pending_remote_text = ""
    
    return {
        "navigation": frame_data["navigation"],
        "objects": frame_data["objects"],
        "advice": final_advice,
        "audio": final_audio,
        "interrupt": interrupt
    }

@app.post("/message")
async def receive_message(payload: dict = Body(...)):
    global pending_remote_audio, pending_remote_text
    msg = payload.get("message", "")
    lang_code = payload.get("lang", "en")
    
    if msg:
        lang_info = LANGUAGE_CONFIGS.get(lang_code, LANGUAGE_CONFIGS["en"])
        try:
            audio_stream = el_client.text_to_speech.convert(
                text=msg,
                voice_id="pNInz6obpgDQGcFmaJgB",
                model_id=lang_info["model_id"]
            )
            audio_content = b"".join(list(audio_stream))
            pending_remote_text = msg
            pending_remote_audio = base64.b64encode(audio_content).decode('utf-8')
            return {"status": f"Queued in {lang_info['name']}"}
        except Exception as e:
            return {"status": f"ElevenLabs Error: {e}"}, 500
            
    return {"status": "Empty message"}, 400

if __name__ == "__main__":
    import uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    server.run()