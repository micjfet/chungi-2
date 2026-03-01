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
# --- CONFIGURATION & GLOBAL STATE ---
# ==========================================
RUN_LIVE = True
ELEVEN_KEY = os.getenv("ELEVEN_API_KEY")

current_language = "es" 

LANGUAGE_CONFIGS = {
    "en": {"name": "English", "model_id": "eleven_turbo_v2_5"},
    "es": {"name": "Spanish", "model_id": "eleven_turbo_v2_5"},
    "fr": {"name": "French", "model_id": "eleven_turbo_v2_5"},
    "vi": {"name": "Vietnamese", "model_id": "eleven_turbo_v2_5"},
    "ja": {"name": "Japanese", "model_id": "eleven_turbo_v2_5"},
    "zh": {"name": "Chinese", "model_id": "eleven_turbo_v2_5"}
}

app = FastAPI()

# --- INITIALIZATION ---
yolo_model = YOLO("yolov8s.onnx", task='detect')
depth_session = ort.InferenceSession("midas_small.onnx", providers=['CPUExecutionProvider'])
el_client = ElevenLabs(api_key=ELEVEN_KEY)

# State Management
last_gemini_time = 0
cooldown_seconds = 5
current_gemini_statement = "System Initialized."
last_priority_level = 0 
pending_remote_audio = None
pending_remote_text = ""

# --- VISION LOGIC ---
def get_depth_map(img_bgr):
    img_input = cv2.resize(img_bgr, (256, 256))
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img_input = (img_input - mean) / std
    img_input = img_input.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)
    onnx_input = {depth_session.get_inputs()[0].name: img_input}
    depth_output = depth_session.run(None, onnx_input)[0]
    depth_map = np.squeeze(depth_output)
    return (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

def analyze_obstacle_density(depth_norm):
    h, w = depth_norm.shape
    center_strip = depth_norm[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
    danger_score = float(np.percentile(center_strip, 90))
    if danger_score > 0.8: return "DANGER", danger_score
    return "CLEAR", danger_score

# --- CORE ENGINE ---
def process_frame(frame, lang_code=None):
    global last_gemini_time, current_gemini_statement, last_priority_level, current_language
    
    active_lang = lang_code if lang_code else current_language
    lang_info = LANGUAGE_CONFIGS.get(active_lang, LANGUAGE_CONFIGS["en"])

    depth_map = get_depth_map(frame)
    _, score = analyze_obstacle_density(depth_map)
    yolo_results = yolo_model.predict(source=frame, conf=0.4, verbose=False)
    
    frame_data = {
        "navigation": {"danger_score": round(score, 3)},
        "objects": [],
        "target_language": lang_info["name"]
    }

    # YOLO Logic
    annotated_frame = yolo_results[0].plot()
    for r in yolo_results:
        for box in r.boxes:
            label = yolo_model.names[int(box.cls[0])]
            frame_data["objects"].append({"label": label})

    audio_b64 = None
    interrupt = False
    now = time.time()
    
    current_priority = 2 if score > 0.8 else (1 if score > 0.5 else 0)
    should_trigger = (now - last_gemini_time > cooldown_seconds and current_priority > 0) or \
                     (current_priority == 2 and last_priority_level < 2)

    if should_trigger:
        if current_priority == 2: interrupt = True
        last_gemini_time = now
        last_priority_level = current_priority
        
        try:
            current_gemini_statement = get_gemini_analysis(frame, json.dumps(frame_data))
            audio_stream = el_client.text_to_speech.convert(
                text=current_gemini_statement,
                voice_id="pNInz6obpgDQGcFmaJgB",
                model_id=lang_info["model_id"]
            )
            audio_b64 = base64.b64encode(b"".join(list(audio_stream))).decode('utf-8')
        except Exception as e:
            print(f"Error: {e}")

    cv2.imshow("Vision Dashboard", annotated_frame)
    cv2.waitKey(1) 
    return frame_data, audio_b64, interrupt

# --- API ENDPOINTS ---

@app.post("/settings")
async def update_settings(payload: dict = Body(...)):
    """Live language switching endpoint"""
    global current_language
    new_lang = payload.get("lang")
    if new_lang in LANGUAGE_CONFIGS:
        current_language = new_lang
        return {"status": "updated", "system_language": LANGUAGE_CONFIGS[new_lang]["name"]}
    return {"status": "error", "message": "Invalid language code"}, 400

@app.post("/detect")
async def detect_broadcast(payload: dict = Body(...)):
    global current_language, pending_remote_audio, pending_remote_text
    
    img_bytes = base64.b64decode(payload['image'])
    # Allows individual requests to override system language if desired
    lang_code = payload.get('lang', current_language)
    
    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    frame_data, gemini_audio, interrupt = process_frame(frame, lang_code)
    
    final_audio = pending_remote_audio if pending_remote_audio else gemini_audio
    final_advice = f"[REMOTE]: {pending_remote_text}" if pending_remote_text else current_gemini_statement
    
    if pending_remote_audio:
        interrupt = True
        pending_remote_audio, pending_remote_text = None, ""
    
    return {"advice": final_advice, "audio": final_audio, "interrupt": interrupt}

@app.post("/message")
async def receive_message(payload: dict = Body(...)):
    global current_language, pending_remote_audio, pending_remote_text
    msg = payload.get("message", "")
    lang_code = payload.get("lang", current_language)
    
    if msg:
        lang_info = LANGUAGE_CONFIGS.get(lang_code, LANGUAGE_CONFIGS[current_language])
        audio_stream = el_client.text_to_speech.convert(
            text=msg, voice_id="pNInz6obpgDQGcFmaJgB", model_id=lang_info["model_id"]
        )
        pending_remote_text = msg
        pending_remote_audio = base64.b64encode(b"".join(list(audio_stream))).decode('utf-8')
        return {"status": "queued"}
    return {"status": "empty"}, 400

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)