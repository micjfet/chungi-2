import os
import cv2
import numpy as np
import onnxruntime as ort
import json
import base64
import time
import threading
from dotenv import load_dotenv
from fastapi import FastAPI, Body
from ultralytics import YOLO
from elevenlabs.client import ElevenLabs

# 1. LOAD ENVIRONMENT
load_dotenv()

# 2. IMPORT GEMINI WRAPPER
from gemini import get_gemini_analysis

# ==========================================
# --- CONFIGURATION & MODE SELECTOR ---
# ==========================================
RUN_LIVE = False  # Set to False to run the TEST_VIDEO_PATH locally
TEST_VIDEO_PATH = "file.mp4"
ELEVEN_KEY = os.getenv("ELEVEN_API_KEY")

# --- LANGUAGE SETTINGS ---
# Change this variable to update the system-wide language (e.g., "es", "fr", "zh")
CURRENT_LANGUAGE = "ja" 

LANGUAGE_CONFIGS = {
    "en": {"name": "English", "model_id": "eleven_turbo_v2_5"},
    "es": {"name": "Spanish", "model_id": "eleven_turbo_v2_5"},
    "fr": {"name": "French", "model_id": "eleven_turbo_v2_5"},
    "vi": {"name": "Vietnamese", "model_id": "eleven_turbo_v2_5"},
    "ja": {"name": "Japanese", "model_id": "eleven_turbo_v2_5"},
    "zh": {"name": "Chinese", "model_id": "eleven_turbo_v2_5"}
}

STATUS_TRANSLATIONS = {
    "en": {"danger": "IMMEDIATE DANGER", "warn": "WARNING: Obstacle", "clear": "PATH CLEAR"},
    "es": {"danger": "PELIGRO INMEDIATO", "warn": "ADVERTENCIA: Obstáculo", "clear": "CAMINO DESPEJADO"},
    "fr": {"danger": "DANGER IMMÉDIAT", "warn": "ATTENTION : Obstacle", "clear": "VOIE LIBRE"},
    "vi": {"danger": "NGUY HIỂM LẬP TỨC", "warn": "CẢNH BÁO: Vật cản", "clear": "ĐƯỜNG TRỐNG"},
    "ja": {"danger": "直ちに危険", "warn": "警告：障害物", "clear": "道は開いています"},
    "zh": {"danger": "立即危险", "warn": "警告：有障碍物", "clear": "道路畅通"}
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

# --- 2. UPDATED VISION LOGIC ---
def analyze_obstacle_density(depth_norm):
    h, w = depth_norm.shape
    center_strip = depth_norm[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
    danger_score = float(np.percentile(center_strip, 90))
    
    # Return a status KEY and the color, rather than a hardcoded string
    if danger_score > 0.8: 
        return "danger", (0, 0, 255), danger_score
    elif danger_score > 0.5: 
        return "warn", (0, 165, 255), danger_score
    
    return "clear", (0, 255, 0), danger_score

# --- 3. UPDATED CORE PROCESSING ENGINE ---
def process_frame(frame, lang_code=None):
    global last_gemini_time, current_gemini_statement, last_priority_level
    
    # Use global language if none provided by API
    if lang_code is None:
        lang_code = CURRENT_LANGUAGE
        
    frame_h, frame_w = frame.shape[:2]
    lang_info = LANGUAGE_CONFIGS.get(lang_code, LANGUAGE_CONFIGS["en"])

    # Vision Processing
    depth_map = get_depth_map(frame)
    status_key, text_color, score = analyze_obstacle_density(depth_map)
    yolo_results = yolo_model.predict(source=frame, conf=0.4, verbose=False)
    
    # Translation Logic: Look up the string based on the current language
    translations = STATUS_TRANSLATIONS.get(lang_code, STATUS_TRANSLATIONS["en"])
    display_status = translations.get(status_key, "UNKNOWN")
    
    frame_data = {
        "navigation": {"status": display_status, "danger_score": round(score, 3)},
        "objects": [],
        "resolution": {"w": frame_w, "h": frame_h},
        "target_language": lang_info["name"]
    }

    # UI Rendering
    annotated_frame = yolo_results[0].plot(line_width=2)
    depth_viz = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    depth_viz = cv2.resize(depth_viz, (frame_w, frame_h))
    
    combined_view = np.hstack((annotated_frame, depth_viz))
    
    # Scaling for Screen
    MAX_WINDOW_WIDTH = 1280
    orig_h, orig_w = combined_view.shape[:2]
    scale_factor = MAX_WINDOW_WIDTH / orig_w if orig_w > MAX_WINDOW_WIDTH else 1.0
    new_w, new_h = int(orig_w * scale_factor), int(orig_h * scale_factor)
    display_view = cv2.resize(combined_view, (new_w, new_h))

    # HUD Overlay on Python Dashboard
    overlay = display_view.copy()
    cv2.rectangle(overlay, (0, new_h - 50), (new_w, new_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display_view, 0.4, 0, display_view)

    # Note: Using the translated display_status here too!
    cv2.putText(display_view, f"LANG: {lang_info['name'].upper()}", (15, new_h - 18), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(display_view, f"NAV: {display_status}", (int(new_w/2) + 15, new_h - 18), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    cv2.imshow("Vision Intelligence Dashboard", display_view)
    cv2.waitKey(1) 

    # Gemini Triggering Logic
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
        
        for r in yolo_results:
            for box in r.boxes:
                label = yolo_model.names[int(box.cls[0])]
                coords = box.xyxy[0].tolist()
                center_x = (coords[0] + coords[2]) / 2
                position = "Center" if frame_w*0.33 < center_x < frame_w*0.66 else ("Left" if center_x < frame_w*0.33 else "Right")
                frame_data["objects"].append({"label": label, "position": position})

        try:
            current_gemini_statement = get_gemini_analysis(frame, json.dumps(frame_data))
            audio_stream = el_client.text_to_speech.convert(
                text=current_gemini_statement,
                voice_id="pNInz6obpgDQGcFmaJgB",
                model_id=lang_info["model_id"]
            )
            audio_content = b"".join(list(audio_stream))
            audio_b64 = base64.b64encode(audio_content).decode('utf-8')
        except Exception as e:
            print(f" Gemini/Voice Error: {e}")

    if current_priority == 0: last_priority_level = 0
    return frame_data, audio_b64, interrupt_current_audio

# --- 4. API ENDPOINTS ---
@app.post("/detect")
async def detect_broadcast(payload: dict = Body(...)):
    global pending_remote_audio, pending_remote_text
    
    img_bytes = base64.b64decode(payload['image'])
    lang_code = payload.get('lang', CURRENT_LANGUAGE) # Defaults to CURRENT_LANGUAGE
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
    lang_code = payload.get("lang", CURRENT_LANGUAGE)
    
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

@app.post("/language")
async def update_language(payload: dict = Body(...)):
    global CURRENT_LANGUAGE
    new_lang = payload.get("lang", "en")
    
    if new_lang in LANGUAGE_CONFIGS:
        CURRENT_LANGUAGE = new_lang
        lang_name = LANGUAGE_CONFIGS[new_lang]["name"]
        print(f"[!] System Language updated to: {lang_name}")
        return {"status": "success", "language": lang_name}
    
    return {"status": "error", "message": "Invalid language code"}, 400

# --- 5. EXECUTION ---
if __name__ == "__main__":
    import uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    api_thread = threading.Thread(target=server.run, daemon=True)
    api_thread.start()

    if RUN_LIVE:
        print(f"[!] SERVER RUNNING. Language set to: {CURRENT_LANGUAGE}")
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt: pass
    else:
        print(f"[!] TEST MODE: Loading {TEST_VIDEO_PATH}")
        cap = cv2.VideoCapture(TEST_VIDEO_PATH)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            # Passes CURRENT_LANGUAGE variable
            process_frame(frame, CURRENT_LANGUAGE)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()