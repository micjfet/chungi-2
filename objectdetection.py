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
CURRENT_LANGUAGE = "en" 

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
    elif danger_score > 0.5: return "WARNING: Obstacle", (0, 165, 255), danger_score
    return "PATH CLEAR", (0, 255, 0), danger_score

def process_frame(frame, lang_code=None):
    global last_gemini_time, current_gemini_statement, last_priority_level
    
    # 1. Handle Language Selection
    if lang_code is None:
        lang_code = CURRENT_LANGUAGE
        
    frame_h, frame_w = frame.shape[:2]
    lang_info = LANGUAGE_CONFIGS.get(lang_code, LANGUAGE_CONFIGS["en"])

    # 2. Vision Processing (AI Inference)
    depth_map = get_depth_map(frame)
    depth_text, text_color, score = analyze_obstacle_density(depth_map)
    yolo_results = yolo_model.predict(source=frame, conf=0.4, verbose=False)
    
    # 3. Data Preparation for Gemini
    frame_data = {
        "navigation": {"status": depth_text, "danger_score": round(score, 3)},
        "objects": [],
        "resolution": {"w": frame_w, "h": frame_h},
        "target_language": lang_info["name"]
    }

    # 4. Create Visual Components
    # YOLO Detections
    annotated_frame = yolo_results[0].plot(line_width=2)
    
    # Depth Heatmap (using Magma for better contrast)
    depth_viz = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    depth_viz = cv2.resize(depth_viz, (frame_w, frame_h))
    
    # Stack them side-by-side
    combined_view = np.hstack((annotated_frame, depth_viz))
    
    # --- 5. SCREEN SIZE FIX: Dynamic Scaling ---
    MAX_WINDOW_WIDTH = 1280  # Prevents the window from going off-screen
    orig_h, orig_w = combined_view.shape[:2]
    
    if orig_w > MAX_WINDOW_WIDTH:
        scale_factor = MAX_WINDOW_WIDTH / orig_w
        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)
        display_view = cv2.resize(combined_view, (new_w, new_h))
    else:
        display_view = combined_view.copy()
        new_w, new_h = orig_w, orig_h

    # --- 6. Professional HUD Overlays (Drawn on scaled image) ---
    # Semi-transparent footer bar
    overlay = display_view.copy()
    cv2.rectangle(overlay, (0, new_h - 50), (new_w, new_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display_view, 0.4, 0, display_view)

    # Status Text (Left)
    mode_label = "LIVE" if RUN_LIVE else "TEST"
    cv2.putText(display_view, f"{mode_label} | {lang_info['name'].upper()}", (15, new_h - 18), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Navigation Status (Middle-Right)
    # Positioned at the start of the depth map half
    cv2.putText(display_view, f"NAV: {depth_text}", (int(new_w/2) + 15, new_h - 18), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    # Danger Meter (Vertical Bar on Far Right)
    meter_top = 40
    meter_bottom = new_h - 80
    meter_full_h = meter_bottom - meter_top
    current_meter_h = int(score * meter_full_h)
    
    # Meter Background
    cv2.rectangle(display_view, (new_w - 30, meter_top), (new_w - 15, meter_bottom), (40, 40, 40), -1)
    # Active Meter Level
    cv2.rectangle(display_view, (new_w - 30, meter_bottom - current_meter_h), (new_w - 15, meter_bottom), text_color, -1)

    # 7. Render to Screen
    cv2.imshow("Vision Intelligence Dashboard", display_view)
    cv2.waitKey(1) 

    # --- 8. Gemini Triggering Logic ---
    audio_b64 = None
    interrupt_current_audio = False
    now = time.time()
    
    # Determine Priority (0: Clear, 1: Warning, 2: Immediate Danger)
    current_priority = 2 if score > 0.8 else (1 if score > 0.5 else 0)
    
    # Trigger Gemini if: Cooldown is over AND (Priority > 0 OR Priority just escalated to 2)
    should_trigger = (now - last_gemini_time > cooldown_seconds and current_priority > 0) or \
                     (current_priority == 2 and last_priority_level < 2)

    if should_trigger:
        if current_priority == 2: interrupt_current_audio = True
        last_gemini_time = now
        last_priority_level = current_priority
        
        # Re-populate objects specifically for the Gemini prompt
        for r in yolo_results:
            for box in r.boxes:
                label = yolo_model.names[int(box.cls[0])]
                coords = box.xyxy[0].tolist()
                center_x = (coords[0] + coords[2]) / 2
                position = "Center" if frame_w*0.33 < center_x < frame_w*0.66 else ("Left" if center_x < frame_w*0.33 else "Right")
                frame_data["objects"].append({"label": label, "position": position})

        try:
            # Get analysis from Gemini in the target language
            current_gemini_statement = get_gemini_analysis(frame, json.dumps(frame_data))
            
            # Generate TTS Audio via ElevenLabs
            audio_stream = el_client.text_to_speech.convert(
                text=current_gemini_statement,
                voice_id="pNInz6obpgDQGcFmaJgB",
                model_id=lang_info["model_id"]
            )
            audio_content = b"".join(list(audio_stream))
            audio_b64 = base64.b64encode(audio_content).decode('utf-8')
        except Exception as e:
            print(f" Gemini/Voice Error: {e}")

    # Reset priority tracking if path is clear
    if current_priority == 0: 
        last_priority_level = 0
        
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