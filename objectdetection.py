import cv2
import numpy as np
import onnxruntime as ort
import json
import base64
import time
import os
from fastapi import FastAPI, Body
from ultralytics import YOLO
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

# Assuming your gemini.py has get_gemini_analysis(image, json_data)
from gemini import get_gemini_analysis 

load_dotenv()
ELEVEN_KEY = os.getenv("ELEVEN_API_KEY")

from gemini import get_gemini_analysis 

# ==========================================
# --- MODE SELECTOR ---
# Set to True for Phone/API, False for Local Video
RUN_LIVE = False 
TEST_VIDEO_PATH = "file.mp4"
# ==========================================

app = FastAPI()

# --- 1. CONFIGURATION & INITIALIZATION ---
providers = ['CPUExecutionProvider'] 
yolo_model = YOLO("yolov8s.onnx", task='detect')
depth_session = ort.InferenceSession("midas_small.onnx", providers=providers)
el_client = ElevenLabs(api_key=ELEVEN_KEY)

# State Management
last_gemini_time = 0
cooldown_seconds = 5
current_gemini_statement = ""
# Track if the last alert was high priority to avoid redundant "Emergency" loops
last_priority_level = 0 

# --- 2. VISION LOGIC ---
def get_depth_map(img_bgr):
    img_input = cv2.resize(img_bgr, (256, 256))
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    img_input = img_input.astype(np.float32) / 255.0
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

# --- 3. BROADCAST ENDPOINT ---
@app.post("/detect")
async def detect_broadcast(payload: dict = Body(...)):
    global last_gemini_time, current_gemini_statement, last_priority_level
    
    img_bytes = base64.b64decode(payload['image'])
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    frame_h, frame_w = frame.shape[:2]

    depth_map = get_depth_map(frame)
    depth_text, text_color, score = analyze_obstacle_density(depth_map)
    yolo_results = yolo_model.predict(source=frame, conf=0.4, verbose=False)
    
    frame_data = {
        "navigation": {"status": depth_text, "danger_score": round(score, 3)},
        "objects": [],
        "resolution": {"w": frame_w, "h": frame_h}
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
    
    # DETERMINE IF WE SHOULD OVERRIDE COOLDOWN
    # Priority 2: Immediate Danger (>0.8)
    # Priority 1: Warning (>0.5)
    current_priority = 2 if score > 0.8 else (1 if score > 0.5 else 0)
    
    # Logic: Trigger if (Cooldown expired AND danger > 0.5) OR (Critical danger just appeared)
    should_trigger = (now - last_gemini_time > cooldown_seconds and current_priority > 0) or \
                     (current_priority == 2 and last_priority_level < 2)

    if should_trigger:
        if current_priority == 2:
            interrupt_current_audio = True # Tell phone to kill existing audio
        
        last_gemini_time = now
        last_priority_level = current_priority
        print(f" [!] Fetching {'PRIORITY ' if interrupt_current_audio else ''}Gemini Advice...")
        
        try:
            current_gemini_statement = get_gemini_analysis(frame, json.dumps(frame_data))
            
            audio_stream = el_client.text_to_speech.convert(
                text=current_gemini_statement,
                voice_id="pNInz6obpgDQGcFmaJgB",
                model_id="eleven_turbo_v2_5"
            )
            
            audio_content = b"".join(list(audio_stream))
            audio_b64 = base64.b64encode(audio_content).decode('utf-8')
            
        except Exception as e:
            print(f" Gemini/Voice Error: {e}")
    
    # Reset priority level if path is clear
    if current_priority == 0:
        last_priority_level = 0

    # PC Visual Debugging
    depth_viz = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    depth_color_res = cv2.resize(depth_viz, (frame_w, frame_h))
    combined_view = np.hstack((annotated_frame, depth_color_res))
    
    if current_gemini_statement:
        cv2.putText(combined_view, current_gemini_statement[:75], (50, frame_h - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("PickHacks: YOLO + MiDaS + Gemini", combined_view)
    cv2.waitKey(1)

    return {
        "navigation": frame_data["navigation"],
        "objects": frame_data["objects"],
        "advice": current_gemini_statement,
        "audio": audio_b64
    }

# --- 4. TEST MODE RUNNER ---
def run_test_mode():
    cap = cv2.VideoCapture(TEST_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open {TEST_VIDEO_PATH}")
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
            continue
        
        process_frame(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()

# --- 5. MAIN ENTRY ---
if __name__ == "__main__":
    import uvicorn
    if RUN_LIVE:
        print("[!] Starting in LIVE MODE (FastAPI)")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print(f"[!] Starting in TEST MODE (File: {TEST_VIDEO_PATH})")
        run_test_mode()