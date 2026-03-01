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
# --- MODE SELECTOR ---
RUN_LIVE = True
TEST_VIDEO_PATH = "file.mp4"
# ==========================================

ELEVEN_KEY = os.getenv("ELEVEN_API_KEY")

app = FastAPI()

# --- 1. CONFIGURATION & INITIALIZATION ---
providers = ['CPUExecutionProvider'] 
yolo_model = YOLO("yolov8s.onnx", task='detect')
depth_session = ort.InferenceSession("midas_small.onnx", providers=providers)
el_client = ElevenLabs(api_key=ELEVEN_KEY)

# State Management
last_gemini_time = 0
cooldown_seconds = 5
current_gemini_statement = "System Initialized."
last_priority_level = 0 

# NEW: Bridge for Remote Messages
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

# --- 3. CORE PROCESSING ENGINE ---
def process_frame(frame):
    global last_gemini_time, current_gemini_statement, last_priority_level
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
    
    current_priority = 2 if score > 0.8 else (1 if score > 0.5 else 0)
    should_trigger = (now - last_gemini_time > cooldown_seconds and current_priority > 0) or \
                     (current_priority == 2 and last_priority_level < 2)

    if should_trigger:
        if current_priority == 2: interrupt_current_audio = True
        last_gemini_time = now
        last_priority_level = current_priority
        
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

    if current_priority == 0: last_priority_level = 0

    # UI Rendering
    depth_viz = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    depth_color_res = cv2.resize(depth_viz, (frame_w, frame_h))
    combined_view = np.hstack((annotated_frame, depth_color_res))
    cv2.putText(combined_view, "LIVE MODE" if RUN_LIVE else "TEST MODE", (20, 30), 1, 1.5, (0, 255, 255), 2)
    
    cv2.imshow("Vision Dashboard", combined_view)
    cv2.waitKey(1) 

    return frame_data, audio_b64, interrupt_current_audio

# --- 4. API ENDPOINTS ---
@app.post("/detect")
async def detect_broadcast(payload: dict = Body(...)):
    global pending_remote_audio, pending_remote_text
    
    img_bytes = base64.b64decode(payload['image'])
    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    # Process the frame normally
    frame_data, gemini_audio, interrupt = process_frame(frame)
    
    # CHECK IF A REMOTE MESSAGE IS WAITING
    # If there is a curl message, we override the Gemini audio to play the emergency msg
    final_audio = gemini_audio
    final_advice = current_gemini_statement
    
    if pending_remote_audio:
        final_audio = pending_remote_audio
        final_advice = f"[REMOTE]: {pending_remote_text}"
        interrupt = True # Force phone to stop talking and play this
        
        # Clear the queue so it only plays once
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
    
    if msg:
        print(f"\n[QUEUEING EMERGENCY MESSAGE]: {msg}")
        try:
            # Generate the audio and store it for the phone to pick up
            audio_stream = el_client.text_to_speech.convert(
                text=msg,
                voice_id="pNInz6obpgDQGcFmaJgB",
                model_id="eleven_turbo_v2_5"
            )
            audio_content = b"".join(list(audio_stream))
            
            pending_remote_text = msg
            pending_remote_audio = base64.b64encode(audio_content).decode('utf-8')
            
            return {"status": "Message queued for next phone update"}
        except Exception as e:
            return {"status": f"ElevenLabs Error: {e}"}, 500
            
    return {"status": "Empty message"}, 400

# --- 5. EXECUTION ---
if __name__ == "__main__":
    import uvicorn
    
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    api_thread = threading.Thread(target=server.run, daemon=True)
    api_thread.start()

    if RUN_LIVE:
        print("[!] SERVER RUNNING. Use the Expo App to connect.")
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        # (Test mode code omitted for brevity)
        pass