import cv2
import numpy as np
import onnxruntime as ort
import json
import base64
import time
import os
from fastapi import FastAPI, Body
from ultralytics import YOLO
from elevenlabs.client import ElevenLabs # Modern 2026 SDK
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

ELEVEN_KEY = os.getenv("ELEVEN_API_KEY")


# Assuming your gemini.py has get_gemini_analysis(image, json_data)
from gemini import get_gemini_analysis 

app = FastAPI()

# --- 1. CONFIGURATION & INITIALIZATION ---
providers = ['CPUExecutionProvider'] 
yolo_model = YOLO("yolov8s.onnx", task='detect')
depth_session = ort.InferenceSession("midas_small.onnx", providers=providers)

# ElevenLabs Setup
el_client = ElevenLabs(api_key=ELEVEN_KEY)

# Gemini Throttle State
last_gemini_time = 0
cooldown_seconds = 5
current_gemini_statement = ""

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
    danger_score = float(np.percentile(center_strip, 90)) # Your high-accuracy logic
    if danger_score > 0.8: return "IMMEDIATE DANGER", (0, 0, 255), danger_score
    elif danger_score > 0.5: return "WARNING: Obstacle", (0, 165, 255), danger_score
    return "PATH CLEAR", (0, 255, 0), danger_score

# --- 3. BROADCAST ENDPOINT ---
@app.post("/detect")
async def detect_broadcast(payload: dict = Body(...)):
    global last_gemini_time, current_gemini_statement
    
    # A. Decode Phone Image
    img_bytes = base64.b64decode(payload['image'])
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    frame_h, frame_w = frame.shape[:2]

    # B. Run Vision Models
    depth_map = get_depth_map(frame)
    depth_text, text_color, score = analyze_obstacle_density(depth_map)
    yolo_results = yolo_model.predict(source=frame, conf=0.4, verbose=False)
    
    frame_data = {
        "navigation": {"status": depth_text, "danger_score": round(score, 3)},
        "objects": [],
        "resolution": {"w": frame_w, "h": frame_h}
    }

    # C. Process YOLO & Annotated Frame
    annotated_frame = yolo_results[0].plot()
    for r in yolo_results:
        for box in r.boxes:
            label = yolo_model.names[int(box.cls[0])]
            coords = box.xyxy[0].tolist()
            center_x = (coords[0] + coords[2]) / 2
            position = "Center" if frame_w*0.33 < center_x < frame_w*0.66 else ("Left" if center_x < frame_w*0.33 else "Right")
            frame_data["objects"].append({"label": label, "position": position})

    # D. Gemini Advice & ElevenLabs Voice
    audio_b64 = None
    now = time.time()
    
    if score > 0.5 and (now - last_gemini_time) > cooldown_seconds:
        last_gemini_time = now
        print(f" [!] Fetching Gemini Advice...")
        try:
            # Get Gemini text
            current_gemini_statement = get_gemini_analysis(frame, json.dumps(frame_data))
            
            # Convert text to audio bytes
            audio_stream = el_client.text_to_speech.convert(
                text=current_gemini_statement,
                voice_id="pNInz6obpgDQGcFmaJgB", # Adam
                model_id="eleven_turbo_v2_5"
            )
            
            # Collect audio chunks into one Base64 string
            audio_content = b"".join(list(audio_stream))
            audio_b64 = base64.b64encode(audio_content).decode('utf-8')
            
        except Exception as e:
            print(f" Gemini/Voice Error: {e}")

    # E. Visual Debugging on PC
    depth_viz = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    depth_color_res = cv2.resize(depth_viz, (frame_w, frame_h))
    combined_view = np.hstack((annotated_frame, depth_color_res))
    
    if current_gemini_statement:
        cv2.putText(combined_view, current_gemini_statement[:75], (50, frame_h - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("PickHacks: YOLO + MiDaS + Gemini", combined_view)
    cv2.waitKey(1)

    # F. Return Data to Phone
    return {
        "navigation": frame_data["navigation"],
        "objects": frame_data["objects"],
        "advice": current_gemini_statement,
        "audio": audio_b64 # Only present if Gemini was called
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)