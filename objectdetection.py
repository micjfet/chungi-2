import cv2
import numpy as np
import onnxruntime as ort
import json
import base64
import time
import os
import threading
from fastapi import FastAPI, Body
from ultralytics import YOLO
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
ELEVEN_KEY = os.getenv("ELEVEN_API_KEY")

# Assuming your gemini.py has get_gemini_analysis(image, json_data)
from gemini import get_gemini_analysis 

app = FastAPI()

# --- 1. CONFIGURATION & INITIALIZATION ---
providers = ['CPUExecutionProvider'] 
yolo_model = YOLO("yolov8s.onnx", task='detect')
depth_session = ort.InferenceSession("midas_small.onnx", providers=providers)
el_client = ElevenLabs(api_key=ELEVEN_KEY)

# State Management
last_gemini_time = 0
cooldown_seconds = 5
current_gemini_statement = "Initializing..."
last_phone_request_time = 0  # Global "heartbeat" tracker

def analyze_obstacle_density(depth_norm):
    h, w = depth_norm.shape
    # Focus on the 90th percentile of the center strip to catch flat walls
    center_strip = depth_norm[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
    danger_score = float(np.percentile(center_strip, 90))
    
    if danger_score > 0.8: 
        return "IMMEDIATE DANGER", (0, 0, 255), danger_score
    elif danger_score > 0.5: 
        return "WARNING: Obstacle", (0, 165, 255), danger_score
    return "PATH CLEAR", (0, 255, 0), danger_score

# --- 2. THE VISION ENGINE (Shared by Live and Test Mode) ---
def process_frame(frame, is_live=False):
    """Processes a single frame and returns the result/data."""
    global last_gemini_time, current_gemini_statement
    
    frame_h, frame_w = frame.shape[:2]

    # A. Depth Calculation
    img_input = cv2.resize(frame, (256, 256))
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img_input = (img_input - mean) / std
    img_input = img_input.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)
    
    onnx_input = {depth_session.get_inputs()[0].name: img_input}
    depth_output = depth_session.run(None, onnx_input)[0]
    depth_map = np.squeeze(depth_output)
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    # B. Danger Analysis
    depth_text, text_color, score = analyze_obstacle_density(depth_norm)
    
    # C. YOLO Detections
    yolo_results = yolo_model.predict(source=frame, conf=0.4, verbose=False)
    annotated_frame = yolo_results[0].plot()
    
    frame_data = {
        "navigation": {"status": depth_text, "danger_score": round(score, 3)},
        "objects": [],
        "resolution": {"w": frame_w, "h": frame_h}
    }

    for r in yolo_results:
        for box in r.boxes:
            label = yolo_model.names[int(box.cls[0])]
            coords = box.xyxy[0].tolist()
            center_x = (coords[0] + coords[2]) / 2
            position = "Center" if frame_w*0.33 < center_x < frame_w*0.66 else ("Left" if center_x < frame_w*0.33 else "Right")
            frame_data["objects"].append({"label": label, "position": position})

    # D. Gemini & Voice (Triggered only on Danger)
    audio_b64 = None
    now = time.time()
    
    if score > 0.5 and (now - last_gemini_time) > cooldown_seconds:
        last_gemini_time = now
        print(f" [!] {depth_text} detected. Fetching Gemini Advice...")
        try:
            current_gemini_statement = get_gemini_analysis(frame, json.dumps(frame_data))
            
            # Text to Speech
            audio_stream = el_client.text_to_speech.convert(
                text=current_gemini_statement,
                voice_id="pNInz6obpgDQGcFmaJgB",
                model_id="eleven_turbo_v2_5"
            )
            audio_content = b"".join(list(audio_stream))
            audio_b64 = base64.b64encode(audio_content).decode('utf-8')
        except Exception as e:
            print(f" AI/Voice Error: {e}")

    # E. Display Visualization
    depth_viz = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    depth_color_res = cv2.resize(depth_viz, (frame_w, frame_h))
    combined_view = np.hstack((annotated_frame, depth_color_res))
    
    mode_label = "LIVE MODE" if is_live else "TEST MODE (FALLBACK)"
    cv2.putText(combined_view, mode_label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    if current_gemini_statement:
        cv2.putText(combined_view, current_gemini_statement[:75], (50, frame_h - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("Vision Engine Dashboard", combined_view)
    cv2.waitKey(1)
    
    return frame_data, audio_b64

# --- 3. BROADCAST ENDPOINT (Live Mode) ---
@app.post("/detect")
async def detect_broadcast(payload: dict = Body(...)):
    global last_phone_request_time
    last_phone_request_time = time.time()  # Update heartbeat
    
    img_bytes = base64.b64decode(payload['image'])
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    frame_data, audio_b64 = process_frame(frame, is_live=True)

    return {
        "navigation": frame_data["navigation"],
        "objects": frame_data["objects"],
        "advice": current_gemini_statement,
        "audio": audio_b64
    }

# --- 4. FALLBACK LOGIC ---
def fallback_test_loop():
    """Runs the vision engine on file.mp4 if no live stream is detected."""
    video_path = "file.mp4"
    if not os.path.exists(video_path):
        print(f" [!] Warning: {video_path} not found. Fallback mode unavailable.")
        return

    cap = cv2.VideoCapture(video_path)
    print(" [*] Fallback Test Loop ready. Waiting for live connection...")

    while True:
        # If no phone request has been received in the last 3 seconds
        if (time.time() - last_phone_request_time) > 3.0:
            success, frame = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
                continue
            
            process_frame(frame, is_live=False)
        else:
            # Phone is active, wait and let live mode handle processing
            time.sleep(1)

# --- 5. RUNTIME ---
if __name__ == "__main__":
    import uvicorn
    # Start the FastAPI server in a background thread
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000), 
        daemon=True
    )
    server_thread.start()
    
    # Run the fallback loop in the main thread (OpenCV requires the main thread for UI)
    fallback_test_loop()