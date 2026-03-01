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

from gemini import get_gemini_analysis 

# ==========================================
# --- MODE SELECTOR ---
# Set to True for Phone/API, False for Local Video
RUN_LIVE = True 
TEST_VIDEO_PATH = "file.mp4"
# ==========================================

app = FastAPI()

# --- 1. CONFIGURATION & INITIALIZATION ---
providers = ['CPUExecutionProvider'] 
yolo_model = YOLO("yolov8s.onnx", task='detect')
depth_session = ort.InferenceSession("midas_small.onnx", providers=providers)
el_client = ElevenLabs(api_key=ELEVEN_KEY)

last_gemini_time = 0
cooldown_seconds = 5
current_gemini_statement = "System Initialized."

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

# --- 2. THE CORE VISION ENGINE ---
def process_frame(frame):
    """Handles AI logic and returns display image + data for API."""
    global last_gemini_time, current_gemini_statement
    frame_h, frame_w = frame.shape[:2]

    # Vision Models
    depth_map = get_depth_map(frame)
    depth_text, text_color, score = analyze_obstacle_density(depth_map)
    yolo_results = yolo_model.predict(source=frame, conf=0.4, verbose=False)
    
    frame_data = {
        "navigation": {"status": depth_text, "danger_score": round(score, 3)},
        "objects": [],
        "resolution": {"w": frame_w, "h": frame_h}
    }

    # YOLO Processing
    annotated_frame = yolo_results[0].plot()
    for r in yolo_results:
        for box in r.boxes:
            label = yolo_model.names[int(box.cls[0])]
            coords = box.xyxy[0].tolist()
            center_x = (coords[0] + coords[2]) / 2
            position = "Center" if frame_w*0.33 < center_x < frame_w*0.66 else ("Left" if center_x < frame_w*0.33 else "Right")
            frame_data["objects"].append({"label": label, "position": position})

    # Gemini & Voice
    audio_b64 = None
    now = time.time()
    if score > 0.5 and (now - last_gemini_time) > cooldown_seconds:
        last_gemini_time = now
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

    # Visual Output
    depth_viz = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    depth_color_res = cv2.resize(depth_viz, (frame_w, frame_h))
    combined_view = np.hstack((annotated_frame, depth_color_res))
    
    # Overlay Mode Info
    mode_text = "LIVE MODE" if RUN_LIVE else "TEST MODE (FILE)"
    cv2.putText(combined_view, mode_text, (20, 30), 1, 1.5, (0, 255, 255), 2)

    if current_gemini_statement:
        cv2.putText(combined_view, current_gemini_statement[:75], (50, frame_h - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("Vision Dashboard", combined_view)
    cv2.waitKey(1) # Necessary for OpenCV window

    return frame_data, audio_b64

# --- 3. LIVE ENDPOINT ---
@app.post("/detect")
async def detect_broadcast(payload: dict = Body(...)):
    img_bytes = base64.b64decode(payload['image'])
    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    frame_data, audio_b64 = process_frame(frame)
    
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

# --- NEW: MESSAGE HANDLING ---

def speak_message(text: str):
    """Helper to print to terminal and play via ElevenLabs."""
    print(f"\n[REMOTE MESSAGE]: {text}")
    try:
        audio_stream = el_client.text_to_speech.convert(
            text=text,
            voice_id="pNInz6obpgDQGcFmaJgB", # You can use a different voice ID for messages
            model_id="eleven_turbo_v2_5"
        )
        # For local playback, you'd typically use a library like 'mpv' or 'pydub'
        # Since this script is a server, we'll just log that it's processing.
        # If you want the computer RUNNING this script to talk out loud:
        import ioprocessing_placeholder # Use a local player if needed
    except Exception as e:
        print(f"Error playing remote message: {e}")

@app.post("/message")
async def receive_message(payload: dict = Body(...)):
    """
    Endpoint for other computers to send text.
    Payload format: {"message": "Hello from the other side"}
    """
    msg = payload.get("message", "")
    if msg:
        # We run this in a thread so the API response isn't delayed by TTS generation
        threading.Thread(target=speak_message, args=(msg,)).start()
        return {"status": "Message received and being spoken"}
    return {"status": "Empty message"}, 400

# --- MODIFIED MAIN ENTRY ---
if __name__ == "__main__":
    import uvicorn
    # If RUN_LIVE is False, uvicorn doesn't normally run. 
    # To support both, we'll run uvicorn in a separate thread if in TEST MODE.
    
    if not RUN_LIVE:
        # Start the API server in the background so it can still receive messages 
        # while the local video window is running.
        api_thread = threading.Thread(
            target=uvicorn.run, 
            args=(app,), 
            kwargs={"host": "0.0.0.0", "port": 8000}, 
            daemon=True
        )
        api_thread.start()
        
        print(f"[!] Starting in TEST MODE (File: {TEST_VIDEO_PATH})")
        print("[!] API Server also running on port 8000 for remote messages.")
        run_test_mode()
    else:
        print("[!] Starting in LIVE MODE (FastAPI)")
        uvicorn.run(app, host="0.0.0.0", port=8000)