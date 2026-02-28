import cv2
import numpy as np
import onnxruntime as ort
import json
from ultralytics import YOLO

providers = ['CPUExecutionProvider'] 
yolo_model = YOLO("yolov8n.onnx", task='detect')
depth_session = ort.InferenceSession("midas_small.onnx", providers=providers)

def get_depth_map(img_bgr):
    img_input = cv2.resize(img_bgr, (256, 256))
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    img_input = img_input.astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_input = (img_input - mean) / std
    img_input = img_input.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)

    onnx_input = {depth_session.get_inputs()[0].name: img_input}
    depth_output = depth_session.run(None, onnx_input)[0]

    depth_map = np.squeeze(depth_output)
    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
    return depth_norm

def analyze_obstacle_density(depth_norm):
    h, w = depth_norm.shape
    center_strip = depth_norm[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
    danger_score = float(np.mean(center_strip))
    
    if danger_score > 0.8:
        return "IMMEDIATE DANGER", (0, 0, 255), danger_score
    elif danger_score > 0.5:
        return "WARNING: Obstacle", (0, 165, 255), danger_score
    return "PATH CLEAR", (0, 255, 0), danger_score

video_path = "file.mp4"
cap = cv2.VideoCapture(video_path)

original_fps = cap.get(cv2.CAP_PROP_FPS)
target_fps = 4
frame_step = int(original_fps / target_fps) if original_fps > 0 else 1
current_frame_idx = 0

print(f"Video FPS: {original_fps} | Processing every {frame_step} frames to reach {target_fps} FPS.")

while cap.isOpened():
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
    success, frame = cap.read()
    if not success: break
    
    frame_h, frame_w = frame.shape[:2]

    depth_map = get_depth_map(frame)
    depth_text, text_color, score = analyze_obstacle_density(depth_map)
    yolo_results = yolo_model.predict(source=frame, conf=0.4, verbose=False)
    
    frame_data = {
        "timestamp_sec": round(current_frame_idx / original_fps, 2),
        "navigation": {"status": depth_text, "danger_score": round(score, 3)},
        "objects": [],
        "resolution": {"w": frame_w, "h": frame_h}
    }

    for r in yolo_results:
        annotated_frame = r.plot() 
        for box in r.boxes:
            label = yolo_model.names[int(box.cls[0])]
            coords = box.xyxy[0].tolist()
            center_x = (coords[0] + coords[2]) / 2
            position = "Center"
            if center_x < frame_w * 0.33: position = "Left"
            elif center_x > frame_w * 0.66: position = "Right"

            frame_data["objects"].append({
                "label": label, "position": position, "box": [round(x, 1) for x in coords]
            })

    print("\n" + "="*70)
    print(f" TIME: {frame_data['timestamp_sec']}s | STATUS: {depth_text} | DANGER SCORE: {round(score, 3)}")
    print("-"*70)
    
    if not frame_data["objects"]:
        print(" [!] No objects detected in this frame.")
    else:
        print(f"{'OBJECT':<15} | {'POSITION':<10} | {'BOUNDING BOX (x1, y1, x2, y2)':<30}")
        print("-"*70)
        
        for obj in frame_data["objects"]:
            label = obj['label'].upper()
            pos = obj['position']
            box = str(obj['box'])
            print(f"{label:<15} | {pos:<10} | {box:<30}")
            
    print("="*70)
    
    depth_viz = (depth_map * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_viz, cv2.COLORMAP_MAGMA)
    depth_color_resized = cv2.resize(depth_color, (frame_w, frame_h))
    combined_view = np.hstack((annotated_frame, depth_color_resized))
    
    display_text = f"{depth_text} ({round(score, 2)})"
    cv2.putText(combined_view, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 3)
    cv2.imshow("YOLOv8 + MiDaS Depth", combined_view)

    current_frame_idx += frame_step

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()