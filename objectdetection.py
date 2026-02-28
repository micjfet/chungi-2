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
    danger_score = np.mean(center_strip)
    
    if danger_score > 0.8:
        return "IMMEDIATE DANGER: Wall or Large Object Detected", (0, 0, 255)
    elif danger_score > 0.5:
        return "WARNING: Obstacle Approaching", (0, 165, 255)
    return "PATH CLEAR", (0, 255, 0)

video_path = "file.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    depth_map = get_depth_map(frame)
    depth_text, text_color = analyze_obstacle_density(depth_map)

    yolo_results = yolo_model.predict(source=frame, conf=0.4, verbose=False)
    
    frame_data = {
        "navigation": depth_text,
        "objects": []
    }

    for r in yolo_results:
        annotated_frame = r.plot() 
        for box in r.boxes:
            label = yolo_model.names[int(box.cls[0])]
            coords = box.xyxy[0].tolist()
            frame_data["objects"].append({
                "label": label,
                "box": [round(x, 1) for x in coords]
            })

    print(f"STREAMS DATA: {json.dumps(frame_data)}")
    
    depth_viz = (depth_map * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_viz, cv2.COLORMAP_MAGMA)
    depth_color_resized = cv2.resize(depth_color, (frame.shape[1], frame.shape[0]))
    
    combined_view = np.hstack((annotated_frame, depth_color_resized))
    
    cv2.putText(combined_view, depth_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 3)
    cv2.imshow("YOLOv8 + MiDaS Depth (WSL)", combined_view)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
