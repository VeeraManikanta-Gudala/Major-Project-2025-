import cv2
import torch
from ultralytics import YOLO
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config

# Load YOLOv8 model (small version for faster inference)
model = YOLO('yolov8s.pt')

# Initialize DeepSORT
cfg = get_config()
cfg.merge_from_file('deep_sort/configs/deep_sort.yaml')
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT, 
                    max_dist=cfg.DEEPSORT.MAX_DIST, 
                    min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, 
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE, 
                    max_age=cfg.DEEPSORT.MAX_AGE, 
                    n_init=cfg.DEEPSORT.N_INIT, 
                    nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=torch.cuda.is_available())

cap = cv2.VideoCapture("rtsp://192.168.1.104:8080/h264_ulaw.sdp")

# Check if the connection was successful
if not cap.isOpened():
    print("Failed to connect to the RTSP stream.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Displaying the stream with YOLOv8 object detection and DeepSORT tracking
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to retrieve frame.")
        break
    
    # Performing object detection
    results = model.predict(frame)[0]  # Get first batch result
    
    # Extracting detected boxes, classes, and confidence scores
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        conf = box.conf[0]                     # Confidence score
        cls = int(box.cls[0])                  # Class label
        detections.append([x1, y1, x2, y2, conf, cls])
    
    # Prepare data for DeepSORT
    if detections:
        bboxes = [det[:4] for det in detections]
        scores = [det[4] for det in detections]
        class_ids = [det[5] for det in detections]
        
        # Update tracker
        outputs = deepsort.update(torch.tensor(bboxes), torch.tensor(scores), frame)
        
        # Draw tracking boxes and IDs
        for output in outputs:
            x1, y1, x2, y2, track_id = output
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Displaying the frame
    cv2.imshow("YOLOv8 + DeepSORT Tracking", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

