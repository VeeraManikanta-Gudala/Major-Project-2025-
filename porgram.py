import cv2
from ultralytics import YOLO

# Load YOLOv8 model (small version for faster inference)
model = YOLO('yolov8s.pt')

cap = cv2.VideoCapture("rtsp://192.168.1.104:8080/h264_ulaw.sdp")

# Check if the connection was successful
if not cap.isOpened():
    print("Failed to connect to the RTSP stream.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Displaying the stream with YOLOv8 object detection
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to retrieve frame.")
        break
    
    # Performing object detection
    results = model.predict(frame)
    
    # Extracting detected boxes, classes, and confidence scores
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Getting box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Getting class and confidence
            cls = int(box.cls[0])
            conf = box.conf[0]
            label = f"{model.names[cls]} {conf:.2f}"
            
            # Drawing rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Displaying the frame
    cv2.imshow("YOLOv8 Object Detection", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# cap = cv2.VideoCapture("rtsp://192.168.106.193:8080/h264_ulaw.sdp")
