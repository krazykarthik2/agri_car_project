import cv2
import numpy as np
from tensorflow_lite_support.task.vision import PoseLandmarker

# Initialize PoseLandmarker
landmarker = PoseLandmarker(model_path="pose_landmarker_lite.task")  # Adjust model path as needed

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (192, 192))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[None, :, :, :]

    outputs = landmarker.detect(img)  # Use PoseLandmarker for inference
    keypoints = outputs.keypoints  # Access keypoints from outputs

    for i in range(len(keypoints)):
        x, y, confidence = keypoints[i]
        if confidence > 0.5:  # Only draw keypoints with high confidence
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    # Draw skeleton connections (example for a few connections)
    connections = [(0, 1), (1, 2), (2, 3), (1, 4)]  # Example connections
    for start, end in connections:
        if keypoints[start][2] > 0.5 and keypoints[end][2] > 0.5:  # Check confidence
            cv2.line(frame, (int(keypoints[start][0]), int(keypoints[start][1])),
                     (int(keypoints[end][0]), int(keypoints[end][1])), (255, 0, 0), 2)

    cv2.imshow("Pose", frame)
    if cv2.waitKey(1) == 27:
        break
cap.release()  # END: Release resources
