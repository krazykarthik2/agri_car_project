import cv2
import numpy as np
import tntorch  # TensorRT python loader (or use trt-engine API)

engine = tntorch.load("movenet_lightning_fp16.trt")

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (192, 192))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[None, :, :, :]

    outputs = engine(img)  # run inference
    # (standard MoveNet output parsing)
    keypoints = outputs[0]  # Assuming outputs contains keypoints
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
cap.release()
