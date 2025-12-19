import cv2
import mediapipe as mp
import numpy as np
import time
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# Initialize MediaPipe components
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Check if points intersect with polygons
def check_intersection(multi_pose_landmarks, polygons, image_width, image_height):
    status = {}
    
    # Initialize all as clear first
    for i in range(len(polygons)):
        status[i+1] = "clear,{}".format(i+1)
        
    if not multi_pose_landmarks:
        return status
        
    # Check each polygon
    for i, poly in enumerate(polygons):
        frame_num = i + 1
        is_occupied = False
        
        # Iterate over all detected people
        for landmarks in multi_pose_landmarks:
            if is_occupied:
                break
                
            # Iterate over all points of the person
            for landmark in landmarks:
                x = int(landmark.x * image_width)
                y = int(landmark.y * image_height)
                point = Point(x, y)
                
                if poly.contains(point):
                    is_occupied = True
                    break
        
        if is_occupied:
            status[frame_num] = "enter,{}".format(frame_num)
            
    return status

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through each detected person
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks
        # We need to convert the task result landmarks to the proto format for the drawing utils
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
            for landmark in pose_landmarks
        ])
        
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style())
            
    return annotated_image

import os

def main():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pose_landmarker_lite.task')
    
    # Create an PoseLandmarker object.
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=5) # Detect up to 5 people
        
    with PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        
        # Define 3 polygonal frames (4 points each)
        poly1_coords = [(50, 50), (250, 50), (250, 150), (50, 200)]
        poly2_coords = [(300, 100), (500, 100), (550, 350), (250, 300)]
        poly3_coords = [(400, 350), (600, 350), (600, 400), (400, 450)]
        
        shapely_polys = [
            Polygon(poly1_coords),
            Polygon(poly2_coords),
            Polygon(poly3_coords)
        ]
        
        numpy_polys = [
            np.array(poly1_coords, np.int32).reshape((-1, 1, 2)),
            np.array(poly2_coords, np.int32).reshape((-1, 1, 2)),
            np.array(poly3_coords, np.int32).reshape((-1, 1, 2))
        ]
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Convert to RGB and create mp Image
            image = cv2.flip(image, 1) # Optional: mirror image for better UX
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            # Calculate timestamp in ms
            timestamp_ms = int(time.time() * 1000)
            
            # Detect poses
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            image_height, image_width, _ = image.shape
            
            # Draw stick figures on a copy of the original BGR image (converted from the RGB we used for processing)
            # Actually, let's use the helper which works on numpy array. 
            # We need to pass the BGR image to it if we want the result in BGR,
            # but mp drawing utils normally expect RGB or work on whatever you give them.
            # To be safe, let's work on the BGR image 'image' directly.
            
            # Helper function expects RGB or BGR? 
            # The mp.solutions.drawing_utils uses opencv drawing which works on BGR if passed.
            # But the helper I wrote takes 'rgb_image'. Let's adjust it to take 'image' (BGR).
            annotated_image = draw_landmarks_on_image(image, detection_result)
            
            # Check conditions
            if detection_result.pose_landmarks:
                status_updates = check_intersection(detection_result.pose_landmarks, shapely_polys, image_width, image_height)
            else:
                status_updates = {i+1: "clear,{}".format(i+1) for i in range(len(shapely_polys))}
            
            # Draw Polygons and Status
            for i, poly_pts in enumerate(numpy_polys):
                frame_num = i + 1
                curr_status = status_updates.get(frame_num, "clear,{}".format(frame_num))
                
                if "enter" in curr_status:
                    poly_color = (0, 0, 255) # Red
                else:
                    poly_color = (0, 255, 0) # Green
                
                cv2.polylines(annotated_image, [poly_pts], True, poly_color, 2)
                
                # Put text near the polygon
                text_pos = poly_pts[0][0]
                cv2.putText(annotated_image, curr_status, (text_pos[0], text_pos[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, poly_color, 2)
            
            cv2.imshow('Multi-Person Pose Detection', annotated_image)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
