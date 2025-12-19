import cv2
import torch
import time
import numpy as np
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='run depth estimation with MiDaS models')
    parser.add_argument('-m', '--model', default='MiDaS_small', 
                        choices=['MiDaS_small', 'DPT_Hybrid', 'DPT_Large'],
                        help='select model type: MiDaS_small, DPT_Hybrid, DPT_Large')
    parser.add_argument('-v', '--view', default='blended',
                        choices=['blended', 'depth', 'side-by-side'],
                        help='visualization mode: blended, depth, side-by-side')
    args = parser.parse_args()
    
    model_type = args.model
    print(f"Loading {model_type} from torch.hub...")
    
    try:
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Select device (GPU if available is heavily optimized, else CPU)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Running on device: {device}")
    midas.to(device)
    midas.eval()

    # Load transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    
    if model_type == "MiDaS_small":
        transform = midas_transforms.small_transform
    else:
        transform = midas_transforms.dpt_transform

    # Open Webcam
    cap = cv2.VideoCapture(0)
    
    # Optional: Request higher camera resolution if supported
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Starting Depth Estimation Loop. Press 'ESC' to exit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        # Transform input frame
        # cv2 uses BGR, model needs RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms and move to device
        input_batch = transform(img).to(device)

        # Predict
        with torch.no_grad():
            prediction = midas(input_batch)
            
            # Resize prediction to original image resolution
            # Interpolation is critical for quality
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Post-process for visualization
        depth_map = prediction.cpu().numpy()
        
        # Normalize depth map to 0-255 for 8-bit image visualization
        depth_map = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply pseudo-color map (MAGMA is distinct for depth: Black=Far, White/Bright=Near)
        depth_map_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
        
        # Visualization Mode
        if args.view == 'blended':
             # 0.6 Original + 0.4 Depth Map
            display_img = cv2.addWeighted(frame, 0.6, depth_map_color, 0.4, 0)
        elif args.view == 'side-by-side':
            # Resize both to 50% width/height to fit on screen
            small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            small_depth = cv2.resize(depth_map_color, (0,0), fx=0.5, fy=0.5)
            display_img = np.hstack((small_frame, small_depth))
        else:
            display_img = depth_map_color
        
        cv2.imshow('MiDaS Depth Estimation', display_img)
        
        if cv2.waitKey(1) & 0xFF == 27: # ESC key
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
