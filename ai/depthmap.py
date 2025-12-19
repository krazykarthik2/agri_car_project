import cv2
import torch
import time
import numpy as np
import argparse

class DepthEstimator:
    def __init__(self, model_type="MiDaS_small"):
        self.model_type = model_type
        print("Loading {} from torch.hub...".format(model_type))
        
        try:
            self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        except Exception as e:
            print("Error loading model: {}".format(e))
            raise e

        # Select device (GPU if available is heavily optimized, else CPU)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("Running on device: {}".format(self.device))
        self.midas.to(self.device)
        self.midas.eval()

        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        
        if model_type == "MiDaS_small":
            self.transform = midas_transforms.small_transform
        else:
            self.transform = midas_transforms.dpt_transform

    def estimate_depth(self, frame):
        """
        Takes a BGR frame (numpy array), returns a BGR color-mapped depth map.
        """
        # Transform input frame
        # cv2 uses BGR, model needs RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms and move to device
        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            
            # Resize prediction to original image resolution
            # Interpolation is critical for quality
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic"
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        
        # Normalize depth map to 0-255 for 8-bit image visualization
        depth_map = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply pseudo-color map (MAGMA is distinct for depth: Black=Far, White/Bright=Near)
        depth_map_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
        
        return depth_map_color

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
    
    # Initialize Estimator
    try:
        estimator = DepthEstimator(args.model)
    except Exception as e:
        print("Failed to initialize DepthEstimator: {}".format(e))
        return

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
            
        # Get Depth Map
        depth_map_color = estimator.estimate_depth(frame)
        
        # Visualization Mode
        if args.view == 'blended':
            display_img = cv2.addWeighted(frame, 0.6, depth_map_color, 0.4, 0)
        elif args.view == 'side-by-side':
            sf = cv2.resize(frame, None, fx=0.5, fy=0.5)
            sd = cv2.resize(depth_map_color, None, fx=0.5, fy=0.5)
            display_img = np.hstack((sf, sd))
        else:
            display_img = depth_map_color

        cv2.imshow("MiDaS Depth Estimation", display_img)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


