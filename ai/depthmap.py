import cv2
import torch
import time
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run depth estimation with MiDaS models')
    parser.add_argument('-m', '--model', default='MiDaS_small',
                        choices=['MiDaS_small', 'DPT_Hybrid', 'DPT_Large'],
                        help='Select model type')
    parser.add_argument('-v', '--view', default='blended',
                        choices=['blended', 'depth', 'side-by-side'],
                        help='Visualization mode')
    args = parser.parse_args()

    model_type = args.model
    print("Loading {} from torch.hub...".format(model_type))

    try:
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
    except Exception as e:
        print("Error loading model:", e)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)

    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "MiDaS_small":
        transform = midas_transforms.small_transform
    else:
        transform = midas_transforms.dpt_transform

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Webcam not accessible")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Starting Depth Estimation. Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(img).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic"
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        depth_map = cv2.normalize(
            depth_map, None, 0, 255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U
        )

        depth_map_color = cv2.applyColorMap(
            depth_map, cv2.COLORMAP_MAGMA
        )

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


