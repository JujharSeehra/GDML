import time
from collections import deque
import cv2
import numpy as np
import torch
from capture import ScreenCapture
from controller import GDController
from train_bc import DashNet

FPS = 20

def main():
    capture = ScreenCapture(left = 90, top = 100, width = 1000, height=700)
    controller = GDController()

    if torch.backends.mps.is_available():
        device = torch.device("mps")

    elif torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    model = DashNet().to(device)

    model.load_state_dict(
        torch.load("models/dash_bc.pth", map_location=device))

    model.eval()
    frame_buffer = deque(maxlen=4)

    print("Switch to Geometry Dash!")

    time.sleep(3)
    print("DashRL is playing.")

    try:

        while True:

            start = time.time()
            frame = capture.grab_processed()
            frame = frame.astype(np.float32) / 255.0
            frame_buffer.append(frame)

            if len(frame_buffer) < 4:
                continue

            observation = np.stack(frame_buffer)
            observation = torch.tensor(observation, dtype=torch.float32)
            observation = observation.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(observation)
                action = output.argmax(dim=1).item()
            if action == 1:
                controller.press()
            else:
                controller.release()
            elapsed = (time.time() - start)
            delay = (1 / FPS) - elapsed
            if delay > 0:
                time.sleep(delay)

    except KeyboardInterrupt:
        controller.release()
        print("\nDashRL stopped.")

if __name__ == "__main__":
    main()