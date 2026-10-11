import time
from collections import deque

import numpy as np
import torch

from capture import ScreenCapture
from controller import GDController
from train_bc import DashNet

FPS = 30
FRAME_STACK = 4

def main():
    capture = ScreenCapture( left= 90, top = 100, width = 1000, height = 700)
    controller = GDController()
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using Device {device}")
    model = DashNet().to(device)
    model.load_state_dict(torch.load("models/dash_bc_baseline.pth", map_location=device))
    model.eval()
    frame_buffer = deque(maxlen=FRAME_STACK)
    print("ML Starts in 3 seconds")
    time.sleep(3)
    print("DashRL is playing")
    previous_action = None
    try:
        while True:
            start_time = time.time()
            frame = capture.grab_processed()
            frame = frame.astype(np.float32)/255.0
            frame_buffer.append(frame)
            if len(frame_buffer) < FRAME_STACK:
                continue
            observation = np.stack(frame_buffer)
            observation = torch.tensor(observation, dtype=torch.float32)
            observation = observation.unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(observation)
                probabilities = torch.softmax(output,dim=1)
                action=probabilities.argmax(dim=1).item()
            if action == 1:
                controller.press()
            else:
                controller.release()
            if action != previous_action:
                release_prob = probabilities[0][0].item()
                hold_prob = probabilities[0][1].item()
                print(f"Action {'HOLD' if action else 'RELEASE'} release = {release_prob:.3f}, hold = {hold_prob:.3f}")
                previous_action = action
            elapsed = time.time()-start_time
            delay = 1/FPS - elapsed
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        controller.release()
        print("Stopped GDML")
if __name__ == "__main__":
    main()