import os
import time
import cv2
import numpy as np
from pynput.mouse import Listener, Button
from capture import ScreenCapture
import Quartz

BASE_DIR = "recordings"


def get_next_session():
    os.makedirs(BASE_DIR, exist_ok=True)

    index = 1

    while True:
        path = os.path.join(BASE_DIR, f"session_{index:03d}")
        if not os.path.exists(path):
            return path
        index += 1


SAVE_DIR = get_next_session()
FPS = 30
def mouse_is_held():
    return Quartz.CGEventSourceButtonState(Quartz.kCGEventSourceStateHIDSystemState, Quartz.kCGMouseButtonLeft)
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    capture = ScreenCapture(left = 90, top = 100, width = 1000, height = 700)
    labels = []
    print("Recording Starts in 3 seconds")
    time.sleep(3)
    print("Recording")
    frame_index = 0
    frame_time = 1/FPS
    previous_action = None
    try:
        while True:
            start_time = time.time()
            held = mouse_is_held()
            action = 1 if held else 0
            if action != previous_action:
                if action == 1:
                    print("Down")
                else:
                    print("Up")
                previous_action = action
            frame = capture.grab_processed()
            filename = f"frame_{frame_index:06d}.png"
            path = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(path, frame)
            labels.append(f"{filename},{action}\n")
            frame_index += 1
            elapsed = time.time() - start_time
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\nSTOPPING RECORDING")
    csv_path = os.path.join(SAVE_DIR,"labels.csv")
    with open(csv_path, "w") as f:
        f.write("filename,action\n")
        f.writelines(labels)
    print()
    print(f"Saved {frame_index} frames")
    print(f"Dataset: {SAVE_DIR}")
if __name__ == "__main__":
    main()