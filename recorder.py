import os
import time
import cv2
import numpy as np
from pynput.mouse import Listener, Button
from capture import ScreenCapture

SAVE_DIR = "recordings/session_001"
FPS = 20
mouse_held = False
running = True
def on_click(x,y,button,pressed):
    global mouse_held
    if button == Button.left:
        mouse_held = pressed
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    capture = ScreenCapture(left=90, top=100, width=1000, height=700)
    labels = []
    listener = Listener(on_click=on_click)
    listener.start
    print("Recording starts in 3 seconds")
    time.sleep(3)
    print("RECORDING")
    frame_index = 0
    frame_time = 1/FPS
    try:
        while True:
            start = time.time()
            frame = capture.grab_processed()
            filename = f"frame_{frame_index:06d}.png"
            path = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(path, frame)
            action = 1 if mouse_held else 0
            labels.append(f"{filename},{action}\n")
            frame_index += 1
            elapsed = time.time() - start
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("Stopping Recording")
    with open(os.path.join(SAVE_DIR, "labels.csv"), "w") as f:
        f.write("filename,action\n")
        f.writelines(labels)
    listener.stop()
    print(f"Saved {frame_index} frames.")
    print(f"Dataset saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()