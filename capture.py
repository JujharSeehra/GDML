from mss import MSS
import cv2
import numpy as np

class ScreenCapture:
    def __init__(self, left, top, width, height):
        self.sct = MSS()
        self.region = {
            "left": left,
            "top": top,
            "width": width,
            "height": height
        }

    def grab_raw(self):
        frame = np.array(self.sct.grab(self.region))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame
    
    def grab_processed(self):
        frame = self.grab_raw()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA)
        return frame

if __name__ == "__main__":
    capture = ScreenCapture(left=90, top=100,  width=1000, height=700)
    while True:

        frame = capture.grab_processed()
        cv2.imshow("GDML Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()