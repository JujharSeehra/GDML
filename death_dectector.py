import cv2
import numpy as np


class DeathDetector:
    def __init__(
        self,
        motion_threshold=1.0,
        required_static_frames=18,
        min_episode_frames=30
    ):
        self.motion_threshold = motion_threshold
        self.required_static_frames = required_static_frames
        self.min_episode_frames = min_episode_frames

        self.previous_frame = None
        self.static_frames = 0
        self.episode_frames = 0

    def reset(self):
        self.previous_frame = None
        self.static_frames = 0
        self.episode_frames = 0

    def update(self, frame):
        """
        frame should be the 96x96 grayscale uint8 image
        returned by capture.grab_processed()
        """

        self.episode_frames += 1

        if self.previous_frame is None:
            self.previous_frame = frame.copy()
            return False, 0.0

        # Ignore borders/UI somewhat
        current = frame[8:-8, 8:-8]
        previous = self.previous_frame[8:-8, 8:-8]

        difference = cv2.absdiff(
            current,
            previous
        )

        motion_score = float(
            np.mean(difference)
        )

        self.previous_frame = frame.copy()

        if self.episode_frames < self.min_episode_frames:
            return False, motion_score

        if motion_score < self.motion_threshold:
            self.static_frames += 1
        else:
            self.static_frames = 0

        dead = (
            self.static_frames >=
            self.required_static_frames
        )

        return dead, motion_score