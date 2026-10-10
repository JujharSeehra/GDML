import cv2
import numpy as np


class DeathDetector:
    def __init__(
        self,
        fps=30,
        motion_threshold=1.0,
        required_static_frames=18,
        min_episode_seconds=2.0,
    ):
        self.fps = fps

        self.motion_threshold = motion_threshold

        self.required_static_frames = (
            required_static_frames
        )

        self.min_episode_frames = int(
            min_episode_seconds * fps
        )

        self.reset()

    def reset(self):
        self.previous_frame = None

        self.static_frames = 0

        self.episode_frames = 0

    def update(self, frame):

        self.episode_frames += 1

        if self.previous_frame is None:
            self.previous_frame = frame.copy()
            return False, 0.0

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

    # Build evidence of a stopped screen
        if motion_score < self.motion_threshold:
            self.static_frames += 1

        else:
        # Don't instantly erase all evidence.
        # Small animations on the death screen are common.
            self.static_frames = max(
            0,
            self.static_frames - 2
        )

        dead = (
        self.static_frames
        >= self.required_static_frames
    )

        return dead, motion_score