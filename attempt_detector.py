import numpy as np


class AttemptDetector:
    def __init__(
        self,
        bright_threshold=210,
        min_bright_pixels=15,
        required_present_frames=2,
        rearm_absent_frames=5,
    ):
        self.bright_threshold = bright_threshold
        self.min_bright_pixels = min_bright_pixels

        self.required_present_frames = required_present_frames
        self.rearm_absent_frames = rearm_absent_frames

        self.present_frames = 0
        self.absent_frames = 0

        self.armed = True

    def reset(self):
        self.present_frames = 0
        self.absent_frames = 0
        self.armed = True

    def update(self, frame):
        """
        frame:
            96x96 grayscale uint8 image from
            ScreenCapture.grab_processed()

        returns:
            event
            banner_present
            bright_pixels
        """

        # Region where "ATTEMPT XX" appears.
        #
        # Excludes:
        # - percentage at top
        # - spikes at bottom
        # - pause button at right
        roi = frame[
            15:30,
            15:75
        ]

        bright_pixels = int(
            np.count_nonzero(
                roi >= self.bright_threshold
            )
        )

        banner_present = (
            bright_pixels
            >= self.min_bright_pixels
        )

        event = False

        if banner_present:

            self.present_frames += 1
            self.absent_frames = 0

            if (
                self.armed
                and
                self.present_frames
                >= self.required_present_frames
            ):
                event = True
                self.armed = False

        else:

            self.present_frames = 0
            self.absent_frames += 1

            if (
                not self.armed
                and
                self.absent_frames
                >= self.rearm_absent_frames
            ):
                self.armed = True

        return (
            event,
            banner_present,
            bright_pixels,
        )