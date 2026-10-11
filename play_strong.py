import time

from collections import deque

import numpy as np
import torch

from capture import ScreenCapture
from controller import GDController

from train_strong import (
    StrongDashNet,
    FRAME_STACK,
)


FPS = 30


MODEL_PATHS = [
    "models/dash_strong_1.pth",
]


# Slight hysteresis prevents HOLD/RELEASE flickering.
HOLD_THRESHOLD = 0.52

RELEASE_THRESHOLD = 0.48


def get_device():

    if torch.backends.mps.is_available():

        return torch.device(
            "mps"
        )

    if torch.cuda.is_available():

        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


def load_models(
    device
):

    models = []

    for path in MODEL_PATHS:

        model = (
            StrongDashNet()
            .to(device)
        )

        model.load_state_dict(
            torch.load(
                path,
                map_location=device
            )
        )

        model.eval()

        models.append(
            model
        )

        print(
            f"Loaded {path}"
        )

    return models


def main():

    device = get_device()

    print(
        f"Using device: "
        f"{device}"
    )

    capture = ScreenCapture(
        left=90,
        top=100,
        width=1000,
        height=700,
    )

    controller = GDController()

    models = load_models(
        device
    )

    frame_buffer = deque(
        maxlen=FRAME_STACK
    )

    current_action = 0

    print()
    print(
        "Strong ensemble starts "
        "in 3 seconds."
    )

    time.sleep(3)

    try:

        while True:

            start_time = (
                time.time()
            )

            frame = (
                capture.grab_processed()
            )

            frame = (
                frame.astype(
                    np.float32
                )
                / 255.0
            )

            frame_buffer.append(
                frame
            )

            if (
                len(frame_buffer)
                < FRAME_STACK
            ):

                controller.release()

                continue

            observation = np.stack(
                frame_buffer
            )

            observation = (
                torch.tensor(
                    observation,
                    dtype=torch.float32
                )
                .unsqueeze(0)
                .to(device)
            )

            hold_probabilities = []

            with torch.no_grad():

                for model in models:

                    output = model(
                        observation
                    )

                    probabilities = (
                        torch.softmax(
                            output,
                            dim=1
                        )
                    )

                    hold_probability = float(
                        probabilities[
                            0,
                            1
                        ].item()
                    )

                    hold_probabilities.append(
                        hold_probability
                    )

            hold_probability = float(
                np.mean(
                    hold_probabilities
                )
            )

            # =========================================
            # HYSTERESIS
            # =========================================

            previous_action = (
                current_action
            )

            if current_action == 0:

                if (
                    hold_probability
                    >= HOLD_THRESHOLD
                ):

                    current_action = 1

            else:

                if (
                    hold_probability
                    <= RELEASE_THRESHOLD
                ):

                    current_action = 0

            # =========================================
            # CONTROLLER
            # =========================================

            if current_action == 1:

                controller.press()

            else:

                controller.release()

            if (
                current_action
                != previous_action
            ):

                print(
                    f"{'HOLD' if current_action else 'RELEASE'} "
                    f"| ensemble="
                    f"{hold_probability:.3f} "
                    f"| models="
                    f"{[round(x, 3) for x in hold_probabilities]}"
                )

            elapsed = (
                time.time()
                - start_time
            )

            delay = (
                1 / FPS
                - elapsed
            )

            if delay > 0:

                time.sleep(
                    delay
                )

    except KeyboardInterrupt:

        controller.release()

        print()
        print(
            "Strong player stopped."
        )


if __name__ == "__main__":
    main()