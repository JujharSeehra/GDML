import csv
import glob
import os
import random
import shutil
import time

from collections import deque

import numpy as np
import torch

from pynput.keyboard import (
    Listener,
)

from capture import ScreenCapture
from controller import GDController
from attempt_detector import AttemptDetector

from train_bc import DashNet

from online_train import (
    train_online_model,
)

from online_config import (
    CAPTURE_LEFT,
    CAPTURE_TOP,
    CAPTURE_WIDTH,
    CAPTURE_HEIGHT,

    FPS,
    FRAME_STACK,

    BASE_MODEL_PATH,
    ONLINE_MODEL_PATH,

    LEVEL_DURATION_SECONDS,
    MIN_EPISODE_SECONDS,
    RESTART_WAIT_SECONDS,

    CORRECTION_RELEASE_KEY,
    CORRECTION_HOLD_KEY,
    CORRECTION_SAVE_EVERY,

    EXPLORATION_RATE,
    UNCERTAINTY_THRESHOLD,

    MIN_ELITE_SECONDS,
    ELITE_FRACTION_OF_BEST,
    MAX_ELITE_FILES,

    RETRAIN_EVERY_ATTEMPTS,
    ATTEMPT_BRIGHT_THRESHOLD,
    ATTEMPT_MIN_BRIGHT_PIXELS,
    ATTEMPT_PRESENT_FRAMES,
    ATTEMPT_REARM_FRAMES,
    EPISODE_TAIL_TRIM_FRAMES,
)


RUN_LOG = "logs/online_runs.csv"


# =========================================================
# HUMAN CORRECTION KEYS
# =========================================================

class CorrectionKeys:

    def __init__(self):

        self.override_action = None

        self.retrain_requested = False

    def on_press(self, key):

        try:

            char = key.char.lower()

        except Exception:

            return

        if (
            char
            == CORRECTION_RELEASE_KEY
        ):

            self.override_action = 0

        elif (
            char
            == CORRECTION_HOLD_KEY
        ):

            self.override_action = 1

        elif char == "r":

            self.retrain_requested = True

    def on_release(self, key):

        try:

            char = key.char.lower()

        except Exception:

            return

        if (
            char
            == CORRECTION_RELEASE_KEY
            and self.override_action == 0
        ):

            self.override_action = None

        elif (
            char
            == CORRECTION_HOLD_KEY
            and self.override_action == 1
        ):

            self.override_action = None


# =========================================================
# DEVICE
# =========================================================

def get_device():

    if torch.backends.mps.is_available():

        return torch.device("mps")

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


# =========================================================
# MODEL
# =========================================================

def load_model(device):

    model = (
        DashNet().to(
            device
        )
    )

    model.load_state_dict(
        torch.load(
            ONLINE_MODEL_PATH,
            map_location=device
        )
    )

    model.eval()

    return model


# =========================================================
# CORRECTION STORAGE
# =========================================================

def save_correction(
    frame_stack,
    action,
):

    os.makedirs(
        "online_data/corrections",
        exist_ok=True
    )

    timestamp = (
        time.time_ns()
    )

    path = (
        "online_data/corrections/"
        f"correction_{timestamp}.npz"
    )

    frames = np.stack(
        list(frame_stack)
    ).astype(np.uint8)

    np.savez_compressed(
        path,
        frames=frames,
        action=np.array(
            action,
            dtype=np.int64
        )
    )


# =========================================================
# ELITE TRAJECTORIES
# =========================================================

def save_elite(
    attempt,
    score,
    frames,
    actions,
):

    if len(frames) < FRAME_STACK:
        return

    os.makedirs(
        "online_data/elite",
        exist_ok=True
    )

    path = (
        "online_data/elite/"
        f"attempt_{attempt:06d}_"
        f"score_{score:.3f}.npz"
    )

    np.savez_compressed(
        path,
        frames=np.stack(
            frames
        ).astype(np.uint8),
        actions=np.array(
            actions,
            dtype=np.int64
        ),
        score=np.array(
            score,
            dtype=np.float32
        )
    )


def prune_elites():

    paths = glob.glob(
        "online_data/elite/*.npz"
    )

    if len(paths) <= MAX_ELITE_FILES:
        return

    scored = []

    for path in paths:

        try:

            data = np.load(path)

            score = float(
                data["score"]
            )

            scored.append(
                (
                    score,
                    path
                )
            )

        except Exception:

            pass

    scored.sort(
        reverse=True
    )

    for (
        _,
        path
    ) in scored[
        MAX_ELITE_FILES:
    ]:

        os.remove(
            path
        )


# =========================================================
# RUN LOG
# =========================================================

def ensure_run_log():

    os.makedirs(
        "logs",
        exist_ok=True
    )

    if os.path.exists(
        RUN_LOG
    ):
        return

    with open(
        RUN_LOG,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "attempt",
                "score_seconds",
                "progress_percent",
                "best_seconds",
                "elite_saved",
                "corrections",
            ]
        )


def load_best_score():

    if not os.path.exists(
        RUN_LOG
    ):

        return 0.0

    best = 0.0

    with open(
        RUN_LOG,
        newline=""
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            try:

                best = max(
                    best,
                    float(
                        row[
                            "score_seconds"
                        ]
                    )
                )

            except Exception:

                pass

    return best


def append_run(
    attempt,
    score,
    percent,
    best,
    elite_saved,
    corrections,
):

    with open(
        RUN_LOG,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                attempt,
                f"{score:.4f}",
                (
                    ""
                    if percent is None
                    else f"{percent:.3f}"
                ),
                f"{best:.4f}",
                int(
                    elite_saved
                ),
                corrections,
            ]
        )


# =========================================================
# PROGRESS
# =========================================================

def get_progress_percent(
    score_seconds
):

    if (
        LEVEL_DURATION_SECONDS
        is None
    ):

        return None

    percent = (
        score_seconds
        / LEVEL_DURATION_SECONDS
        * 100.0
    )

    return min(
        percent,
        100.0
    )


# =========================================================
# INITIAL SETUP
# =========================================================

def prepare_online_model():

    if os.path.exists(
        ONLINE_MODEL_PATH
    ):

        return

    if not os.path.exists(
        BASE_MODEL_PATH
    ):

        raise FileNotFoundError(
            BASE_MODEL_PATH
        )

    shutil.copy2(
        BASE_MODEL_PATH,
        ONLINE_MODEL_PATH
    )

    print(
        "Created initial online model "
        "from dash_bc.pth."
    )


# =========================================================
# MAIN
# =========================================================

def main():

    prepare_online_model()

    ensure_run_log()

    device = get_device()

    print(
        f"Device: {device}"
    )

    capture = ScreenCapture(
        left=CAPTURE_LEFT,
        top=CAPTURE_TOP,
        width=CAPTURE_WIDTH,
        height=CAPTURE_HEIGHT,
    )

    controller = GDController()

    attempt_detector = AttemptDetector(

    bright_threshold=
        ATTEMPT_BRIGHT_THRESHOLD,

    min_bright_pixels=
        ATTEMPT_MIN_BRIGHT_PIXELS,

    required_present_frames=
        ATTEMPT_PRESENT_FRAMES,

    rearm_absent_frames=
        ATTEMPT_REARM_FRAMES,
)

    keys = CorrectionKeys()

    listener = Listener(
        on_press=keys.on_press,
        on_release=keys.on_release,
    )

    listener.start()

    model = load_model(
        device
    )

    frame_buffer = deque(
        maxlen=FRAME_STACK
    )

    trajectory_frames = []
    trajectory_actions = []

    correction_frames_this_attempt = 0

    attempt = 1

    best_score = (
        load_best_score()
    )

    attempts_since_training = 0

    new_training_data = False

    previous_action = None

    correction_counter = 0

    print()
    print(
        "Controls:"
    )

    print(
        f"  {CORRECTION_RELEASE_KEY.upper()} "
        "= force RELEASE"
    )

    print(
        f"  {CORRECTION_HOLD_KEY.upper()} "
        "= force HOLD"
    )

    print(
        "  R = retrain after next death"
    )

    print(
        "  Ctrl+C = stop"
    )

    print()
    print(
        "Switch to Geometry Dash."
    )

    print(
        "Starting in 3 seconds..."
    )

    time.sleep(3)

    attempt_start = (
        time.time()
    )

    try:

        while True:

            loop_start = (
                time.time()
            )

            raw_frame = (
                capture.grab_processed()
            )

            # ------------------------------------------------
            # NEW ATTEMPT / EPISODE DETECTION
            # ------------------------------------------------

            new_attempt, banner_present, bright_pixels = (
    attempt_detector.update(
        raw_frame
    )
)


            if new_attempt:

                controller.release()

                elapsed = (
        time.time()
        - attempt_start
    )

                print(
        f"\nATTEMPT BANNER DETECTED "
        f"(bright={bright_pixels})"
    )

    # --------------------------------------------
    # If the program started while the initial
    # ATTEMPT banner was already visible,
    # don't count that as a death.
    # --------------------------------------------

                if elapsed < MIN_EPISODE_SECONDS:

                    print(
            "Synchronizing to level start."
        )

                    attempt_start = (
            time.time()
        )

                    frame_buffer.clear()

                    trajectory_frames = []
                    trajectory_actions = []

                    correction_frames_this_attempt = 0

                    previous_action = None
                    correction_counter = 0

                    continue


                # ============================================
                # Previous attempt has ended
                # ============================================

                score = elapsed

                progress = (
        get_progress_percent(
            score
        )
    )


                old_best = best_score


    # --------------------------------------------
    # Remove death animation / reset frames
    # from self-imitation data
    # --------------------------------------------

                if (
        len(trajectory_frames)
        > EPISODE_TAIL_TRIM_FRAMES
        + FRAME_STACK
    ):

                    usable_frames = (
            trajectory_frames[
                :-EPISODE_TAIL_TRIM_FRAMES
            ]
        )

                    usable_actions = (
            trajectory_actions[
                :-EPISODE_TAIL_TRIM_FRAMES
            ]
        )

                else:

                    usable_frames = (
            trajectory_frames
        )

                    usable_actions = (
            trajectory_actions
        )


                elite_threshold = max(
        MIN_ELITE_SECONDS,

        old_best
        * ELITE_FRACTION_OF_BEST,
    )


                elite_saved = False


                if score >= elite_threshold:

                    save_elite(
            attempt,
            score,
            usable_frames,
            usable_actions,
        )

                    prune_elites()

                    elite_saved = True

                    new_training_data = True


                if score > best_score:

                    best_score = score


                append_run(
        attempt,
        score,
        progress,
        best_score,
        elite_saved,
        correction_frames_this_attempt,
    )


                print()
                print(
        "============================="
    )

                print(
        f"Attempt {attempt} finished"
    )

                print(
        f"Score: {score:.2f}s"
    )

                if progress is not None:

                    print(
            f"Progress: "
            f"{progress:.2f}%"
        )

                print(
        f"Best: {best_score:.2f}s"
    )

                print(
        f"Corrections: "
        f"{correction_frames_this_attempt}"
    )

                if elite_saved:

                    print(
            "Saved as ELITE trajectory"
        )

                print(
        "============================="
    )


                attempts_since_training += 1


    # ============================================
    # ONLINE RETRAIN
    # ============================================

                should_train = (

                    new_training_data

                    and

        (
            attempts_since_training
            >= RETRAIN_EVERY_ATTEMPTS

            or

            keys.retrain_requested
        )
    )


                if should_train:

                    print()
                    print(
            "Pausing Geometry Dash "
            "for retraining..."
        )

                    controller.toggle_pause()

                    time.sleep(0.3)


                    accepted = (
            train_online_model()
        )


                    if accepted:

                        model = (
                load_model(
                    device
                )
            )

                        print(
                "Loaded improved policy."
            )

                    else:

                        print(
                "Keeping previous policy."
            )


                    attempts_since_training = 0

                    keys.retrain_requested = False

                    new_training_data = False


                    print(
            "Resuming Geometry Dash."
        )

                    controller.toggle_pause()

                    time.sleep(0.3)


    # ============================================
    # NEW ATTEMPT
    # ============================================

                attempt += 1

                attempt_start = (
        time.time()
    )

                frame_buffer.clear()

                trajectory_frames = []
                trajectory_actions = []

                correction_frames_this_attempt = 0

                previous_action = None

                correction_counter = 0

                continue


# ------------------------------------------------
# While ATTEMPT XX is still visible:
# do not allow AI inputs.
# ------------------------------------------------

            if banner_present:

                controller.release()

                continue        

            frame_buffer.append(
                raw_frame.copy()
            )

            if (
                len(frame_buffer)
                < FRAME_STACK
            ):

                controller.release()

                trajectory_frames.append(
                    raw_frame.copy()
                )

                trajectory_actions.append(
                    0
                )

                continue

            # ------------------------------------------------
            # BUILD MODEL INPUT
            # ------------------------------------------------

            observation = np.stack(
                frame_buffer
            ).astype(
                np.float32
            ) / 255.0

            tensor = torch.tensor(
                observation,
                dtype=torch.float32
            )

            tensor = (
                tensor
                .unsqueeze(0)
                .to(device)
            )

            # ------------------------------------------------
            # POLICY
            # ------------------------------------------------

            with torch.no_grad():

                output = model(
                    tensor
                )

                probabilities = (
                    torch.softmax(
                        output,
                        dim=1
                    )[0]
                )

            normal_action = int(
                probabilities.argmax()
                .item()
            )

            confidence = float(
                probabilities.max()
                .item()
            )

            action = (
                normal_action
            )

            # ------------------------------------------------
            # LIMITED EXPLORATION
            # ------------------------------------------------
            #
            # Only explore when the model admits
            # uncertainty.
            #

            if (
                confidence
                < UNCERTAINTY_THRESHOLD
                and
                random.random()
                < EXPLORATION_RATE
            ):

                action = (
                    1
                    - normal_action
                )

            # ------------------------------------------------
            # HUMAN DAgger CORRECTION
            # ------------------------------------------------

            if (
                keys.override_action
                is not None
            ):

                action = (
                    keys.override_action
                )

                correction_counter += 1

                if (
                    correction_counter
                    % CORRECTION_SAVE_EVERY
                    == 0
                ):

                    save_correction(
                        frame_buffer,
                        action
                    )

                    correction_frames_this_attempt += 1

                    new_training_data = True

            # ------------------------------------------------
            # EXECUTE
            # ------------------------------------------------

            if action == 1:

                controller.press()

            else:

                controller.release()

            trajectory_frames.append(
                raw_frame.copy()
            )

            trajectory_actions.append(
                action
            )

            # ------------------------------------------------
            # DEBUG
            # ------------------------------------------------

            if (
                action
                != previous_action
            ):

                mode = (
                    "CORRECTION"
                    if keys.override_action
                    is not None
                    else "AI"
                )

                print(
                    f"{mode} | "
                    f"{'HOLD' if action else 'RELEASE'} "
                    f"| confidence="
                    f"{confidence:.3f}"
                )

                previous_action = (
                    action
                )

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            elapsed = (
                time.time()
                - loop_start
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

        listener.stop()

        print()
        print(
            "Online learning stopped."
        )


if __name__ == "__main__":

    main()