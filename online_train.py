import glob
import os
import shutil

import cv2
import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset,
)

from train_strong import StrongDashNet

from online_config import (
    TRAIN_SESSIONS,
    VAL_SESSION,
    FRAME_STACK,
    ONLINE_MODEL_PATH,
    CANDIDATE_MODEL_PATH,
    ONLINE_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    EXPERT_SAMPLE_WEIGHT,
    CORRECTION_SAMPLE_WEIGHT,
    ELITE_SAMPLE_WEIGHT,
    MAX_HOLD_CLASS_WEIGHT,
    ACTION_LOOKAHEAD,
)


# =========================================================
# ORIGINAL HUMAN DATA
# =========================================================

class ExpertDataset(Dataset):

    def __init__(
        self,
        session_dir,
        sample_weight=1.0,
    ):

        self.session_dir = session_dir

        self.sample_weight = sample_weight

        csv_path = os.path.join(
            session_dir,
            "labels.csv"
        )

        self.data = pd.read_csv(csv_path)

    def __len__(self):

        return (
        len(self.data)
        - FRAME_STACK
        - ACTION_LOOKAHEAD
        + 1
        )

    def __getitem__(self, index):

        frames = []

        for offset in range(FRAME_STACK):

            row_index = index + offset

            filename = self.data.iloc[
                row_index
            ]["filename"]

            path = os.path.join(
                self.session_dir,
                filename
            )

            frame = cv2.imread(
                path,
                cv2.IMREAD_GRAYSCALE
            )

            if frame is None:
                raise FileNotFoundError(path)

            frames.append(frame)

        frames = np.stack(
            frames
        ).astype(
            np.float32
        ) / 255.0

        label_index = (
    index
    + FRAME_STACK
    - 1
    + ACTION_LOOKAHEAD
)

        action = int(
            self.data.iloc[
                label_index
            ]["action"]
        )

        return (
            torch.tensor(
                frames,
                dtype=torch.float32
            ),
            torch.tensor(
                action,
                dtype=torch.long
            ),
            torch.tensor(
                self.sample_weight,
                dtype=torch.float32
            ),
        )


# =========================================================
# HUMAN CORRECTIONS
# =========================================================

class CorrectionDataset(Dataset):

    def __init__(self):

        self.samples = []

        files = sorted(
            glob.glob(
                "online_data/corrections/*.npz"
            )
        )

        for path in files:

            data = np.load(path)

            frames = data[
                "frames"
            ].astype(np.uint8)

            action = int(
                data["action"]
            )

            self.samples.append(
                (
                    frames,
                    action
                )
            )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(self, index):

        frames, action = (
            self.samples[index]
        )

        frames = (
            frames.astype(
                np.float32
            )
            / 255.0
        )

        return (
            torch.tensor(
                frames,
                dtype=torch.float32
            ),
            torch.tensor(
                action,
                dtype=torch.long
            ),
            torch.tensor(
                CORRECTION_SAMPLE_WEIGHT,
                dtype=torch.float32
            ),
        )


# =========================================================
# SUCCESSFUL AI TRAJECTORIES
# =========================================================

class EliteDataset(Dataset):

    def __init__(self):

        self.trajectories = []

        self.index = []

        paths = sorted(
            glob.glob(
                "online_data/elite/*.npz"
            )
        )

        for path in paths:

            data = np.load(path)

            frames = data[
                "frames"
            ].astype(np.uint8)

            actions = data[
                "actions"
            ].astype(np.int64)

            trajectory_index = len(
                self.trajectories
            )

            self.trajectories.append(
                (
                    frames,
                    actions
                )
            )

            for i in range(
                FRAME_STACK - 1,
                len(frames)
            ):

                self.index.append(
                    (
                        trajectory_index,
                        i
                    )
                )

    def __len__(self):

        return len(
            self.index
        )

    def __getitem__(self, index):

        trajectory_index, i = (
            self.index[index]
        )

        frames, actions = (
            self.trajectories[
                trajectory_index
            ]
        )

        stack = frames[
            i - FRAME_STACK + 1:
            i + 1
        ]

        stack = (
            stack.astype(
                np.float32
            )
            / 255.0
        )

        action = int(
            actions[i]
        )

        return (
            torch.tensor(
                stack,
                dtype=torch.float32
            ),
            torch.tensor(
                action,
                dtype=torch.long
            ),
            torch.tensor(
                ELITE_SAMPLE_WEIGHT,
                dtype=torch.float32
            ),
        )


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
# CLASS WEIGHTS
# =========================================================

def get_class_weights(device):

    labels = []

    # Human demonstrations
    for session in TRAIN_SESSIONS:

        path = os.path.join(
            session,
            "labels.csv"
        )

        df = pd.read_csv(path)

        session_labels = (
    df["action"]
    .iloc[
        FRAME_STACK - 1 + ACTION_LOOKAHEAD:
    ]
    .astype(int)
    .tolist()
)

        labels.extend(
            session_labels
        )

    # Human corrections
    for path in glob.glob(
        "online_data/corrections/*.npz"
    ):

        data = np.load(path)

        labels.append(
            int(data["action"])
        )

    counts = np.bincount(
        labels,
        minlength=2
    )

    release_count = max(
        int(counts[0]),
        1
    )

    hold_count = max(
        int(counts[1]),
        1
    )

    ratio = (
        release_count
        / hold_count
    )

    hold_weight = min(
        ratio,
        MAX_HOLD_CLASS_WEIGHT
    )

    weights = torch.tensor(
        [
            1.0,
            hold_weight
        ],
        dtype=torch.float32,
        device=device
    )

    print(
        "Human label counts:"
    )

    print(
        f"  RELEASE: {release_count}"
    )

    print(
        f"  HOLD:    {hold_count}"
    )

    print(
        f"Class weights: {weights}"
    )

    return weights


# =========================================================
# VALIDATION
# =========================================================

def evaluate(
    model,
    loader,
    device
):

    model.eval()

    release_correct = 0
    release_total = 0

    hold_correct = 0
    hold_total = 0

    total_correct = 0
    total = 0

    with torch.no_grad():

        for (
            frames,
            labels,
            _
        ) in loader:

            frames = frames.to(
                device
            )

            labels = labels.to(
                device
            )

            output = model(
                frames
            )

            predictions = (
                output.argmax(
                    dim=1
                )
            )

            total_correct += (
                predictions
                == labels
            ).sum().item()

            total += labels.size(0)

            release_mask = (
                labels == 0
            )

            hold_mask = (
                labels == 1
            )

            release_total += (
                release_mask
                .sum()
                .item()
            )

            hold_total += (
                hold_mask
                .sum()
                .item()
            )

            release_correct += (
                (
                    predictions[
                        release_mask
                    ]
                    == 0
                )
                .sum()
                .item()
            )

            hold_correct += (
                (
                    predictions[
                        hold_mask
                    ]
                    == 1
                )
                .sum()
                .item()
            )

    accuracy = (
        total_correct
        / max(total, 1)
    )

    release_recall = (
        release_correct
        / max(
            release_total,
            1
        )
    )

    hold_recall = (
        hold_correct
        / max(
            hold_total,
            1
        )
    )

    balanced_accuracy = (
        release_recall
        + hold_recall
    ) / 2

    return {
        "accuracy":
            accuracy,

        "release_recall":
            release_recall,

        "hold_recall":
            hold_recall,

        "balanced_accuracy":
            balanced_accuracy,
    }


# =========================================================
# ONLINE TRAIN
# =========================================================

def train_online_model():

    device = get_device()

    print()
    print(
        "===== ONLINE TRAINING ====="
    )

    print(
        f"Device: {device}"
    )

    datasets = []

    # Human data
    for session in TRAIN_SESSIONS:

        datasets.append(
            ExpertDataset(
                session,
                EXPERT_SAMPLE_WEIGHT
            )
        )

    # Corrections
    correction_dataset = (
        CorrectionDataset()
    )

    if len(
        correction_dataset
    ) > 0:

        datasets.append(
            correction_dataset
        )

    # Good AI trajectories
    elite_dataset = (
        EliteDataset()
    )

    if len(
        elite_dataset
    ) > 0:

        datasets.append(
            elite_dataset
        )

    train_dataset = (
        ConcatDataset(
            datasets
        )
    )

    val_dataset = (
        ExpertDataset(
            VAL_SESSION,
            1.0
        )
    )

    train_loader = (
        DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )
    )

    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
    )

    print(
        f"Human correction samples: "
        f"{len(correction_dataset)}"
    )

    print(
        f"Elite samples: "
        f"{len(elite_dataset)}"
    )

    print(
        f"Total training samples: "
        f"{len(train_dataset)}"
    )

    # --------------------------------------------------
    # CURRENT MODEL
    # --------------------------------------------------

    old_model = (
        StrongDashNet().to(
            device
        )
    )

    old_model.load_state_dict(
        torch.load(
            ONLINE_MODEL_PATH,
            map_location=device
        )
    )

    old_metrics = evaluate(
        old_model,
        val_loader,
        device
    )

    print()
    print(
        "Current validation:"
    )

    print(
        old_metrics
    )

    # --------------------------------------------------
    # CANDIDATE MODEL
    # --------------------------------------------------

    model = (
        StrongDashNet().to(
            device
        )
    )

    model.load_state_dict(
        torch.load(
            ONLINE_MODEL_PATH,
            map_location=device
        )
    )

    class_weights = (
        get_class_weights(
            device
        )
    )

    criterion = (
        nn.CrossEntropyLoss(
            weight=class_weights,
            reduction="none"
        )
    )

    optimizer = (
        torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE
        )
    )

    for epoch in range(
        ONLINE_EPOCHS
    ):

        model.train()

        running_loss = 0.0

        batches = 0

        for (
            frames,
            labels,
            sample_weights
        ) in train_loader:

            frames = frames.to(
                device
            )

            labels = labels.to(
                device
            )

            sample_weights = (
                sample_weights.to(
                    device
                )
            )

            optimizer.zero_grad()

            outputs = model(
                frames
            )

            losses = criterion(
                outputs,
                labels
            )

            loss = (
                losses
                * sample_weights
            ).mean()

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
            )

            batches += 1

        print(
            f"Online epoch "
            f"{epoch + 1}/"
            f"{ONLINE_EPOCHS} | "
            f"loss="
            f"{running_loss / max(batches, 1):.4f}"
        )

    new_metrics = evaluate(
        model,
        val_loader,
        device
    )

    print()
    print(
        "Candidate validation:"
    )

    print(
        new_metrics
    )

    torch.save(
        model.state_dict(),
        CANDIDATE_MODEL_PATH
    )

    # --------------------------------------------------
    # SAFETY GATE
    # --------------------------------------------------
    #
    # Don't accept a model that badly damages
    # independent validation performance.
    #

    balanced_ok = (
        new_metrics[
            "balanced_accuracy"
        ]
        >=
        old_metrics[
            "balanced_accuracy"
        ] - 0.01
    )

    hold_ok = (
        new_metrics[
            "hold_recall"
        ]
        >=
        old_metrics[
            "hold_recall"
        ] - 0.05
    )

    accepted = (
        balanced_ok
        and hold_ok
    )

    if accepted:

        shutil.copy2(
            CANDIDATE_MODEL_PATH,
            ONLINE_MODEL_PATH
        )

        print()
        print(
            "ACCEPTED candidate model."
        )

    else:

        print()
        print(
            "REJECTED candidate model."
        )

        print(
            "Keeping previous policy."
        )

    print(
        "==========================="
    )

    return accepted


if __name__ == "__main__":

    train_online_model()