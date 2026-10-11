import os
import random

import cv2
import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset

TRAIN_SESSIONS = [
    "recordings/session_001",
    "recordings/session_002",
    "recordings/session_003",
    "recordings/session_004",
    "recordings/session_004",
]

VAL_SESSION = "recordings/session_005"

FRAME_STACK = 6

BATCH_SIZE = 64

EPOCHS = 30

LEARNING_RATE = 0.0003

WEIGHT_DECAY = 0.0001

PATIENCE = 6

TRANSITION_WEIGHT = 3.0

MAX_HOLD_WEIGHT = 4.0

SEEDS = [
    42
]
class StrongGDDataset(Dataset):

    def __init__(
        self,
        session_dir,
        frame_stack=FRAME_STACK,
    ):

        self.session_dir = session_dir

        self.frame_stack = frame_stack

        csv_path = os.path.join(
            session_dir,
            "labels.csv"
        )

        self.data = pd.read_csv(
            csv_path
        )

        self.actions = (
            self.data["action"]
            .astype(int)
            .to_numpy()
        )

    def __len__(self):

        return (
            len(self.data)
            - self.frame_stack
            + 1
        )

    def __getitem__(
        self,
        index
    ):

        frames = []

        for offset in range(
            self.frame_stack
        ):

            row_index = (
                index + offset
            )

            filename = (
                self.data.iloc[
                    row_index
                ]["filename"]
            )

            path = os.path.join(
                self.session_dir,
                filename
            )

            frame = cv2.imread(
                path,
                cv2.IMREAD_GRAYSCALE
            )

            if frame is None:

                raise FileNotFoundError(
                    path
                )

            frame = (
                frame.astype(
                    np.float32
                )
                / 255.0
            )

            frames.append(
                frame
            )

        frames = np.stack(
            frames
        )

        label_index = (
            index
            + self.frame_stack
            - 1
        )

        label = int(
            self.actions[
                label_index
            ]
        )

        # ---------------------------------------------
        # Detect whether this frame is close to an
        # action transition.
        #
        # These are the most important frames because
        # they determine jump timing.
        # ---------------------------------------------

        start = max(
            0,
            label_index - 2
        )

        end = min(
            len(self.actions),
            label_index + 3
        )

        nearby_actions = (
            self.actions[
                start:end
            ]
        )

        near_transition = (
            np.any(
                nearby_actions
                != label
            )
        )

        sample_weight = (
            TRANSITION_WEIGHT
            if near_transition
            else 1.0
        )

        return (
            torch.tensor(
                frames,
                dtype=torch.float32
            ),

            torch.tensor(
                label,
                dtype=torch.long
            ),

            torch.tensor(
                sample_weight,
                dtype=torch.float32
            ),

            torch.tensor(
                int(
                    near_transition
                ),
                dtype=torch.long
            ),
        )


# =========================================================
# NETWORK
# =========================================================

class StrongDashNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                FRAME_STACK,
                32,
                kernel_size=5,
                stride=2,
                padding=2
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),


            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),


            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),


            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),


            nn.AvgPool2d(
                kernel_size=2
            ),
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 6 * 6,
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.25
            ),

            nn.Linear(
                256,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.15
            ),

            nn.Linear(
                64,
                2
            ),
        )

    def forward(
        self,
        x
    ):

        x = self.features(
            x
        )

        return self.classifier(
            x
        )


# =========================================================
# DEVICE
# =========================================================

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


# =========================================================
# RANDOM SEED
# =========================================================

def set_seed(seed):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )


# =========================================================
# CLASS WEIGHTS
# =========================================================

def get_class_weights(
    device
):

    labels = []

    for session in TRAIN_SESSIONS:

        csv_path = os.path.join(
            session,
            "labels.csv"
        )

        df = pd.read_csv(
            csv_path
        )

        session_labels = (
            df["action"]
            .iloc[
                FRAME_STACK - 1:
            ]
            .astype(int)
            .tolist()
        )

        labels.extend(
            session_labels
        )

    counts = np.bincount(
        labels,
        minlength=2
    )

    release_count = (
        counts[0]
    )

    hold_count = (
        counts[1]
    )

    ratio = (
        release_count
        / max(
            hold_count,
            1
        )
    )

    hold_weight = min(
        ratio,
        MAX_HOLD_WEIGHT
    )

    print(
        f"Release labels: "
        f"{release_count}"
    )

    print(
        f"Hold labels: "
        f"{hold_count}"
    )

    print(
        f"HOLD weight: "
        f"{hold_weight:.3f}"
    )

    return torch.tensor(
        [
            1.0,
            hold_weight
        ],
        dtype=torch.float32,
        device=device
    )


# =========================================================
# VALIDATION
# =========================================================

def evaluate(
    model,
    loader,
    device
):

    model.eval()

    total = 0
    correct = 0

    release_total = 0
    release_correct = 0

    hold_total = 0
    hold_correct = 0

    transition_total = 0
    transition_correct = 0

    with torch.no_grad():

        for (
            frames,
            labels,
            _,
            transitions
        ) in loader:

            frames = frames.to(
                device
            )

            labels = labels.to(
                device
            )

            transitions = (
                transitions.to(
                    device
                )
            )

            output = model(
                frames
            )

            predictions = (
                output.argmax(
                    dim=1
                )
            )

            matches = (
                predictions
                == labels
            )

            total += (
                labels.size(0)
            )

            correct += (
                matches
                .sum()
                .item()
            )

            # -----------------------------------------
            # RELEASE
            # -----------------------------------------

            mask = (
                labels == 0
            )

            release_total += (
                mask
                .sum()
                .item()
            )

            release_correct += (
                matches[
                    mask
                ]
                .sum()
                .item()
            )

            # -----------------------------------------
            # HOLD
            # -----------------------------------------

            mask = (
                labels == 1
            )

            hold_total += (
                mask
                .sum()
                .item()
            )

            hold_correct += (
                matches[
                    mask
                ]
                .sum()
                .item()
            )

            # -----------------------------------------
            # TRANSITION FRAMES
            # -----------------------------------------

            mask = (
                transitions == 1
            )

            transition_total += (
                mask
                .sum()
                .item()
            )

            transition_correct += (
                matches[
                    mask
                ]
                .sum()
                .item()
            )

    accuracy = (
        correct
        / max(
            total,
            1
        )
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
    ) / 2.0

    transition_accuracy = (
        transition_correct
        / max(
            transition_total,
            1
        )
    )

    # Give extra importance to jump timing.
    selection_score = (
        0.60
        * balanced_accuracy
        +
        0.40
        * transition_accuracy
    )

    return {
        "accuracy":
            accuracy,

        "release_recall":
            release_recall,

        "hold_recall":
            hold_recall,

        "balanced_accuracy":
            balanced_accuracy,

        "transition_accuracy":
            transition_accuracy,

        "selection_score":
            selection_score,
    }


# =========================================================
# TRAIN ONE MODEL
# =========================================================

def train_one_model(
    seed,
    model_number,
    train_loader,
    val_loader,
    weights,
    device,
):

    set_seed(
        seed
    )

    print()
    print(
        "=================================="
    )

    print(
        f"TRAINING MODEL {model_number}"
    )

    print(
        f"Seed: {seed}"
    )

    print(
        "=================================="
    )

    model = StrongDashNet().to(
        device
    )

    criterion = (
        nn.CrossEntropyLoss(
            weight=weights,
            reduction="none",
            label_smoothing=0.02,
        )
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
    )

    best_score = -1.0

    epochs_without_improvement = 0

    save_path = (
        f"models/"
        f"dash_strong_{model_number}.pth"
    )

    for epoch in range(
        EPOCHS
    ):

        model.train()

        running_loss = 0.0

        batches = 0

        for (
            frames,
            labels,
            sample_weights,
            _
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

            output = model(
                frames
            )

            losses = criterion(
                output,
                labels
            )

            loss = (
                losses
                * sample_weights
            ).mean()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0
            )

            optimizer.step()

            running_loss += (
                loss.item()
            )

            batches += 1

        metrics = evaluate(
            model,
            val_loader,
            device
        )

        average_loss = (
            running_loss
            / max(
                batches,
                1
            )
        )

        print()
        print(
            f"Epoch "
            f"{epoch + 1}/"
            f"{EPOCHS}"
        )

        print(
            f"Loss: "
            f"{average_loss:.4f}"
        )

        print(
            f"Accuracy: "
            f"{metrics['accuracy'] * 100:.2f}%"
        )

        print(
            f"Balanced: "
            f"{metrics['balanced_accuracy'] * 100:.2f}%"
        )

        print(
            f"HOLD recall: "
            f"{metrics['hold_recall'] * 100:.2f}%"
        )

        print(
            f"RELEASE recall: "
            f"{metrics['release_recall'] * 100:.2f}%"
        )

        print(
            f"Transition accuracy: "
            f"{metrics['transition_accuracy'] * 100:.2f}%"
        )

        print(
            f"Selection score: "
            f"{metrics['selection_score']:.4f}"
        )

        if (
            metrics[
                "selection_score"
            ]
            > best_score
        ):

            best_score = (
                metrics[
                    "selection_score"
                ]
            )

            torch.save(
                model.state_dict(),
                save_path
            )

            epochs_without_improvement = 0

            print(
                f"Saved {save_path}"
            )

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "Early stopping."
            )

            break

    print(
        f"Best score for model "
        f"{model_number}: "
        f"{best_score:.4f}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    os.makedirs(
        "models",
        exist_ok=True
    )

    device = get_device()

    print(
        f"Using device: "
        f"{device}"
    )

    train_dataset = (
        ConcatDataset(
            [
                StrongGDDataset(
                    session
                )
                for session
                in TRAIN_SESSIONS
            ]
        )
    )

    val_dataset = (
        StrongGDDataset(
            VAL_SESSION
        )
    )

    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    train_loader = (
        DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
    )

    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    )

    weights = get_class_weights(
        device
    )

    for model_number, seed in enumerate(
        SEEDS,
        start=1
    ):

        train_one_model(
            seed=seed,
            model_number=model_number,
            train_loader=train_loader,
            val_loader=val_loader,
            weights=weights,
            device=device,
        )

    print()
    print(
        "=================================="
    )

    print(
        "ALL STRONG MODELS TRAINED"
    )

    print(
        "=================================="
    )

    print(
        "Created:"
    )

    print(
        "models/dash_strong_1.pth"
    )

    print(
        "models/dash_strong_2.pth"
    )

    print(
        "models/dash_strong_3.pth"
    )


if __name__ == "__main__":
    main()