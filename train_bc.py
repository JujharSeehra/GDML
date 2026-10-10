import os
import cv2
import torch
import pandas as pd
import numpy as np

from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split


SESSION_DIR = "recordings/session_002"

BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001

class GDDataset(Dataset):
    def __init__(self, session_dir, frame_stack=4):
        self.session_dir = session_dir
        self.frame_stack = frame_stack

        csv_path = os.path.join(session_dir, "labels.csv")

        self.data = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.data) - self.frame_stack + 1

    def __getitem__(self, index):
        frames = []
        for offset in range(self.frame_stack):
            row_index = index + offset

            filename = self.data.iloc[row_index]["filename"]

            path = os.path.join(self.session_dir, filename)

            frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        frames = np.stack(frames)
        label_index = index + self.frame_stack - 1

        label = int(self.data.iloc[label_index]["action"] )

        frames = torch.tensor(frames,dtype=torch.float32)

        label = torch.tensor(label,dtype=torch.long
        )

        return frames, label


# --------------------------------------------------
# CNN
# --------------------------------------------------

class DashNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(nn.Conv2d(4, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear( 256, 2))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def main():
    os.makedirs("models", exist_ok=True)
    dataset = GDDataset(SESSION_DIR)
    print(f"Total samples: {len(dataset)}")
    train_size = int(len(dataset) * 0.8)
    val_size = (len(dataset) - train_size)
    train_dataset, val_dataset = random_split(dataset,[train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    model = DashNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for frames, labels in train_loader:

            frames = frames.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_accuracy = (correct / total)
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():

            for frames, labels in val_loader:

                frames = frames.to(device)
                labels = labels.to(device)
                outputs = model(frames)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        val_accuracy = (correct / total)

        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss / len(train_loader):.4f} | Train: {train_accuracy * 100:.2f}% | Val: {val_accuracy * 100:.2f}%")

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy
            torch.save(model.state_dict(), "models/dash_bc.pth")
            print("Saved new best model.")

    print(f"Best validation accuracy: {best_accuracy * 100:.2f}%")
if __name__ == "__main__":
    main()