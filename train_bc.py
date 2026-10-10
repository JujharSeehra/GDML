import os
import cv2
import torch
import pandas as pd
import numpy as np

from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

SESSION_DIR = "recordings/session_002"
FRAME_STACK = 4
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

class GDDataset(Dataset):
    def __init__(self, session_dir, frame_stack = 4):
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
            if frame is None:
                raise FileNotFoundError(path)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        frames = np.stack(frames)
        label_index = index + self.frame_stack - 1
        label = int(self.data.iloc[label_index]["action"])
        return (torch.tensor(frames,dtype=torch.float32),torch.tensor(label,dtype=torch.long))
class DashNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(FRAME_STACK,32,kernel_size=5,stride=2),nn.ReLU(),nn.Conv2d(32,64,kernel_size=3,stride=2),nn.ReLU(),nn.Conv2d(64,128,kernel_size=3, stride=2),nn.ReLU(), nn.AvgPool2d(kernel_size=2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128*5*5,256),nn.ReLU(), nn.Dropout(0.2), nn.Linear(256,2))
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def main():
    os.makedirs("models", exist_ok=True)
    dataset = GDDataset(SESSION_DIR,FRAME_STACK)
    print(f"Total Samples: {len(dataset)}")
    split = int(len(dataset)*0.8)
    train_indices = list(range(0,split))
    val_indices = list(range(split, len(dataset)))
    train_dataset = Subset(dataset,train_indices)
    val_dataset = Subset(dataset, val_indices)
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using Device {device}")

    train_labels = []
    for i in train_indices:
        _, label = dataset[i]
        train_labels.append(label.item())
    counts = np.bincount(train_labels, minlength = 2)
    print(f"Release: {counts[0]}")
    print(f"Hold:  {counts[1]}")
    total = counts.sum()
    weights = total / (2*counts)
    weights=torch.tensor(weights,dtype=torch.float32,device=device)
    print(f"Class Weights {weights}")

    model = DashNet().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    best_val_accuracy = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        correct = 0
        total_samples = 0
        for frames, labels in train_loader:
            frames = frames.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
        train_accuracy = correct / total_samples

        model.eval()
        correct = 0
        total_samples = 0
        predicted_release = 0
        predicted_hold = 0
        actual_release = 0
        actual_hold = 0

        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(device)
                labels = labels.to(device)
                outputs = model(frames)
                predictions = outputs.argmax(dim=1)
                correct += (predictions==labels).sum().item()
                total_samples += labels.size(0)
                predicted_release += (predictions == 0).sum().item()
                predicted_hold += (predictions == 1).sum().item()
                actual_release += (labels == 0).sum().item()
                actual_hold += (labels == 1).sum().item()

        val_accuracy = (correct / total_samples)
        average_loss = (running_loss /len(train_loader))
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Loss: {average_loss:.4f}")
        print(f"Train accuracy: {train_accuracy * 100:.2f}%")
        print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
        print(f"Predicted: release={predicted_release}, hold={predicted_hold}")
        print(f"Actual:    release={actual_release}, hold={actual_hold}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(),"models/dash_bc.pth")
            print("Saved new best model.")
    print("Training complete.")
    print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()