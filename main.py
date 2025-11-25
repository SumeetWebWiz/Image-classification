import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# -----------------------------
# 1. Config
# -----------------------------
BATCH_SIZE = 64
NUM_EPOCHS = 10
LR = 1e-3

# -----------------------------
# 2. Dataset & Dataloaders
# -----------------------------
def get_dataloaders():
    # CIFAR-10 mean & std (for normalization)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_dataset, test_dataset, train_loader, test_loader


# -----------------------------
# 3. Model
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # conv layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # input: [B, 3, 32, 32]
        x = self.pool(F.relu(self.conv1(x)))  # [B, 32, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, 8, 8]
        x = x.view(x.size(0), -1)             # [B, 64*8*8]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -----------------------------
# 4. Training function
# -----------------------------
def train(model, device, train_loader, criterion, optimizer, epoch, dataset_size):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / dataset_size
    epoch_acc = 100.0 * correct / total

    print(f"Epoch [{epoch}] "
          f"Train Loss: {epoch_loss:.4f}  |  Train Acc: {epoch_acc:.2f}%")


# -----------------------------
# 5. Evaluation function
# -----------------------------
def evaluate(model, device, test_loader, classes):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))


# -----------------------------
# 6. Optional: visualize some images
# -----------------------------
def show_sample_images(train_loader, classes):
    images, labels = next(iter(train_loader))
    # un-normalize for display
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

    images_unnorm = images * std + mean

    fig, axes = plt.subplots(1, 4, figsize=(8, 2))
    for i in range(4):
        img = images_unnorm[i].permute(1, 2, 0)  # CHW -> HWC
        axes[i].imshow(torch.clamp(img, 0, 1))
        axes[i].set_title(classes[labels[i]])
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


# -----------------------------
# 7. Main
# -----------------------------
def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Data
    train_dataset, test_dataset, train_loader, test_loader = get_dataloaders()
    classes = train_dataset.classes
    print("Classes:", classes)
    print("Train samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))

    # OPTIONAL: Show some sample images
    # Comment this out if you don't want the popup.
    show_sample_images(train_loader, classes)

    # Model, loss, optimizer
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        train(
            model,
            device,
            train_loader,
            criterion,
            optimizer,
            epoch,
            len(train_dataset),
        )

    print("Training finished.")

    # Evaluation
    evaluate(model, device, test_loader, classes)

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/cifar10_simplecnn.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
