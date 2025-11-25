import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from main import SimpleCNN  # reuse the model class from main.py


def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=False,
        transform=transform,
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    return test_dataset, test_loader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1. Load test data
    test_dataset, test_loader = load_data()
    classes = test_dataset.classes

    # 2. Load model
    model = SimpleCNN(num_classes=10)
    model.load_state_dict(torch.load("models/cifar10_simplecnn.pth", map_location=device))
    model.to(device)
    model.eval()

    # 3. Take one random image and predict
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = outputs.max(1)

    true_label = classes[labels.item()]
    pred_label = classes[predicted.item()]

    print(f"True label:      {true_label}")
    print(f"Predicted label: {pred_label}")

    # 4. Prepare image for display (resize + unnormalize)
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(device)

    # upsample from 32x32 → 256x256 for nicer view
    img_resized = F.interpolate(images, size=(256, 256), mode="nearest")
    img = img_resized[0] * std + mean  # unnormalize
    img = img.cpu().permute(1, 2, 0)  # CHW → HWC

    plt.figure(figsize=(4, 4))
    plt.imshow(torch.clamp(img, 0, 1), interpolation="nearest")
    plt.title(f"True: {true_label} | Pred: {pred_label}")
    plt.axis("off")

    # also save a high-resolution image for your report / resume
    plt.tight_layout()
    plt.savefig("prediction_example.png", dpi=300)
    plt.show()

    print("Saved image to prediction_example.png")


if __name__ == "__main__":
    main()
