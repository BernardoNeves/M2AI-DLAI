import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import FakeImageDetector
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
import numpy as np
from tqdm import tqdm

from preprocess import set_seed, load_data, create_dataloader, print_dataset_info
from utils import plot_confusion_matrix, plot_training_history, save_checkpoint


def get_transforms(image_size=224, augment=True):
    """Get data transforms for training and validation"""

    if augment:
        train_transform = A.Compose(
            [
                A.Resize(image_size + 32, image_size + 32),
                A.RandomCrop(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3
                ),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.Affine(
                    scale=(0.9, 1.1), translate_percent=0.1, rotate=(-15, 15), p=0.3
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        train_transform = A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    val_transform = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    return train_transform, val_transform


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    predictions = []
    targets = []

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        targets.extend(labels.cpu().numpy())

        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Avg Loss": f"{running_loss / (batch_idx + 1):.4f}",
            }
        )

    accuracy = accuracy_score(targets, predictions)
    avg_loss = running_loss / len(dataloader)

    return avg_loss, accuracy, predictions, targets


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []
    probabilities = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Avg Loss": f"{running_loss / (batch_idx + 1):.4f}",
                }
            )

    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average="weighted"
    )

    probs_array = np.array(probabilities)
    auc_roc = roc_auc_score(targets, probs_array[:, 1])

    avg_loss = running_loss / len(dataloader)

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
    }

    return metrics, predictions, targets, probabilities


def train_model(args):
    set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.model}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading training dataset...")
    train_paths, train_labels = load_data(
        args.data_dir, "train", max_images=args.max_images
    )

    if not train_paths:
        raise ValueError(f"No images found in {args.data_dir}/train")

    print_dataset_info(train_paths, train_labels, "Training Set")

    print("Loading validation dataset...")
    val_paths, val_labels = load_data(
        args.data_dir, "validation", max_images=args.max_images
    )

    if not val_paths:
        raise ValueError(f"No images found in {args.data_dir}/validation")

    print_dataset_info(val_paths, val_labels, "Validation Set")

    train_transform, val_transform = get_transforms(
        args.image_size, not args.no_augment
    )

    train_loader = create_dataloader(
        train_paths,
        train_labels,
        train_transform,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
    )
    val_loader = create_dataloader(
        val_paths,
        val_labels,
        val_transform,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
    )

    print(f"Creating model: {args.model}")
    model = FakeImageDetector(
        model_name=args.model, num_classes=2, pretrained=True, dropout_rate=0.2
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = None

    scaler = torch.amp.GradScaler("cuda") if args.mixed_precision else None

    print("Starting training...")
    train_losses = []
    val_metrics_history = []
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)

        train_loss, train_acc, _, _ = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        train_losses.append(train_loss)

        val_metrics, val_preds, val_targets, _ = validate_epoch(
            model, val_loader, criterion, device
        )
        val_metrics_history.append(val_metrics)

        if scheduler:
            if args.scheduler == "plateau":
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(
            f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"
        )
        print(
            f"Val F1: {val_metrics['f1']:.4f}, Val AUC-ROC: {val_metrics['auc_roc']:.4f}"
        )

        is_best = val_metrics["accuracy"] > best_val_acc
        if is_best:
            best_val_acc = val_metrics["accuracy"]
            print(f"New best validation accuracy: {best_val_acc:.4f}")

        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch + 1,
            val_metrics,
            checkpoint_path,
            is_best,
        )

        if is_best:
            plot_confusion_matrix(
                val_targets,
                val_preds,
                ["Real", "Fake"],
                output_dir / "confusion_matrix.png",
            )

    plot_training_history(
        train_losses, val_metrics_history, output_dir / "training_history.png"
    )

    final_metrics = {
        "best_val_accuracy": best_val_acc,
        "final_val_metrics": val_metrics_history[-1],
        "total_epochs": args.epochs,
        "model_name": args.model,
    }

    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2, default=str)

    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")
    print(f"Results saved to: {output_dir}")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fake Image Detection Model")
    parser.add_argument(
        "--data_dir", type=str, default="data/FACE", help="Path to dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnet_b0",
        choices=[
            "efficientnet_b0",
            "efficientnet_b4",
            "resnet50",
            "vit_base_patch16_224",
            "vit_small_patch16_224",
        ],
        help="Model architecture",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory"
    )
    parser.add_argument(
        "--no_augment", action="store_true", help="Disable data augmentation"
    )
    parser.add_argument(
        "--mixed_precision", action="store_true", help="Use mixed precision training"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "cosine", "none"],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to load from the dataset",
    )

    args = parser.parse_args()
    train_model(args)
