import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset_statistics(image_paths, labels):
    """Get basic statistics about the dataset"""
    stats = {
        "total_images": len(image_paths),
        "real_images": labels.count(0),
        "fake_images": labels.count(1),
        "real_percentage": (labels.count(0) / len(labels)) * 100,
        "fake_percentage": (labels.count(1) / len(labels)) * 100,
    }

    return stats


def print_dataset_info(image_paths, labels, split_name="Dataset"):
    """Print dataset information"""
    stats = get_dataset_statistics(image_paths, labels)
    print(f"\n{split_name} Statistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Real images: {stats['real_images']} ({stats['real_percentage']:.1f}%)")
    print(f"  Fake images: {stats['fake_images']} ({stats['fake_percentage']:.1f}%)")


def create_dataloader(
    image_paths, labels, transform, batch_size=32, shuffle=True, num_workers=4
):
    """Create a PyTorch DataLoader from image paths and labels"""

    class SimpleDataset(Dataset):
        def __init__(self, paths, labels, transform):
            self.paths = paths
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img_path = self.paths[idx]
            label = self.labels[idx]

            image = load_image(img_path)
            if image is None:
                image = np.zeros((224, 224, 3), dtype=np.uint8)

            if self.transform:
                if isinstance(self.transform, A.Compose):
                    transformed = self.transform(image=image)
                    image = transformed["image"]
                else:
                    image = Image.fromarray(image)
                    image = self.transform(image)

            return image, torch.tensor(label, dtype=torch.long)

    dataset = SimpleDataset(image_paths, labels, transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True if shuffle else False,
    )


def load_image(img_path):
    """Safely load an image with error handling"""
    try:
        image = cv2.imread(str(img_path))
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None


def load_data(base_data_dir, split_name, max_images=None):
    """Discover images from directory structure (real/ and fake/ subdirs within a split)"""
    split_dir = Path(base_data_dir) / split_name
    image_paths = []
    labels = []

    real_dir = split_dir / "real"
    if real_dir.exists():
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        for ext in extensions:
            for img_path in real_dir.glob(ext):
                image_paths.append(str(img_path))
                labels.append(0)

    fake_dir = split_dir / "fake"
    if fake_dir.exists():
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        for ext in extensions:
            for img_path in fake_dir.glob(ext):
                image_paths.append(str(img_path))
                labels.append(1)

    if max_images is not None:
        combined = list(zip(image_paths, labels))
        label_0 = [item for item in combined if item[1] == 0]
        label_1 = [item for item in combined if item[1] == 1]

        n = min(len(label_0), len(label_1), max_images // 2)

        combined = random.sample(label_0, n) + random.sample(label_1, n)
        random.shuffle(combined)

        image_paths, labels = zip(*combined)
        image_paths = list(image_paths)
        labels = list(labels)

    return image_paths, labels
