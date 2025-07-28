import argparse
import torch
from pathlib import Path
import json

from model import FakeImageDetector
from train import get_transforms, validate_epoch
from preprocess import load_data, create_dataloader, print_dataset_info
from utils import plot_confusion_matrix
from preprocess import load_data, create_dataloader, print_dataset_info

def evaluate_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model_name = checkpoint.get('model_name', 'efficientnet_b0')

    model = FakeImageDetector(model_name=model_name, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loading {args.split} dataset...")
    image_paths, labels = load_data(args.data_dir, args.split, max_images=args.max_images)

    if not image_paths:
        raise ValueError(f"No images found in {args.data_dir}/{args.split}")

    print_dataset_info(image_paths, labels, f"{args.split.capitalize()} Set")

    _, val_transform = get_transforms(args.image_size, augment=False)

    dataloader = create_dataloader(
        image_paths, labels, val_transform,
        batch_size=args.batch_size, shuffle=False, num_workers=8
    )

    criterion = torch.nn.CrossEntropyLoss()

    print(f"Evaluating on {args.split} set...")
    metrics, predictions, targets, probabilities = validate_epoch(
        model, dataloader, criterion, device
    )

    print(f"\n--- {args.split.capitalize()} Evaluation Results ---")
    for metric_name, value in metrics.items():
        print(f"  {metric_name.replace('_', ' ').capitalize()}: {value:.4f}")

    class_names = ['Real', 'Fake']
    print("\n--- Individual Predictions with Confidence ---")
    for i, (pred, target, prob) in enumerate(zip(predictions, targets, probabilities)):
        predicted_class = class_names[pred]
        true_class = class_names[target]
        confidence = prob[pred] * 100
        print(f"Image {i+1}: True: {true_class}, Predicted: {predicted_class} (Confidence: {confidence:.2f}%)")

    output_dir = Path(args.output_dir) / Path(args.model_path).parent.name
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_save_path = output_dir / f'{args.split}_metrics.json'
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation metrics saved to {metrics_save_path}")

    confusion_matrix_path = output_dir / f'{args.split}_confusion_matrix.png'
    plot_confusion_matrix(targets, predictions, ['Real', 'Fake'], confusion_matrix_path)
    print(f"Confusion matrix saved to {confusion_matrix_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a trained model on a dataset split.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the base dataset directory')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to evaluate on (e.g., train, validation, test)')
    parser.add_argument('--image_size', type=int, default=224, help='Image size used for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory for evaluation results')
    parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images to load from the dataset split')

    args = parser.parse_args()
    evaluate_model(args)
