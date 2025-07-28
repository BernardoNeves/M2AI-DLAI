import argparse
from pathlib import Path

from train import train_model
from evaluate import evaluate_model
from grad_cam import generate_grad_cams


def main():
    parser = argparse.ArgumentParser(
        description="End-to-to-end pipeline for Fake Image Detection."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/deepfake-and-real-images",
        help="Path to dataset (e.g., data/deepfake-and-real-images)",
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
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for models and metrics",
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
        help="Maximum number of images to use from the dataset splits (train, validation, test)",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        help="Dataset split to evaluate on after training (e.g., test, validation)",
    )
    parser.add_argument(
        "--grad_cam_split",
        type=str,
        default="test",
        help="Dataset split to generate Grad-CAMs for (e.g., test, validation)",
    )
    parser.add_argument(
        "--grad_cam_max_images",
        type=int,
        default=25,
        help="Maximum number of images to use for Grad-CAM generation",
    )

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("\nStarting model training...")
    trained_model_output_dir = train_model(args)

    latest_model_dir = trained_model_output_dir

    if latest_model_dir:
        best_model_path = latest_model_dir / "best_model.pth"
        if best_model_path.exists():
            print(f"\nStarting model evaluation on {args.eval_split} set...")
            args.model_path = best_model_path
            evaluate_model(args)
        else:
            print(
                f"Warning: best_model.pth not found in {latest_model_dir}. Skipping evaluation."
            )
    else:
        print(
            f"Warning: No model directory found for {args.model}. Skipping evaluation."
        )

    if args.grad_cam_split:
        print(f"Starting Grad-CAM generation for {args.grad_cam_split} split...")
        args.output_folder = args.output_dir
        args.max_images = args.grad_cam_max_images
        generate_grad_cams(args)

    print("\nPipeline execution complete.")


if __name__ == "__main__":
    main()
