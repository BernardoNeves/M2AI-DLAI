# M2AI-DLAI
Bernardo Neves - a23494

[Github Repo](https://github.com/BernardoNeves/M2AI-DLAI)
[Dataset Used](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)

## Setup Environment
```bash
conda env create -f environment.yml
conda activate m2ai-dlai
```

## Usage

```bash
python main.py --model efficientnet_b0 --data_dir data/deepfake-and-real-images
```

```bash
python main.py \
    --data_dir data/deepfake-and-real-images \
    --model vit_small_patch16_224 \
    --epochs 25 \
    --batch_size 16 \
    --lr 1e-5 \
    --image_size 224 \
    --mixed_precision \
    --scheduler cosine \
    --grad_cam_split test
```

### Arguments

*   `--data_dir`: Path to the dataset.
*   `--model`: Model architecture to use.
*   `--epochs`: Number of training epochs.
*   `--batch_size`: Training batch size.
*   `--lr`: Learning rate.
*   `--image_size`: Input image size.
*   `--seed`: Random seed.
*   `--output_dir`: Directory to save models and metrics.
*   `--no_augment`: Disable data augmentation.
*   `--mixed_precision`: Enable mixed precision training.
*   `--scheduler`: Learning rate scheduler.
*   `--max_images`: Maximum number of images to use from the dataset.
*   `--eval_split`: Dataset split to evaluate on after training (e.g., `test`, `validation`).
*   `--grad_cam_split`: Dataset split to generate Grad-CAMs for (e.g., `test`, `validation`).
