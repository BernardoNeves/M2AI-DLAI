import argparse
import torch
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pathlib import Path
import json

from model import FakeImageDetector
from train import get_transforms
from preprocess import load_data


def generate_single_grad_cam(model_path, model_name, image_pil, image_size, device):
    model = FakeImageDetector(model_name=model_name, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    if 'vit' in model_name or 'swin' in model_name:
        target_layer = model.backbone.blocks[-1].norm1
    elif 'efficientnet' in model_name:
        target_layer = model.backbone.conv_head
    else:
        target_layer = model.backbone.layer4

    _, val_transform = get_transforms(image_size, augment=False)
    input_tensor = val_transform(image=np.array(image_pil))['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_class_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class_idx].item() * 100

    if 'vit' in model_name or 'swin' in model_name:
        def reshape_transform(tensor, height=14, width=14):
            result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
            result = result.transpose(2, 3).transpose(1, 2)
            return result
        cam = GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)
    else:
        cam = GradCAM(model=model, target_layers=[target_layer])

    targets = [ClassifierOutputTarget(predicted_class_idx)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    rgb_img = np.array(image_pil.resize((image_size, image_size))) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return Image.fromarray(visualization), predicted_class_idx, confidence


def generate_grad_cams(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_dir = Path(args.output_folder)
    gradcam_output_dir = Path('gradcam')
    gradcam_output_dir.mkdir(exist_ok=True)

    models_info = []
    for model_dir in output_dir.iterdir():
        if model_dir.is_dir():
            model_path = model_dir / 'best_model.pth'
            args_path = model_dir / 'args.json'
            if model_path.exists() and args_path.exists():
                with open(args_path, 'r') as f:
                    model_args = json.load(f)
                model_name = model_args['model']
                models_info.append({'path': model_path, 'name': model_name})

    if not models_info:
        print("No trained models found in the output folder. Skipping Grad-CAM generation.")
        return

    print(f"Loading images from {args.split} split...")
    image_paths, _ = load_data(args.data_dir, args.split, max_images=args.max_images)

    if not image_paths:
        print(f"No images found in {args.data_dir}/{args.split}. Skipping Grad-CAM generation.")
        return

    print(f"Loaded {len(image_paths)} images from {args.split} split for Grad-CAM.")

    for img_path_str in image_paths:
        img_path = Path(img_path_str)
        original_image = Image.open(img_path).convert('RGB').resize((args.image_size, args.image_size))
        images_to_collage = [original_image]

        print(f"Processing image: {img_path.name}")

        for model_info in models_info:
            print(f"  Generating Grad-CAM for model: {model_info['name']}")
            cam_image, predicted_class_idx, confidence = generate_single_grad_cam(
                model_info['path'], model_info['name'], original_image, args.image_size, device
            )
            images_to_collage.append(cam_image)
            class_names = ['Real', 'Fake']
            predicted_label = class_names[predicted_class_idx]
            print(f"    Model {model_info['name']} predicted: {predicted_label} (Confidence: {confidence:.2f}%)")

        widths, heights = zip(*(i.size for i in images_to_collage))
        total_width = sum(widths)
        max_height = max(heights)

        collage = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for img in images_to_collage:
            collage.paste(img, (x_offset, 0))
            x_offset += img.size[0]

        output_filename = f"grad_cam_{img_path.stem}.png"
        output_collage_path = gradcam_output_dir / output_filename
        collage.save(output_collage_path)
        print(f"Saved Grad-CAM collage for {img_path.name} to {output_collage_path}")

    print("Grad-CAM generation complete for all selected images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Grad-CAM for trained models on a dataset split.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder containing model directories')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the base dataset directory')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to generate Grad-CAMs for (e.g., test, validation)')
    parser.add_argument('--image_size', type=int, default=224, help='Image size used for training')
    parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images to load from the dataset split')

    args = parser.parse_args()
    generate_grad_cams(args)
