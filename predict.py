import argparse
from pathlib import Path

import cv2 as cv
import torch
from PIL import Image
from torchvision import transforms

from network import UNet
from util import seg2img


def predict_image(src_path: str, dst_path: str, model_path: str) -> None:
    """Generate a segmentation image from a single input image.

    Args:
        src_path: Path to the source image (should be square 512x512).
        dst_path: File path to save the resulting segmentation image.
        model_path: Path to the trained model weights.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    transform = transforms.Compose(
        [transforms.Resize(512), transforms.ToTensor()]
    )

    model.eval()
    with torch.no_grad():
        img = Image.open(src_path)
        img = transform(img).unsqueeze(0).to(device)

        seg = model(img).squeeze(0).cpu().numpy()
        result = seg2img(seg)
        cv.imwrite(str(dst_path), result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a segmentation map for a single image."
    )
    parser.add_argument("src_path", help="path of source image (square)")
    parser.add_argument("dst_path", help="path to save segmentation image")
    parser.add_argument(
        "--model_path",
        default="model/UNet.pth",
        help="path to load trained U-Net model",
    )
    args = parser.parse_args()

    predict_image(args.src_path, args.dst_path, args.model_path)
