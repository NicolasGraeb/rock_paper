import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def default_preprocessing(image_size: int = 224) -> dict:
    return {"image_size": image_size, "resize": "square", "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)}


class EqualizeLuma:
    #wyrównanie histogramu tylko jasności (Y z YCrCb), kolory zostają
    def __call__(self, image: Image.Image) -> Image.Image:
        ycrcb = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2YCrCb)
        ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
        return Image.fromarray(cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB))


def build_transforms(image_size: int = 224, mean=IMAGENET_MEAN, std=IMAGENET_STD, equalize: bool = False):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            *([EqualizeLuma()] if equalize else []),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
