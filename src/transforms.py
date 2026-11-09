import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def default_preprocessing(image_size: int = 224) -> dict:
    return {"image_size": image_size, "resize": "square", "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)}


def _on_luma(rgb, fn):
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = fn(ycrcb[..., 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def histeq(rgb):
    return _on_luma(rgb, cv2.equalizeHist)


def clahe(rgb):
    op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _on_luma(rgb, op.apply)


def gamma(rgb):
    #gamma dobierana tak, żeby średnia jasność wyszła ok. 0.5
    mean = float(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).mean()) / 255
    g = np.clip(np.log(0.5) / np.log(np.clip(mean, 0.01, 0.99)), 0.4, 2.5)
    lut = (255 * (np.arange(256) / 255) ** g).round().astype(np.uint8)
    return cv2.LUT(rgb, lut)


def grayscale(rgb):
    return cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)


def roi_center(rgb):
    #środkowy kwadrat 80% krótszego boku, bez detektora dłoni (bboxy HaGRID nie pasują do obrazów 512p)
    h, w = rgb.shape[:2]
    side = int(0.8 * min(h, w))
    y0, x0 = (h - side) // 2, (w - side) // 2
    return np.ascontiguousarray(rgb[y0:y0 + side, x0:x0 + side])


def gaussian_blur(rgb):
    return cv2.GaussianBlur(rgb, (0, 0), 1.5)


def white_balance(rgb):
    #gray world: średnia każdego kanału sprowadzona do wspólnej średniej
    means = rgb.reshape(-1, 3).mean(axis=0) + 1e-6
    return np.clip(rgb * (means.mean() / means), 0, 255).astype(np.uint8)


def sharpen(rgb):
    blurred = cv2.GaussianBlur(rgb, (0, 0), 1.0)
    return cv2.addWeighted(rgb, 2.0, blurred, -1.0, 0)


def canny(rgb):
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    return cv2.cvtColor(cv2.Canny(gray, 50, 150), cv2.COLOR_GRAY2RGB)


FILTERS = {
    "none": None,
    "histeq": histeq,
    "clahe": clahe,
    "gamma": gamma,
    "grayscale": grayscale,
    "roi_center": roi_center,
    "blur": gaussian_blur,
    "white_balance": white_balance,
    "sharpen": sharpen,
    "canny": canny,
}

#roi_center tnie oryginał przed resize, reszta działa na obrazie po resize
BEFORE_RESIZE = {"roi_center"}


class CvFilter:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, image: Image.Image) -> Image.Image:
        return Image.fromarray(FILTERS[self.name](np.asarray(image.convert("RGB"))))


class EqualizeLuma(CvFilter):
    #wyrównanie histogramu tylko jasności (Y z YCrCb), kolory zostają
    def __init__(self):
        super().__init__("histeq")


def build_transforms(image_size: int = 224, mean=IMAGENET_MEAN, std=IMAGENET_STD, equalize: bool = False,
                     image_filter: str = "none"):
    if equalize:
        image_filter = "histeq"
    if image_filter not in FILTERS:
        raise ValueError(f"nieznany filtr {image_filter}, dostępne: {list(FILTERS)}")
    f = [CvFilter(image_filter)] if image_filter != "none" else []
    before, after = (f, []) if image_filter in BEFORE_RESIZE else ([], f)
    return transforms.Compose(
        [
            *before,
            transforms.Resize((image_size, image_size)),
            *after,
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
