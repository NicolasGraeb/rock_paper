from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def default_preprocessing(image_size: int = 224) -> dict:
    return {"image_size": image_size, "resize": "square", "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)}


def build_transforms(image_size: int = 224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
