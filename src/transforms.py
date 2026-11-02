from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def default_preprocessing(image_size: int = 224) -> dict:
    """Preprocessing checkpointów sprzed zapisu klucza "preprocessing" (wszystkie używały ImageNet mean/std)."""
    return {"image_size": image_size, "resize": "square", "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)}


def build_transforms(image_size: int = 128, train: bool = False, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    # Augmentacja wyłącznie dla train; val/test/external: tylko resize + normalizacja.
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
