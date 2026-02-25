"""Image and mask transforms for training and validation."""

from torchvision import transforms

_geometric = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
])

train_image_transform = transforms.Compose([
    _geometric,
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_mask_transform = transforms.Compose([
    _geometric,
    transforms.ToTensor(),
])

val_image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_mask_transform = transforms.Compose([
    transforms.ToTensor(),
])
