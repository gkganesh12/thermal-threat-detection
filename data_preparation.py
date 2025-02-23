import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from visualization import show_transformed_images, preview_batch

def get_augmentations(augmentation_level='medium'):
    """
    Define a set of image augmentations using Albumentations library
    
    Args:
        augmentation_level (str): Level of augmentation ('light', 'medium', 'heavy')
    Returns:
        A.Compose: Composition of augmentation transforms
    """
    # ImageNet normalization values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if augmentation_level == 'light':
        return A.Compose([
            A.Resize(height=224, width=224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(limit=15, p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    
    elif augmentation_level == 'medium':
        return A.Compose([
            A.Resize(height=224, width=224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(limit=20, p=0.5),
            A.GaussNoise(p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    
    else:  # heavy
        return A.Compose([
            A.Resize(height=224, width=224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.4),
            A.Rotate(limit=30, p=0.5),
            A.GaussNoise(p=0.3),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.4),
            A.Blur(blur_limit=3, p=0.2),
            A.ElasticTransform(alpha=1, sigma=50, p=0.2),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

class AugmentedDataset(Dataset):
    """
    Custom Dataset class that applies augmentations to images
    """
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform if transform is not None else get_augmentations()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # Convert PIL Image to numpy array for Albumentations
        image_np = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image_np)
            image = augmented['image']
            
        return image, label

def create_dataloaders(data_dir, batch_size=32, num_workers=4, augmentation_level='medium'):
    """
    Create train and validation dataloaders with augmentations
    
    Args:
        data_dir (str): Directory containing the dataset
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        augmentation_level (str): Level of augmentation ('light', 'medium', 'heavy')
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Define transforms for validation (only ToTensorV2)
    val_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Load datasets
    try:
        train_dataset = datasets.ImageFolder(root=f"{data_dir}/train")
        val_dataset = datasets.ImageFolder(root=f"{data_dir}/val")
        
        # Wrap datasets with augmentations
        train_dataset = AugmentedDataset(train_dataset, get_augmentations(augmentation_level))
        val_dataset = AugmentedDataset(val_dataset, val_transform)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    except Exception as e:
        print(f"Error creating dataloaders: {str(e)}")
        return None, None

def create_example_dataset(base_dir, num_classes=2, images_per_class=10):
    """
    Create an example dataset structure with random images
    
    Args:
        base_dir (str): Base directory for the dataset
        num_classes (int): Number of classes to create
        images_per_class (int): Number of images per class
    """
    # Create directory structure
    for split in ['train', 'val']:
        for class_idx in range(num_classes):
            os.makedirs(os.path.join(base_dir, split, f'class_{class_idx}'), exist_ok=True)
    
    # Create random images
    for split in ['train', 'val']:
        for class_idx in range(num_classes):
            for img_idx in range(images_per_class):
                # Create a random image
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                
                # Add some structure to make it more realistic
                cv2.circle(img, 
                          center=(112 + class_idx * 20, 112), 
                          radius=30 + class_idx * 10,
                          color=(200, 0, 0),
                          thickness=2)
                
                # Save the image
                img_path = os.path.join(base_dir, split, f'class_{class_idx}', f'img_{img_idx}.jpg')
                cv2.imwrite(img_path, img)
    
    print(f"Created example dataset at {base_dir}")
    print(f"- Number of classes: {num_classes}")
    print(f"- Images per class: {images_per_class}")
    print(f"- Total images: {num_classes * images_per_class * 2} (train + val)")

if __name__ == "__main__":
    # Create example dataset
    data_dir = "example_dataset"
    create_example_dataset(data_dir, num_classes=3, images_per_class=5)
    
    # Create dataloaders with different augmentation levels
    for aug_level in ['light', 'medium', 'heavy']:
        print(f"\nTesting {aug_level} augmentations:")
        train_loader, val_loader = create_dataloaders(
            data_dir,
            batch_size=4,
            num_workers=0,  # Set to 0 for easy debugging
            augmentation_level=aug_level
        )
        
        if train_loader and val_loader:
            # Show example of transformed images
            train_dataset = train_loader.dataset
            show_transformed_images(train_dataset, num_images=3)
            
            # Preview a batch
            preview_batch(train_loader)
