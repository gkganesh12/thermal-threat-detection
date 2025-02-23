import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
import albumentations as A
from albumentations.pytorch import ToTensorV2

def show_transformed_images(dataset, num_images=5, figsize=(15, 5)):
    """
    Display original and augmented images side by side
    
    Args:
        dataset: Dataset containing images
        num_images (int): Number of images to display
        figsize (tuple): Figure size for the plot
    """
    fig, axes = plt.subplots(2, num_images, figsize=figsize)
    
    for i in range(num_images):
        # Get original image
        image, label = dataset.dataset[i]
        image_np = np.array(image)
        
        # Get augmented image
        aug_image, _ = dataset[i]
        
        # Convert tensor to numpy if necessary
        if isinstance(aug_image, torch.Tensor):
            aug_image = aug_image.numpy().transpose(1, 2, 0)
            
            # Normalize to [0, 1] range if needed
            if aug_image.max() > 1.0:
                aug_image = aug_image / 255.0
        
        # Display images
        axes[0, i].imshow(image_np)
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(aug_image)
        axes[1, i].set_title(f'Augmented {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def preview_batch(dataloader, num_batches=1):
    """
    Preview batches from the dataloader
    
    Args:
        dataloader: PyTorch DataLoader
        num_batches (int): Number of batches to preview
    """
    for i, (images, labels) in enumerate(dataloader):
        if i >= num_batches:
            break
            
        # Convert images to float and normalize if needed
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
            
        # Create a grid of images
        grid = make_grid(images, nrow=8, normalize=True, pad_value=1)
        
        # Convert to numpy for display
        grid_np = grid.numpy().transpose(1, 2, 0)
        
        # Ensure values are in [0, 1] range
        grid_np = np.clip(grid_np, 0, 1)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(grid_np)
        plt.title(f'Batch {i+1} Preview')
        plt.axis('off')
        plt.show()
        
        print(f'Batch {i+1} shape: {images.shape}')
        print(f'Labels: {labels.numpy()}\n')
