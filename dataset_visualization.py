import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from collections import Counter
import pandas as pd
import torch
from data_preparation import create_example_dataset, create_dataloaders

def plot_dataset_distribution(data_dir, title="Dataset Distribution"):
    """
    Plot the distribution of samples across classes in the dataset
    
    Args:
        data_dir (str): Path to the dataset directory
        title (str): Title for the plot
    """
    # Set up the plot style
    sns.set_theme(style="whitegrid")
    
    # Collect counts for train and val sets
    distributions = {}
    for split in ['train', 'val']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        class_counts = []
        labels = []
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                count = len(os.listdir(class_dir))
                class_counts.append(count)
                labels.append(class_name)
        
        distributions[split] = {
            'counts': class_counts,
            'labels': labels
        }
    
    # Create the visualization
    plt.figure(figsize=(10, 6))
    
    # Plot train and val distributions side by side
    bar_width = 0.35
    index = np.arange(len(distributions['train']['labels']))
    
    colors = sns.color_palette("husl", 2)
    plt.bar(index, distributions['train']['counts'], bar_width, 
            label='Train', color=colors[0], alpha=0.8)
    plt.bar(index + bar_width, distributions['val']['counts'], bar_width,
            label='Validation', color=colors[1], alpha=0.8)
    
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.xticks(index + bar_width/2, distributions['train']['labels'], fontsize=10)
    plt.legend(fontsize=10)
    
    # Add value labels on top of bars
    for i, count in enumerate(distributions['train']['counts']):
        plt.text(i, count, str(count), ha='center', va='bottom')
    for i, count in enumerate(distributions['val']['counts']):
        plt.text(i + bar_width, count, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def plot_class_examples(data_loader, num_examples=5, figsize=(15, 3)):
    """
    Plot example images from each class
    
    Args:
        data_loader: PyTorch DataLoader containing the dataset
        num_examples (int): Number of examples to show per class
        figsize (tuple): Figure size
    """
    # Get class names from the dataset
    class_names = data_loader.dataset.dataset.classes
    
    # Create a dictionary to store examples for each class
    class_examples = {class_name: [] for class_name in class_names}
    
    # Collect examples
    for images, labels in data_loader:
        for img, label in zip(images, labels):
            class_name = class_names[label]
            if len(class_examples[class_name]) < num_examples:
                class_examples[class_name].append(img)
        
        # Check if we have enough examples for each class
        if all(len(examples) >= num_examples for examples in class_examples.values()):
            break
    
    # Plot examples
    num_classes = len(class_names)
    fig, axes = plt.subplots(num_classes, num_examples, figsize=(figsize[0], figsize[1] * num_classes))
    
    for i, class_name in enumerate(class_names):
        for j, img in enumerate(class_examples[class_name][:num_examples]):
            if num_classes == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]
            
            # Convert tensor to numpy and transpose to correct format
            img_np = img.numpy().transpose(1, 2, 0)
            
            # Normalize if needed
            if img_np.max() > 1.0:
                img_np = img_np / 255.0
            
            # Display the image
            ax.imshow(img_np)
            ax.axis('off')
            
            # Add class name as title for the first image in each row
            if j == 0:
                ax.set_title(f'Class: {class_name}', fontsize=10, pad=5)
    
    plt.tight_layout()
    plt.show()

def plot_augmentation_comparison(dataset, image_index=0, num_augmentations=5):
    """
    Plot multiple augmented versions of the same image
    
    Args:
        dataset: Dataset with augmentations
        image_index (int): Index of the image to augment
        num_augmentations (int): Number of augmented versions to show
    """
    # Get the original image
    original_image, label = dataset.dataset[image_index]
    original_np = np.array(original_image)
    
    # Create augmented versions
    augmented_images = []
    for _ in range(num_augmentations):
        aug_image, _ = dataset[image_index]
        if isinstance(aug_image, torch.Tensor):
            aug_image = aug_image.numpy().transpose(1, 2, 0)
            if aug_image.max() > 1.0:
                aug_image = aug_image / 255.0
        augmented_images.append(aug_image)
    
    # Plot
    plt.figure(figsize=(15, 3))
    
    # Plot original
    plt.subplot(1, num_augmentations + 1, 1)
    plt.imshow(original_np)
    plt.title('Original', fontsize=10)
    plt.axis('off')
    
    # Plot augmented versions
    for i, aug_image in enumerate(augmented_images):
        plt.subplot(1, num_augmentations + 1, i + 2)
        plt.imshow(aug_image)
        plt.title(f'Augmented {i+1}', fontsize=10)
        plt.axis('off')
    
    plt.suptitle('Augmentation Examples', fontsize=12, y=1.05)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create example dataset if it doesn't exist
    data_dir = "example_dataset"
    if not os.path.exists(data_dir):
        create_example_dataset(data_dir, num_classes=3, images_per_class=5)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(data_dir, batch_size=4)
    
    if train_loader and val_loader:
        # Plot dataset distribution
        plot_dataset_distribution(data_dir, "Example Dataset Distribution")
        
        # Plot class examples
        print("\nShowing examples from each class:")
        plot_class_examples(train_loader, num_examples=3)
        
        # Plot augmentation comparison
        print("\nShowing augmentation examples:")
        plot_augmentation_comparison(train_loader.dataset, image_index=0, num_augmentations=4)
