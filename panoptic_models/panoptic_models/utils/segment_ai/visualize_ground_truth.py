import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from segments import SegmentsClient, SegmentsDataset
from segments.utils import get_semantic_bitmap, export_dataset, load_image_from_url
import cv2
from panoptic_models.config.labels_info import _CATEGORIES
# Function to get color for a given label ID
def get_color_for_label(label_id):
    for category in _CATEGORIES:
        if category['id'] == label_id:
            return category['color']  # Return the color as [B, G, R]
    return [0, 0, 0]  # Default color if not found


def overlay_and_save_semantic_segmentations(dataset, save_dir='overlayed_images'):
    """
    Overlay semantic segmentation on images and save them with alpha transparency,
    using the color mapping provided in the _CATEGORIES list.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    for sample in dataset:
        image_array = np.array(sample['image'])  # Original image (H, W, C)
        semantic_image = get_semantic_bitmap(sample['segmentation_bitmap'], sample['annotations'])  # (H, W)

        # Create an empty image for the colored semantic overlay
        semantic_colored = np.zeros_like(image_array)

        # Map each label in the semantic image to its corresponding colora
        unique_labels = np.unique(semantic_image)
        for label in unique_labels:
            color = get_color_for_label(label)  # Get color for this label
            mask = semantic_image == label
            semantic_colored[mask] = color

        # Overlay the colored semantic image with the original image
        overlayed_image = cv2.addWeighted(image_array, 0.6, semantic_colored, 0.4, 0)

        # Plot overlayed image
        plt.imshow(cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
        plt.show()


# Example usage
api_key = 'your_api_key'
client = SegmentsClient(api_key)
release = client.get_release('leggedrobotics/construction_site', 'v8.0')
dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=['labeled'])
export_dataset(dataset, export_format='coco-panoptic')
overlay_and_save_semantic_segmentations(dataset)


