import torch
from PIL import Image
from pyiqa import create_metric
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
import numpy as np


def assess_image_quality(image_path1, image_path2=None):
    """
    Assess image quality using multiple pre-trained models.

    Parameters:
    image_path (str): Path to the image file
    metrics (list): List of metrics to use. Options include:
                   'niqe': Natural Image Quality Evaluator
                   'brisque': Blind/Referenceless Image Spatial Quality Evaluator
                   'musiq': Multi-scale Image Quality Transformer
                   'dbcnn': Deep Blind Convolutional Neural Network
                   'ilniqe': Integrated Local NIQE
                   'ma': Most Apparent Distortion

    Returns:
    dict: Dictionary containing quality scores from different metrics
    """
    metric = create_metric("musiq")  # ["niqe", "brisque", "musiq", "dbcnn", "ilniqe"]
    metric.eval()
    if torch.cuda.is_available():
        metric.cuda()
    score = metric(image_path1).item()
    return score


def calculate_similarity_score(image_path1, image_path2):
    # Load the images
    image1 = imread(image_path1, as_gray=True)
    image2 = imread(image_path2, as_gray=True)

    # Compute SSIM between the images
    score, _ = ssim(image1, image2, full=True)

    return score


# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = "output-imgs/110.png"

    # You can specify which metrics to use
    # Example with just two metrics:
    # print(image_path)
    score = assess_image_quality(image_path)
    # Or use default metrics:
    # results = assess_image_quality(image_path)

    # print(score)
