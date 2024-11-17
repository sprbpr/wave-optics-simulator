import cv2
import numpy as np
from scipy.ndimage import variance
from skimage.feature import local_binary_pattern


def assess_image_quality(image_path):
    """
    Assess the quality of an image using multiple metrics including edge quality.

    Parameters:
    image_path (str): Path to the image file

    Returns:
    dict: Dictionary containing various quality metrics and overall score
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image")

    # Convert to grayscale for some calculations
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Brightness Assessment
    def assess_brightness():
        brightness = np.mean(gray)
        # Normalize to 0-1 range where 0.5 is optimal
        return 1 - abs(brightness - 127.5) / 127.5

    # 2. Contrast Assessment
    def assess_contrast():
        contrast = np.std(gray)
        # Normalize to 0-1 range
        return min(contrast / 80, 1)  # 80 is a reasonable contrast value

    # 3. Blur Detection using Laplacian variance
    def detect_blur():
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize to 0-1 range where higher is better (less blurry)
        return min(laplacian_var / 500, 1)  # 500 is a threshold for sharp images

    # 4. Noise Estimation using Local Binary Pattern
    def estimate_noise():
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
        # Calculate histogram of LBP
        hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), density=True)
        # Use entropy of LBP histogram as noise measure
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        # Normalize to 0-1 range where higher is better (less noise)
        return max(1 - entropy / 4, 0)  # 4 is approximate max entropy

    # 5. Edge Quality Assessment
    def assess_edge_quality():
        # Compute edges using Canny edge detector
        edges_low = cv2.Canny(gray, 50, 150)  # Lower threshold
        edges_high = cv2.Canny(gray, 100, 200)  # Higher threshold

        # Calculate edge density
        edge_density_low = np.count_nonzero(edges_low) / edges_low.size
        edge_density_high = np.count_nonzero(edges_high) / edges_high.size

        # Calculate edge strength using Sobel operators
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_strength = np.mean(np.sqrt(sobelx**2 + sobely**2))

        # Calculate edge continuity using morphological operations
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges_low, kernel, iterations=1)
        edge_continuity = np.count_nonzero(edges_low) / (
            np.count_nonzero(dilated) + 1e-6
        )

        # Combine metrics into overall edge score
        edge_density_score = (
            edge_density_low + edge_density_high
        ) / 0.4  # Normalize to 0-1
        edge_strength_score = min(edge_strength / 30, 1)  # Normalize to 0-1
        edge_continuity_score = edge_continuity

        # Weighted combination of edge metrics
        print(edge_density_score)
        print(edge_strength_score)
        print(edge_continuity_score)
        edge_score = (
            0.4 * min(edge_density_score, 1)
            + 0.3 * edge_strength_score
            + 0.3 * edge_continuity_score
        )

        return edge_score

    # Calculate individual metrics
    brightness_score = assess_brightness()
    contrast_score = assess_contrast()
    blur_score = detect_blur()
    noise_score = estimate_noise()
    edge_score = assess_edge_quality()

    # Calculate weighted average for overall score
    weights = {
        "brightness": 0.2,
        "contrast": 0.1,
        "blur": 0.2,
        "noise": 0.15,
        "edge": 0.35,
    }

    overall_score = (
        brightness_score * weights["brightness"]
        + contrast_score * weights["contrast"]
        + blur_score * weights["blur"]
        + noise_score * weights["noise"]
        + edge_score * weights["edge"]
    )

    # Return all metrics
    return {
        "overall_score": round(overall_score * 100, 2),
        "brightness_score": round(brightness_score * 100, 2),
        "contrast_score": round(contrast_score * 100, 2),
        "blur_score": round(blur_score * 100, 2),
        "noise_score": round(noise_score * 100, 2),
        "edge_score": round(edge_score * 100, 2),
        "is_good_quality": overall_score > 0.7,
    }


# Example usage
if __name__ == "__main__":
    try:
        # Replace with your image path
        image_path = "path/to/your/image.jpg"
        results = assess_image_quality(image_path)

        print("\nImage Quality Assessment Results:")
        print(f"Overall Score: {results['overall_score']}%")
        print(f"Brightness Score: {results['brightness_score']}%")
        print(f"Contrast Score: {results['contrast_score']}%")
        print(f"Blur Score: {results['blur_score']}%")
        print(f"Noise Score: {results['noise_score']}%")
        print(f"Edge Score: {results['edge_score']}%")
        print(f"Good Quality: {'Yes' if results['is_good_quality'] else 'No'}")

    except Exception as e:
        print(f"Error: {str(e)}")
