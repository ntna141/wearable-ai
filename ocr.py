import cv2
import numpy as np
from PIL import Image

def calculate_baseline_metrics(baseline_image_path):
    baseline_image = cv2.imread(baseline_image_path)
    
    baseline_gray = cv2.cvtColor(baseline_image, cv2.COLOR_BGR2GRAY)

    
    sharpness = cv2.Laplacian(baseline_gray, cv2.CV_64F).var()
    print(sharpness)
    
    return sharpness


def is_image_suitable_for_ocr(image_path, baseline_sharpness, tolerance=0.8):
    image = cv2.imread(image_path)

    sharpness = cv2.Laplacian(image, cv2.CV_64F).var()
    print("sharpness", sharpness)
    if sharpness < baseline_sharpness * tolerance:
        return False
    return True

# Example usage
baseline_image_path = 'next.png'
baseline_sharpness = calculate_baseline_metrics(baseline_image_path)

image_path = 'blur2.jpg'
if is_image_suitable_for_ocr(image_path, baseline_sharpness):
    print("The image is suitable for OCR.")
else:
    print("The image is not suitable for OCR.")