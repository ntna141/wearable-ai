from flask import Flask, request, jsonify
import openai
import base64
import cv2

app = Flask(__name__)
import imghdr
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
MODEL = 'gpt-4o'

import numpy as np

def calculate_color_percentages(image_path, color_ranges):
    # Load the image
    image = cv2.imread(image_path)

    # Calculate total pixels
    total_pixels = image.shape[0] * image.shape[1]

    # Initialize a dictionary to store color percentages
    color_percentages = {}

    # Iterate over each color range
    for color_name, (lower, upper) in color_ranges.items():
        # Create a mask for the current color range
        mask = cv2.inRange(image, lower, upper)
        
        # Count the pixels within the color range
        color_pixels = cv2.countNonZero(mask)
        
        # Calculate the percentage of pixels for the current color
        color_percentage = (color_pixels / total_pixels) * 100
        
        # Store the percentage in the dictionary
        color_percentages[color_name] = color_percentage

    return color_percentages

color_ranges = {
    "Whitish": ((180, 180, 180), (255, 255, 255)),
    "Blackish": ((0, 0, 0), (70, 70, 70)),
    "Reddish": ((0, 0, 150), (100, 100, 255)),
    "Greenish": ((0, 150, 0), (100, 255, 100)),
    "Bluish": ((150, 0, 0), (255, 100, 100))
}

def is_image_suitable_for_ocr(image_path, sharpness_threshold=0, white_lower=(180, 180, 180), white_upper=(255, 255, 255), black_lower=(0, 0, 0), black_upper=(70, 70, 70)):
    image = cv2.imread(image_path)
    percentages = calculate_color_percentages(image_path, color_ranges)
    print(percentages)
    sharpness = cv2.Laplacian(image, cv2.CV_64F).var()
    print("Sharpness:", sharpness)
    if sharpness < sharpness_threshold:
        return False

    whitish_mask = cv2.inRange(image, white_lower, white_upper)
    whitish_pixels = cv2.countNonZero(whitish_mask)
    total_pixels = image.shape[0] * image.shape[1]
    whitish_percentage = whitish_pixels / total_pixels

    blackish_mask = cv2.inRange(image, black_lower, black_upper)
    blackish_pixels = cv2.countNonZero(blackish_mask)
    blackish_percentage = blackish_pixels / total_pixels
    print(whitish_percentage, blackish_percentage)
    if whitish_percentage > 0.7 and blackish_percentage > 0.01:
        return True
    else:
        return False

@app.route('/answer', methods=['POST'])
def answer():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400

    # Save the image to a temporary file
    image_path = 'temp_image.' + imghdr.what(image_file)
    image_file.save(image_path)

    # Check if the image is suitable for OCR
    if not is_image_suitable_for_ocr(image_path):
        return jsonify({'error': 'Image is not suitable for OCR'}), 400

    # Convert the image to a base64-encoded string
    with open(image_path, 'rb') as f:
            image_data = f.read()
    base64_image = base64.b64encode(image_data).decode('utf-8')
    data_url = f'data:image/jpeg;base64,{base64_image}'

    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a school teacher and understand that a 'question' has a lot of context and not just the 'question' itself, the context might be anywhere on the page as long as it makes sense. Given an image of an exam, your task is to answer the exam question which includes all of its parts and contexts and not just the question itself, and its set of answers. You will transcribe the question and context and answer the question. The question is the one that shows up fully and is in the middle of the image, but its context might be elsewhere and if there are multiple questions and contexts, you need to answer the question within the context that makes sense and of the one that is most centrally located and fully visible.  If the image does not contain a clear question or does not have the context or does not fit these criteria, respond with 'no' and nothing else."},
            {"role": "user", "content": [
                {"type": "text", "text": "You will answer this question sent in the image based on the context, and images if there are any, given alongside the question and give me just the letter of the correct multiple choice answer and absolutely nothing else. "},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]}
        ],
        temperature=0.0,
    )

    answer = response.choices[0].message.content.strip()
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)