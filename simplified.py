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

def is_image_suitable_for_ocr(image_path, sharpness_threshold=0):
    image = cv2.imread(image_path)
    sharpness = cv2.Laplacian(image, cv2.CV_64F).var()
    print("sharpness", sharpness)
    if sharpness < sharpness_threshold:
        return False
    return True

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