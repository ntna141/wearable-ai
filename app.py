from flask import Flask, request, jsonify
import openai
import imghdr
import base64
import cv2
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
app = Flask(__name__)

MODEL = 'gpt-4o'

def is_image_suitable_for_ocr(image_path, sharpness_threshold=0):
    image = cv2.imread(image_path)
    sharpness = cv2.Laplacian(image, cv2.CV_64F).var()
    print("sharpness", sharpness)
    if sharpness < sharpness_threshold:
        return False
    return True

@app.route('/transcribe', methods=['POST'])
def transcribe():
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
    else:
        with open(image_path, 'rb') as f:
            image_data = f.read()

        base64_image = base64.b64encode(image_data).decode('utf-8')
        data_url = f'data:image/jpeg;base64,{base64_image}'

        response = openai.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a school transcription assistant and understand that a 'question' has a lot of context and not just the 'question' itself, the context might be anywhere on the page as long as it makes sense. Given an image of an exam, your task is to transcribe the exam question which includes all of its parts and contexts and not just the question itself, and its set of answers. The question is the one that shows up fully and is in the middle of the image, but its context might be elsewhwere and if there are multiple questions and contexts, you need to transcribe the question AND context of the one that is most centrally located and fully visible.  If the image does not contain a clear question or does not fit these criteria, respond with 'no' and nothing else. Never send me back a question without clear context that allows students to answer. If there is a table, a graph or an image component to the question, return 'graph' and nothing else.  You only do this if the question refers to an underlined portion, indicate where the underline is in your transcript using #, which means the 'text-underlined' is underlined in this case #text-underlined#, remember that there can only be 1 underlined portion."},
                {"role": "user", "content": [
                    {"type": "text", "text": "You will give the transcript for this exam question, with full context, and nothing else. Never send me back a question without clear context that allows students to answer it. If the question refers to an underlined portion look carefully and indicate that in the response, and remember there should be only 1 underlined portion and only do this if it is explicitly stated that there is an underlined portion. If there is a table, a graph or an image component to the question, return 'graph' and nothing else."},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]}
            ],
            temperature=0.0,
        )

        transcript = response.choices[0].message.content.strip()
        if transcript.lower() == "no" or transcript.lower() == "no.":
            return jsonify({'error': 'Image does not contain a single question with answers'}), 400
        if transcript.lower() == "graph" or transcript.lower() == "graph.":
            response_image = openai.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant who will answer this question based on the image and the context given in the image of the exam question sent to you."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "You will give me the answer to the main question in this image, as in the question that shows up most fully and in the center of the picture, and the answer willl just be the correct multiple choice letter and absolutely nothing else"},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]}
                ],
                temperature=0.0,
            )
            answer = response_image.choices[0].message.content.strip()
            return jsonify({'answer': answer})

    print(transcript)
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant, understand that this notation #text-underlined# means the 'text-underlined' is underlined and is a part of the question. Answer the given question with just the letter from the answers set and nothing else. Just 1 letter"},
            {"role": "user", "content": transcript}
        ],
        temperature=0.0,
    )

    answer = response.choices[0].message.content.strip()
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)