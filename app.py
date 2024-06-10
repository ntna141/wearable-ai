from flask import Flask, request, jsonify
import openai
import numpy as np
import base64
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
app = Flask(__name__)

MODEL = 'gpt-4o'

import cv2
import numpy as np

def is_text_on_paper(image_data):
    nparr = np.frombuffer(image_data, np.uint8)
    
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Create a copy of the original image for drawing
    image_with_boxes = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    min_aspect_ratio = 0.2
    max_aspect_ratio = 8.0
    min_area = 20
    max_area = 1000  # Adjust this value based on the maximum expected size of individual text components
    text_component_count = 0
    total_component_count = 0

    for i in range(1, num_labels):
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]

        total_component_count += 1

        if min_area <= area <= max_area:
            aspect_ratio = width / float(height)
            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                cv2.rectangle(image_with_boxes, (x, y), (x + width, y + height), (0, 255, 0), 1)
                text_component_count += 1
        # else:
            cv2.rectangle(image_with_boxes, (x, y), (x + width, y + height), (255, 0, 0), 1)

    text_component_percentage = (text_component_count / float(total_component_count)) * 100

    min_text_percentage = 50 
    contains_text = text_component_percentage >= min_text_percentage

    output_filename = "text_analysis_result.jpg"
    cv2.imwrite(output_filename, image_with_boxes)

    return contains_text

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400

    # Read the image file directly
    image_data = image_file.read()

    # Check if the image is suitable for OCR
    if not is_text_on_paper(image_data):
        return jsonify({'error': 'Image is not suitable for OCR'}), 400
    else:
        # print('hhe')
        # return "hehe"
        base64_image = base64.b64encode(image_data).decode('utf-8')
        data_url = f'data:image/jpeg;base64,{base64_image}'

        response = openai.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a school transcription assistant and understand that a 'question' has a lot of context and not just the 'question' itself, the context might be anywhere on the page as long as it makes sense. Given an image of an exam, your task is to transcribe the exam question which includes all of its parts and contexts and not just the question itself, and its set of answers. The question is the one that shows up fully and is in the middle of the image, but its context might be elsewhere and if there are multiple questions and contexts, you need to transcribe the question AND context of the one that is most centrally located and fully visible.  If the image does not contain a clear question or does not fit these criteria, respond with 'no' and nothing else. Never send me back a question without clear context that allows students to answer. If there is a table, a graph or an image component to the question, return 'graph' and nothing else.  You only do this if the question refers to an underlined portion, indicate where the underline is in your transcript using #, which means the 'text-underlined' is underlined in this case #text-underlined#, remember that there can only be 1 underlined portion."},
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
            print("graph")
            response_image = openai.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant who will answer this question based on the image and the context given in the image of the exam question sent to you."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "You will give me the answer to the main question in this image, as in the question that shows up most fully and in the center of the picture, and the answer will just be the correct multiple choice letter and absolutely nothing else"},
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