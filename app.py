from flask import Flask, request, render_template, send_from_directory
import os
import torch
from PIL import Image
import io
import base64

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def get_image_with_detections(image_path, results):
    # Show the image with detections
    results_img = results.render()[0]  # results.render() returns a list of numpy arrays (images)

    # Convert the numpy array to a PIL Image
    img = Image.fromarray(results_img)

    # Save the image to a BytesIO object
    img_io = io.BytesIO()
    img.save(img_io, 'JPEG')
    img_io.seek(0)

    # Encode the image to base64
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    return img_base64

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            # Perform object detection
            results = model(filepath)

            # Get image with detections as base64
            img_base64 = get_image_with_detections(filepath, results)

            return render_template('upload.html', uploaded_image=filepath, result_image=img_base64)

        except Exception as e:
            return f"Error during object detection: {e}"

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
