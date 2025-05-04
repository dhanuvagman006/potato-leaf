from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import cv2
from inference import get_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Roboflow model (with your API key)
model = get_model(model_id="potato-leaves-disease-detection-bsjop/15", api_key="DmqJHhNAibIGwIrSgCwL")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Run inference on the file
        results = model.infer(filepath)
        print(results)

        # Get predictions from the results
        predictions = results[0].predictions  # Extracting the predictions from the first inference response

        detections = []
        image = cv2.imread(filepath)

        for prediction in predictions:
            x = prediction.x
            y = prediction.y
            width = prediction.width
            height = prediction.height
            conf = prediction.confidence
            cls = prediction.class_name  # Use class_name instead of class

            # Convert center-based coordinates to bounding box (x1, y1, x2, y2)
            x1 = x - width / 2
            y1 = y - height / 2
            x2 = x + width / 2
            y2 = y + height / 2

            detections.append({
                'class': cls,
                'confidence': float(conf),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })

            # Draw the bounding box on the image
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f'{cls}: {conf:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the processed image
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{filename}')
        cv2.imwrite(output_path, image)

        return jsonify({
            'success': True,
            'detections': detections,
            'original_image': f'/static/uploads/{filename}',
            'processed_image': f'/static/uploads/processed_{filename}'
        })

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
