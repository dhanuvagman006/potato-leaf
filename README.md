# Potato Leaf Detection Web Application

This is a web application that uses YOLO (You Only Look Once) for detecting potato leaf diseases. The application allows users to upload images of potato leaves and get real-time detection results.

## Features

- Modern and responsive web interface
- Drag and drop image upload
- Real-time object detection using YOLO
- Display of original and processed images
- Detailed detection results with confidence scores

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd potato-leaf-detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Upload an image of a potato leaf by either:
   - Clicking on the upload area and selecting a file
   - Dragging and dropping an image onto the upload area

4. Wait for the detection results to appear

## Project Structure

- `app.py`: Main Flask application file
- `templates/index.html`: Web interface template
- `static/uploads/`: Directory for storing uploaded and processed images
- `requirements.txt`: Python dependencies

## Notes

- The application uses the default YOLOv8 model. For better results, you can train a custom model specifically for potato leaf diseases and replace the model file.
- Maximum file size is limited to 16MB
- Supported image formats: PNG, JPG, JPEG

## License

This project is licensed under the MIT License - see the LICENSE file for details. 