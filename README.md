🚀 Real-Time Object Detection API (YOLOv8 + Flask)

This project is a Flask-based REST API for performing object detection on images and videos using YOLOv8.
It supports uploading files, running inference, and returning the processed files with bounding-box predictions.

⚙️ Features
    Upload image → get annotated image with detections
    Upload video → get annotated video with detections
    Built-in YOLO model loader
    Input validation (file format, missing file, invalid paths)
    Organized folder structure
    Logging added for:
        Requests
        Errors
        Model predictions
        File save locations

📌 API Endpoints
1. Server Check
    1. GET /ping
    2. Response
    3. "Server is running"
2. Form-Data:
    1. image: File (png/jpg/jpeg)
    2. Respons{
    "status": "success",
    "predicted_image": "/uploads/predictions/output.jpg"
    }
3. Predict on Video
    1. POST /predict/video
    2. Form-Data:
        video: File (mp4)
    3. Response:
        {
        "status": "success",
        "output_video": "/uploads/predictions/output.mp4"
        }
4. Fetch Predicted Files
    1. GET /uploads/predictions/<filename>

🧪 How to Test with Postman
    1. Image Prediction Test
    2. POST → http://127.0.0.1:5000/predict/image
    3. Body → form-data
        1. Key: image
        2. Type: File
        3. Upload a .jpg / .png / .jpeg

    Video Prediction Test
    POST → http://127.0.0.1:5000/predict/video
    Body → form-data
        Key: video
        Type: File
        Upload a .mp4

    If everything is correct, you will get a JSON with a URL.
