##🚀 Real-Time Object Detection API (YOLOv8 + Flask)

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
    GET /ping
    Response
    "Server is running"
2. Form-Data:
    image: File (png/jpg/jpeg)
    Respons{
    "status": "success",
    "predicted_image": "/uploads/predictions/output.jpg"
    }
3. Predict on Video
    POST /predict/video
    Form-Data:
    video: File (mp4)
    Response:
    {
    "status": "success",
    "output_video": "/uploads/predictions/output.mp4"
    }
4. Fetch Predicted Files
    GET /uploads/predictions/<filename>

🧪 How to Test with Postman
    Image Prediction Test
    POST → http://127.0.0.1:5000/predict/image
    Body → form-data
        Key: image
        Type: File
        Upload a .jpg / .png / .jpeg

    Video Prediction Test
    POST → http://127.0.0.1:5000/predict/video
    Body → form-data
        Key: video
        Type: File
        Upload a .mp4

    If everything is correct, you will get a JSON with a URL.