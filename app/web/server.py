import os
import sys 
import uuid
import logging
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
from pathlib import Path
from flask import send_from_directory


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from detection.detector import Detector

UPLOAD_FOLDER = os.path.join(ROOT_DIR, "uploads")
IMAGE_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, "images")
VIDEO_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, "videos")
PREDICTION_FOLDER = os.path.join(UPLOAD_FOLDER, "predictions")

os.makedirs(IMAGE_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTION_FOLDER, exist_ok=True)

# Create logs folder if not exists
LOG_FOLDER = os.path.join(ROOT_DIR, "logs")
os.makedirs(LOG_FOLDER, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_FOLDER, "app.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)

model_loader = Detector("yolov8s.pt")


ALLOWED_EXTENSION_IMAGE = ['png','jpg','jpeg']
ALLOWED_EXTENSION_VIDEO = ['mp4']

def allowed_file_image(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSION_IMAGE

def allowed_file_video(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSION_VIDEO

@app.route('/ping')
def ping():
    return "Server is running"

@app.route('/predict/image',methods = ['POST'])
def predict_image():
    try:
        logging.info("Recieved image prediction request")

        if 'image' not in request.files:
            logging.error("Image file missing from request")
            raise FileNotFoundError("Image not found")

        image = request.files['image']

        if image.filename == "":
            logging.error("Empty file received")
            raise ValueError("No image selected")

        if not allowed_file_image(image.filename):
            logging.error(f"Invalid image format: {image.filename}")
            raise ValueError("Invalid image format")

        save_path = os.path.join(IMAGE_UPLOAD_FOLDER,image.filename)
        image.save(save_path)

        logging.info(f"Image save at: {save_path}")

        #Prediction
        filename = model_loader.detect_image(save_path)

        return jsonify({
            "status" : "success",
            "predicted_image" : f"/uploads/predictions/{filename}"
        })
    
    except Exception as e:
        logging.exception("Error in image prediction")
        return jsonify({"Error" : str(e)}),400


@app.route('/uploads/predictions/<filename>')
def serve_predictions(filename):
    return send_from_directory(PREDICTION_FOLDER, filename)

@app.route('/predict/video',methods = ['POST'])
def predict_video():
    try:
        logging.info("Recieved image prediction request")

        if 'video' not in request.files:
            logging.error("Video file missing from request")
            raise FileNotFoundError("Video not found")

        video = request.files['video']

        if video.filename == "":
            logging.error("Empty video file recieved")
            raise ValueError("No video selected")

        if not allowed_file_video(video.filename):
            logging.error(f"Invalid video format: {video.filename} ")
            raise ValueError("Invalid video format")

        save_path = os.path.join(VIDEO_UPLOAD_FOLDER,video.filename)
        video.save(save_path)

        logging.info(f"Video saved at: {save_path}")

        video_path = str(Path(save_path).resolve())
        result = model_loader.detect_video(video_path)

        logging.info(f"video processing complete : {result}")

        return jsonify({
            "status" : "success",
            "output_video" : f"/uploads/predictions/{result}"
        })
    
    except Exception as e:
        logging.error("Error in video prediction")
        return jsonify({"error": str(e)}),400
    

if __name__ == "__main__":
    app.run()






    
