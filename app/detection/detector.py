import os
import sys
import cv2

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from model_loader import ModelLoader

UPLOAD_FOLDER = os.path.join(ROOT_DIR, "uploads")
IMAGE_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, "images")
VIDEO_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, "videos")
PREDICTION_FOLDER = os.path.join(UPLOAD_FOLDER, "predictions")

os.makedirs(IMAGE_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTION_FOLDER, exist_ok=True)

class Detector:
    def __init__(self,model_path):
        self.model_loader = ModelLoader(model_path)

    def detect_image(self,image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Invalid image path or unable to upload image")
        
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.model_loader.detector(img)
        result = results[0]

        annoted = result.plot()
        annoted_bgr = cv2.cvtColor(annoted, cv2.COLOR_RGB2BGR)

        base = os.path.basename(image_path)
        name, ext = os.path.splitext(base)
        filename = f"{name}_pred{ext}"


        output_path = os.path.join(PREDICTION_FOLDER, filename)
        cv2.imwrite(output_path, annoted_bgr)

        return filename

    def detect_video(self,video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Invalid video path or unable to open video")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # generate output filename
        base = os.path.basename(video_path)
        name , ext = os.path.splitext(base)
        filename = f"{name}_pred{ext}"

        #Output video path
        output_path = os.path.join(PREDICTION_FOLDER, filename)

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            #convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #YOLO prediction
            results = self.model_loader.detector(rgb_frame)
            result = results[0]

            #Annoted frame
            plotted = result.plot()

            #Convert to BGR to saving
            plotted_bgr = cv2.cvtColor(plotted, cv2.COLOR_RGB2BGR)

            out.write(plotted_bgr)

        cap.release()
        out.release()

        return filename

                
