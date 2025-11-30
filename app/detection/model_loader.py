import os
import sys
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

class ModelLoader:
    def __init__(self,model_path):
        self.model_path = model_path
        self.model = None
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self.device = device
    

    def loader(self):
        if self.model is not None:
            return self.model
        
        model_path = self.model_path

        if os.path.exists(model_path):
            try:
                model = YOLO(model_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load local model: {e}")
        else:
            filename_only = "/" not in model_path and "\\" not in model_path

            if filename_only and model_path.endswith(".pt"):
                try:
                    model = YOLO(model_path)
                except Exception as e:
                    raise RuntimeError(f"Failed to download YOLO model {e}")
            else:
                raise FileNotFoundError(
                    f"Model file not found at :{model_path}\n"
                    f"YOLO cannot auto-download if you pass custom directory path."
                )

        model.to(self.device)
        self.model = model
        return self.model
        
    def detector(self, frame, conf = 0.5):
        """
        Run detection on a single image/frame using the loaded YOLO model.
        """
        if frame is None or not hasattr(frame, "shape"):
            raise ValueError("Invalid frame provided for detection")
        
        model = self.loader()

        try:
            results = model.predict(frame, conf = conf)
        except Exception as e:
            raise RuntimeError(f"Detection failed: {e}")
        
        return results
    

if __name__ == "__main__":
    model = ModelLoader("yolov8s.pt")
    model.loader()

