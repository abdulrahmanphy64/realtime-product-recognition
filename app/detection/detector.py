import os
import cv2
from model_loader import ModelLoader


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

        cv2.imshow("Prediction",annoted_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_video(self,video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Invalid video path or unable to open video")
        
        cv2.namedWindow('Prediction',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Prediction',800,800)

        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model_loader.detector(rgb_frame)
            for result in results:
                res = result.plot()
                cv2.imshow("Prediction",res)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return 


    def detect_webcam(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise ValueError("Could not access the webcam")
        
        cv2.namedWindow("Webcam Prediction",cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Webcam Prediction", 800,800)

        while cap.isOpened():
            res, frame = cap.read()
            if not res:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model_loader.detector(rgb_frame)

            for result in results:
                plotted = result.plot()
                cv2.imshow("Webcam Prediction", plotted)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return 
                
