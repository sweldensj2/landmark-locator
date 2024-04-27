# import packages
import os
import cv2
import numpy as np
from IPython.display import Video, display, clear_output
import time
import torch
from ultralytics import RTDETR
import torchvision.transforms as T
from PIL import Image
import random
import uuid
import io
import IPython
import time




from utils.pretrained_deployment import download_images, download_images2, download_images_with_resize, download_images_full_size
from utils.display import *

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##### Initliaze Model
classes = {
    0: "EmpireState",
    1: "WTC",
    2: "432ParkAve",
    3: "UNBuilding",
    4: "Flatiron",
    5: "BrooklynBridge",
    6: "ChryslerBuilding",
    7: "MetlifeBuilding",
    8: "StatueOfLiberty",
    9: "30HudsonYards",
}

# local
weights_path = '/Users/johansweldens/Documents/EECS6692.DLoE/final_project/e6692-2024spring-finalproject-jwss-jws2215/runs/detect/detr/weights/best.pt'

# Load trained weights
model = RTDETR(weights_path)

# CV2 or Predict variables
conf = 0.85 # gotta be 85% sure, its one of the buildings
visualize = False
max_num_buildings = 11 #literally impossible to see more than the 10 (+1) objects lmao
half_precision_inf = False

color = (255, 0, 0) 
thickness = 2



# intialize webcam
cam = cv2.VideoCapture(0)  # 0 is the webcam index



try:
    while True:
        # Read frame from video stream
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        

        # Make a prediction
        start_time = time.time()
        prediction = model(frame, conf = conf, visualize = visualize, device='mps', max_det = max_num_buildings, half = half_precision_inf)[0]
        print("model_time", str(time.time() - start_time))

        process_start = time.time()
        # Draw the boxes
        for box in prediction.boxes:
            print("box", box.shape, type(box))
            # Blue color in BGR 
            xyxy = box.xyxy.squeeze()
            start_point = (int(xyxy[0]), int(xyxy[1]))
            end_point = (int(xyxy[2]), int(xyxy[3]))
            
            item_conf = box.conf.squeeze().cpu().numpy()
            item_cls = box.cls.squeeze().cpu().numpy()
            item_name = classes[int(item_cls)]
            print("Item", item_name, "Item_conf", item_conf)
  
            # Draw a rectangle with blue line borders of thickness of 2 px 
            frame = cv2.rectangle(frame, start_point, end_point, color, thickness) 


        print("process_time", str(time.time() - process_start))
        # Display the frame
        cv2.imshow("Webcam", frame)


        
        # Check for key press to interrupt the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Press 'q' to exit
            break
finally:
    # Release the camera feed
    cam.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()
