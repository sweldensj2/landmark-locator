# import packages
import os
import cv2
import numpy as np
from IPython.display import Video, display, clear_output
import time
import torch
from ultralytics import RTDETR, YOLO
import torchvision.transforms as T
from PIL import Image
import random
import uuid
import io
import IPython
import time




from utils.pretrained_deployment import download_images, download_images2, download_images_with_resize, download_images_full_size
from utils.display import *



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

classes_big = {
    0: "Empire State Building",
    1: "Freedom Tower",
    2: "432 Park Avenue",
    3: "United Nations HQ",
    4: "Flatiron Building",
    5: "Brooklyn Bridge (Tower)",
    6: "Chrysler Building",
    7: "Metlife Building",
    8: "Statue Of Liberty",
    9: "30 Hudson Yards",
}

# Create a dictionary mapping class index to color
class_colors = {
    0: (255, 0, 0),     # Red - Represents strength and prominence of the Empire State Building.
    1: (0, 128, 255),   # Blue - Symbolizes the sky and freedom associated with the Freedom Tower.
    2: (255, 215, 0),   # Gold/Yellow - Reflects the luxurious and opulent nature of 432 Park Avenue.
    3: (0, 255, 0),     # Green - Represents unity and diplomacy, fitting for the United Nations HQ.
    4: (128, 128, 128), # Gray - Matches the iconic iron and limestone facade of the Flatiron Building.
    5: (255, 140, 0),   # Orange - Symbolizes the warmth and energy of the Brooklyn Bridge.
    6: (255, 0, 255),   # Magenta - Represents the Art Deco style and uniqueness of the Chrysler Building.
    7: (255, 105, 180), # Pink - Reflects the elegance and grandeur of the Metlife Building.
    8: (0, 128, 0),     # Dark Green - Symbolizes freedom and nature, fitting for the Statue of Liberty.
    9: (128, 0, 128)    # Purple - Represents modernity and innovation, fitting for 30 Hudson Yards.
}


# local
mode = "nano" #yolo8n
# mode = "detr" #rt-detr

if(mode == "detr"):
    print("Loading RT-DETR Model")
    weights_path = './runs/detect/detr/weights/best.pt'
    # Load trained weights
    model = RTDETR(weights_path)
elif(mode == "nano"):
    print("Loading Yolo8n")
    weights_path = './runs/detect/detr/weights/best.pt'
    # Load trained weights
    model = YOLO(weights_path)





# CV2 or Predict variables
conf = 0.7 # gotta be 85% sure, its one of the buildings
visualize = False
max_num_buildings = 11 #literally impossible to see more than the 10 (+1) objects lmao
half_precision_inf = False

box_thickness = 6

# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
# fontScale 
fontScale = 1.5
# Line thickness of 2 px 
text_thickness = 2



# intialize webcam
cam = cv2.VideoCapture(0)  # 0 is the webcam index



try:
    while True:
        # Read frame from video stream
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            # break
            continue

        

        # Make a prediction
        start_time = time.time()
        prediction = model(frame, conf = conf, visualize = visualize, device='mps', max_det = max_num_buildings, half = half_precision_inf, verbose = False)[0]
        print("model_time", str(time.time() - start_time))

        process_start = time.time()
        print("prediction.boxes", prediction.boxes)
        # Draw the boxes
        for box in prediction.boxes:
            print("box", box.shape, type(box))
            # Blue color in BGR 
            xyxy = box.xyxy.squeeze()
            start_point = (int(xyxy[0]), int(xyxy[1]))
            end_point = (int(xyxy[2]), int(xyxy[3]))
            
            item_conf = box.conf.squeeze().cpu().numpy()
            item_cls = box.cls.squeeze().cpu().numpy()
            print("item_cls", item_cls)
            item_name = classes_big[int(item_cls)]
  
            # Draw a rectangle with blue line borders of thickness of 2 px 
            color = class_colors[int(item_cls)]
            frame = cv2.rectangle(frame, start_point, end_point, color, box_thickness) 

            # Add class name next to box
            text_to_add = item_name + ", " + str(item_conf)
            org = start_point
            frame = cv2.putText(frame, text_to_add, org, font, fontScale, color, text_thickness, cv2.LINE_AA) 


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
