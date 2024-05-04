"""
This program utilizes a webcam to detect and identify various landmarks or buildings in real-time using pre-trained models. It supports three different modes: "detr" for RT-DETR model, "nano" for YOLO8n model, and "yolo" for YOLO8l model. The detected buildings are highlighted with bounding boxes on the video feed, along with their confidence scores. Each building is labeled with its name and confidence score, and the bounding boxes are color-coded to represent different buildings. The program also calculates and prints the average model inference time and the average processing time per frame. To interrupt the execution, press the "q" key.

Johan Sweldens
Columbia University
EECS6692 Spring 2024
"""

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
import sys
import webbrowser



from utils.pretrained_deployment import download_images, download_images2, download_images_with_resize, download_images_full_size
from utils.display import *



##### Initliaze Class Dictionaries #####
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

# Dictionary mapping building names to website URLs
class_websites = {
    0: "https://en.wikipedia.org/wiki/Empire_State_Building",
    1: "https://en.wikipedia.org/wiki/One_World_Trade_Center",
    2: "https://en.wikipedia.org/wiki/432_Park_Avenue",
    3: "https://en.wikipedia.org/wiki/Headquarters_of_the_United_Nations",
    4: "https://en.wikipedia.org/wiki/Flatiron_Building",
    5: "https://en.wikipedia.org/wiki/Brooklyn_Bridge",
    6: "https://en.wikipedia.org/wiki/Chrysler_Building",
    7: "https://en.wikipedia.org/wiki/MetLife_Building",
    8: "https://en.wikipedia.org/wiki/Statue_of_Liberty",
    9: "https://en.wikipedia.org/wiki/30_Hudson_Yards",
}




# Read in the mode selection
mode = sys.argv[1]

if(mode == "detr"):
    print("Loading RT-DETR Model")
    weights_path = './runs/detect/detr_e100/weights/best.pt'
    # Load trained weights
    model = RTDETR(weights_path)
elif(mode == "nano"):
    print("Loading Yolo8n")
    weights_path = './runs/detect/yolo8n_e100/weights/best.pt'
    # Load trained weights
    model = YOLO(weights_path)
elif(mode == "yolo"):
    print("Loading Yolo8l")
    weights_path = './runs/detect/yolo8l_e100/weights/best.pt'
    # Load trained weights
    model = YOLO(weights_path)




# CV2 or Predict variables
conf = 0.7 # gotta be 85% sure, its one of the buildings
visualize = False
max_num_buildings = 11 #literally impossible to see more than the 10 (+1) objects lmao
half_precision_inf = False
box_thickness = 6
log_time = True
total_process_time = 0
total_model_time = 0
count = 0

# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
# fontScale 
fontScale = 1.5
# Line thickness of 2 px 
text_thickness = 2



######### Intialize webcam and click operations #########

# Exit button rectangle coordinates (x1, y1, x2, y2)
exit_button_coords = (10, 10, 50, 50)

# Mouse click event callback function
def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Click Detected", x, y)
        
        # Check if click coordinates fall within the exit button rectangle, this totally doesn't work
        if exit_button_coords[0] <= x <= exit_button_coords[2] and exit_button_coords[1] <= y <= exit_button_coords[3]:
            print("Should exit now?")
            exit_clicked = True 
            
        # Click on a box and see a website for the item
        for box in prediction.boxes:
            xyxy = box.xyxy.squeeze()
            start_point = (int(xyxy[0]), int(xyxy[1]))
            end_point = (int(xyxy[2]), int(xyxy[3]))
            
            item_cls = box.cls.squeeze().cpu().numpy()
            if start_point[0] <= x <= end_point[0] and start_point[1] <= y <= end_point[1]: # if there was a click inside a box
                print("Valid Box Click", classes_big[int(item_cls)])
                website_url = class_websites[int(item_cls)]
                webbrowser.open(website_url)

                    
cam = cv2.VideoCapture(0)  # 0 is the pluggin index, 1 is the laptop camera
# Set up OpenCV window and mouse callback
cv2.namedWindow("Webcam")
cv2.setMouseCallback("Webcam", mouse_click)




####### Main Loop #######
# Initialize exit button flag
exit_clicked = False

try:
    while True:
        # Read frame from video stream
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            # break
            continue

        

        # Make a prediction
        model_time = time.time()
        prediction = model(frame, conf = conf, visualize = visualize, device='mps', max_det = max_num_buildings, half = half_precision_inf, verbose = False)[0]
        model_time = time.time() - model_time
        total_model_time +=  model_time
        
        # print("model_time", model_time)

        process_time = time.time()
        # Draw the boxes
        for box in prediction.boxes:
            xyxy = box.xyxy.squeeze()
            start_point = (int(xyxy[0]), int(xyxy[1]))
            end_point = (int(xyxy[2]), int(xyxy[3]))
            
            item_conf = box.conf.squeeze().cpu().numpy()
            item_cls = box.cls.squeeze().cpu().numpy()

            
            item_name = classes_big[int(item_cls)]
  
            # Draw a rectangle with blue line borders of thickness of 2 px 
            color = class_colors[int(item_cls)]
            frame = cv2.rectangle(frame, start_point, end_point, color, box_thickness) 

            # Add class name next to box
            text_to_add = item_name + ", " + str(item_conf)
            org = (start_point[0] - 75, start_point[1]-15)
            frame = cv2.putText(frame, text_to_add, org, font, fontScale, color, text_thickness, cv2.LINE_AA) 

        # Display exit button (red square) in top right corner
        frame = cv2.rectangle(frame, (exit_button_coords[0], exit_button_coords[1]), (exit_button_coords[2], exit_button_coords[3]), (0, 0, 255), -1)
        
        process_time = time.time() - process_time
        total_process_time += process_time
        # print("process_time", process_time)
        
        # Display the frame
        cv2.imshow("Webcam", frame)
        count += 1


        
        # Check for key press to interrupt the loop or if exit button is pressed
        key = cv2.waitKey(1) & 0xFF
        if exit_clicked or key == ord("q"):  # Press 'q' to exit
            total_process_time = total_process_time / count
            total_model_time = total_model_time / count
            print("Average Model Time:", total_model_time)
            print("Average Process Time:", total_process_time)
            break
finally:
    # Release the camera feed
    cam.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()
