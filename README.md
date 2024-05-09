[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/UHGdSN-p)
# E6692 Spring 2024: Final Project
This project presents a software demo called landmark_locator.py aimed at assisting tourists in identifying famous New York City landmarks in real-time without the need for an internet connection. The program utilizes object detection to identify ten major landmarks and runs locally on edge devices. The project incorporates three different models for inference: RT-DETR-l, YOLOv8l, and YOLOv8n, each offering unique trade-offs between accuracy and throughput. The paper outlines the methodology used in training these models, highlighting challenges such as dataset collection, model selection, and evaluation metrics. Additionally, the results of the trained models are discussed, comparing them with metrics from reference papers and evaluating their performance in live testing scenarios across various locations in Manhattan. The study concludes with insights into the discrepancy between observed throughput metrics and those reported in the original paper, suggesting areas for further investigation and refinement of the models.
![val_batch2_pred](https://github.com/eecse6692/e6692-2024spring-finalproject-jwss-jws2215/assets/144495665/2116268f-76fc-4924-af5e-a13b89205d18)

## landmark_locator.py
- **Description**: Runs the object detection models in inference mode. It supports three different modes: "detr" for RT-DETR model, "nano" for YOLO8n model, and "yolo" for YOLO8l model. The detected buildings are highlighted with bounding boxes on the video feed, along with their confidence scores. Each building is labeled with its name and confidence score, and the bounding boxes are color-coded to represent different buildings. Additionally, the program calculates and prints the average model inference time and the average processing time per frame. To interrupt the execution, the user can press the "q" key or click on the red box.

## training_models.ipynb
- **Description**: Trainings the RT-DETR and YOLOv8 models. It incorporates transfer learning by loading in pretrained weights from Ultralytics that were optimized on the 80 classes of COCO. The results of the training runs are stored in the folder runs/detect.

## dataset.yaml
- **Description**: Configuration file that contains the classes and paths for trianing the Ultralytics models on the nyc_landmarks dataset. 

## Pretrained models
- **Description**: 'rtdetr-l.pt', 'yolov8l.pt', and 'yolov8n.pt' are all the Ultralytics pretrained weights that were used in the training_models workbook.

## landmark_videos
- **Description**: Folder containing the screen captures recorded from testing the program landmark_locator.py at a variety of locations in Manhattan. The folder contains both vidoes, converted gifs, and notes taken from each location.

## runs
- **Description**: Folder containing the trained models and their respective training metrics in a subfolder called detect. The folder predict contains visualizations of the convolutional layers from a prediction that was made.

## Legacy Code
- **Description**: Contains code from the old repository, mainly the data_training.ipynb workbook and the llm_experiment.ipynb workbook. The process for scrapping the internet for images and combining the images into the nyc_landmarks dataset is handled by the data_training workbook. The LLM experiment was a failed attempt at trying to implement a small language model that could run locally and provide insights on NYC landmarks.

## E6692.2024Spring.JWSS.report.jws2215.pdf
- **Description**: Final Report
  
## nyc_landmarks dataset
- **Description**: Link to created dataset
- **Link**: [nyc_landamrks](https://www.kaggle.com/datasets/jws2215/nyc-landmarks-object-detection)
