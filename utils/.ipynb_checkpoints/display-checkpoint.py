"""
This file contains functions for displaying inferences of the FaceNet MTCNN model.

E6692 Spring 2024
"""

# import modules
import os
import cv2
import numpy as np
from PIL import Image
import IPython

from .pretrained_deployment import display_image, show_array

DOWNLOADS_PATH = './downloads/'
DATA_PATH = './data/'

def display_images(query, num_images=3):
    """
    Displays the images in the directory './<downloads_path>/<query>'.

    params:
        query (string): image download query
        num_images (int): max numer of images to display

    DO NOT MODIFY THIS FUNCTION
    """
    folder_name = query.replace(' ', '_')
    image_names = os.listdir(os.path.join(DOWNLOADS_PATH, query)) # list image names
    image_count = 1 # initialize image count
    for image_name in image_names: # iterate through image names
        if image_count > num_images: # if image count is larger than num_images, break
            break
        if image_name != '.ipynb_checkpoints': # exclude .ipynb_checkpoints
            image = os.path.join(DOWNLOADS_PATH, query, image_name) # define image path
            display_image(image) # display image with display_image()
            image_count += 1 # increment image count


def display_faces(query, face_detector, num_images=3):
    """
    Display only the faces of the queried images.

    params:
        query (string): image download query
        face_detector (facenet_pytorch.models.mtcnn.MTCNN): face detection model
        num_images (int): max number of images to display

    HINT: Your implementation can be similar to display_images() and should use display_image().
    """

    #####################################################################################
    # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
    #####################################################################################

    folder_name = query.replace(' ', '_')
    image_names = os.listdir(os.path.join(DOWNLOADS_PATH, query)) # list image names
    image_count = 1 # initialize image count
    for image_name in image_names: # iterate through image names
        if image_count > num_images: # if image count is larger than num_images, break
            break
        if image_name != '.ipynb_checkpoints': # exclude .ipynb_checkpoints
            image = os.path.join(DOWNLOADS_PATH, query, image_name) # define image path
            
            
            #process the image here
#             print("image", image, type(image))

            img = Image.open(image) #actual image
#             print("img", type(img))
            
            
            new_image_name = image_name.replace(".jpg", "_face.jpg")
            edit_path_save = os.path.join(DOWNLOADS_PATH, query, new_image_name) #rewritting the old pictures
#             print("Edited Path", edit_path_save)
            face_image = face_detector(img, save_path=edit_path_save) #actually has face image
            
            display_image(edit_path_save) # display image with display_image()
            image_count += 1 # increment image count
   

    #####################################################################################
    # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
    #####################################################################################


def draw_boxes_and_landmarks(frame, boxes, landmarks):
    """
    This function draws bounding boxes and landmarks on a frame. It uses cv2.recangle() to
    draw the bounding boxes and cv2.circle to draw the landmarks.

    See OpenCV docs for more information on cv2.rectangle() and cv2.circle().

    https://www.geeksforgeeks.org/python-opencv-cv2-rectangle-method/
    https://www.geeksforgeeks.org/python-opencv-cv2-circle-method/

    params:
        frame (PIL.Image or np.array): the input frame
        boxes (list): 2D list of bounding box coordinates with shape (num_boxes, 4)
        landmark (list): 3D list of landmark points with shape (num_landmark_groups, 5, 2)

    returns:
        frame (np.array): the frame with bounding boxes and landmarks drawn.
    """
    #####################################################################################
    # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
    #####################################################################################

    #lets try to draw the boxes
    import PIL
    for box in boxes:
#         print("Box", box, type(box))
        start_point = np.array([box[0], box[1]])
#         end_point = np.array([start_point[0]+box[2], start_point[1]+box[3]]) #add width and heigh to start points
        #alternate idea
        end_point = np.array([box[2], box[3]])
        
        start_point = start_point.astype(int)
        end_point = end_point.astype(int)
#         print("start_point", start_point, "end_point", end_point, type(start_point))
        color = (255, 0, 0)
#         print("Color", type(color))
        thickness = 2
        frame_np = cv2.rectangle(np.array(frame), tuple (start_point), tuple (end_point), color = (255, 0,0), thickness = 2)
        
        frame = PIL.Image.fromarray(frame_np)
        
    for landmark in landmarks: #for every landmark that was passed in 
        for point in landmark:
            x1 = point[0]
            x2 = point[1]
            points = np.array([x1, x2])
            points = points.astype(int)
            color = (0, 255, 0)
            thickness = 1
            frame_np = cv2.circle(np.array(frame), tuple (points), 5, color, thickness)
    
            frame = PIL.Image.fromarray(frame_np)
        
    

    #####################################################################################
    # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
    #####################################################################################

    return frame


def display_detection_and_keypoints(query, face_detector, num_images=3):
    """
    This function displays the bounding boxes and keypoints (landmarks) for the images
    at the query directory. It uses draw_boxes_and_landmarks() to draw the bounding boxes
    and landmarks, then displays the frame.

    params:
        query (string): the query used to download google images.
        face_detector (facenet_pytorch.models.mtcnn.MTCNN): face detection model
        num_images (int): max number of images to display
    """
    #####################################################################################
    # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
    #####################################################################################

    
    import cv2
    import PIL
    folder_name = query.replace(' ', '_')
    image_names = os.listdir(os.path.join(DOWNLOADS_PATH, query)) # list image names
    image_count = 1 # initialize image count
    for image_name in image_names: # iterate through image names
        if image_count > num_images: # if image count is larger than num_images, break
            break
        if image_name != '.ipynb_checkpoints': # exclude .ipynb_checkpoints
            image = os.path.join(DOWNLOADS_PATH, query, image_name) # define image path
            
            
            #process the image here
            img = Image.open(image) #actual image
            boxes, _, landmarks = face_detector.detect(img, landmarks = True) 
#             print("Boxes", type(boxes), print(boxes))
#             print("Landmarks", type(landmarks), landmarks)
            
            img_boxed = draw_boxes_and_landmarks(img, boxes, landmarks)
            
            #save the image
            image_name_drawn = image_name.replace(".jpg", "_drawn.jpg")
            img_boxed.save(image_name_drawn)
            
            #display the image
#             edit_path_save = os.path.join(DOWNLOADS_PATH, query, image_name_drawn)
#             display_image(edit_path_save) # display image with display_image()
            
            display_image(image_name_drawn)
            image_count += 1 # increment image count
    
    
    

    #####################################################################################
    # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
    #####################################################################################


def video_inference(video_path, face_detector, max_frames=30):
    """
    This function uses the face detection model to generate a "detected version" of the
    specified video.

    params:
        video_path (string): path to the video to do inference on
        face_detector (facenet_pytorch.models.mtcnn.MTCNN): face detection model
        max_frames (int): the maximum frames to do inference with. Default is 30 frames.

    returns:
        detected_video_path (string): the path to the detected video

    DO NOT MODIFY CODE OUTSIDE OF YOUR IMPLEMENTATION AREA
    """
    video_name = video_path.split('/')[-1].split('.')[0] # get name of video
    detected_video_name = video_name + '-detected.mp4' # append detected name
    detected_video_path = os.path.join(DATA_PATH, detected_video_name) # define detected video path

    v_cap = cv2.VideoCapture(video_path) # initialize the video capture
    fourcc = cv2.VideoWriter_fourcc(*'VP90') # define encoding type
    fps = 30.0 # define frame rate
    video_dims = (960, 540) # define output dimensions
    out = cv2.VideoWriter(detected_video_path, fourcc, fps, video_dims) # initialize video writer

    frame_count = 0 # initialize frame count

    while True:
        frame_count += 1 # increment frame count

        success, frame = v_cap.read() # read frame from video

        if frame_count % 10 == 0:
            print("Frames detected: ", frame_count)

        if not success or frame_count >= max_frames: # break if end of video or max frames is reached
            break

        frame = Image.fromarray(frame) # read frame as PIL Image

        #####################################################################################
        # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
        #####################################################################################
        boxes, _, landmarks = face_detector.detect(frame, landmarks = True)
        frame = draw_boxes_and_landmarks(frame, boxes, landmarks)
        

        #####################################################################################
        # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
        #####################################################################################

        out.write(np.array(frame)) # write detected frame to output video

    v_cap.release() # release video reader and writer
    out.release()
    cv2.destroyAllWindows()

    return detected_video_path


def webcam_inference(face_detector):
    """
    This function implements the webcam display and performs inference using a face detection
    model.

    param:
        face_detector (facenet_pytorch.models.mtcnn.MTCNN): face detection model

    DO NOT MODIFY CODE OUTSIDE OF YOUR IMPLEMENTATION AREA
    """
    cam = cv2.VideoCapture(0) # define camera stream

    try: # start video feed
        print("Video feed started.")

        while True:
            success, frame = cam.read() # read frame from video stream

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert raw frame from BGR to RGB

            #####################################################################################
            # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
            #####################################################################################

            boxes, _, landmarks = face_detector.detect(frame, landmarks = True)
            frame = draw_boxes_and_landmarks(frame, boxes, landmarks)

            #####################################################################################
            # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
            #####################################################################################

            show_array(frame) # display the frame in JupyterLab

            IPython.display.clear_output(wait=True) # clear the previous frame

    except KeyboardInterrupt: # if interrupted
        print("Video feed stopped.")
        cam.release() # release the camera feed

        
