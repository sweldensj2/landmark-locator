"""
This file contains helper functions for the pretrained deployment portion of Lab-JetsonNanoSetup-PretrainedDeployment.

E6692 Spring 2024

YOU DO NOT NEED TO MODIFY THESE FUNCTIONS TO COMPLETE THE ASSIGNMENT
"""
import torch
import os
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import IPython
from IPython.display import display
import requests
from bs4 import BeautifulSoup
import json

from .install_dependancies import install_dependancies

try:
    from bs4 import BeautifulSoup
except:
    install_dependancies()
    from bs4 import BeautifulSoup

DOWNLOADS_PATH = './downloads/'
DATA_PATH = './data/'

def download_images(query, num_images):
    """
    Download the first N google images into the "downloads" folder
    of the root directory.
    
    params:
        query (string) : Google image query
        num_images (int) : Number of images to retrieve
    """

    query = query.replace(' ', '+')

    # URL for Google Images
    url = f"https://www.google.com/search?q={query}&tbm=isch"

    # Make a request to the website
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all image tags
    img_tags = soup.find_all("img")

    # Create a directory to save images
    folder_name = query.replace('+', '_')
    save_path = os.path.join(DOWNLOADS_PATH, folder_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Download images
    count = 0
    for img in img_tags:
        # Stop when we have enough images
        if count >= num_images:
            break

        # Get image URL
        img_url = img['src']
        try:
            # Send a request to the image URL
            img_response = requests.get(img_url)

            # Save the image
            with open(f"{save_path}/{count}.jpg", 'wb') as f:
                f.write(img_response.content)
            count += 1

        except:
            # Skip if there's any issue with one image
            pass
    
def download_images2(query, num_images):
    """
    Download the first N google images into the "downloads" folder
    of the root directory.

    params:
        query (string) : Google image query
        num_images (int) : Number of images to retrieve
    """

    query = query.replace(' ', '+')
    base_url = "https://www.google.com/search"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    params = {
        "q": query,
        "tbm": "isch",
        "ijn": "0",  # Start index of the results
    }

    # Create a directory to save images
    folder_name = query.replace('+', '_')
    save_path = os.path.join(DOWNLOADS_PATH, folder_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    count = 0
    while count < num_images:
        try:
            # Send a request to the Google Images URL
            response = requests.get(base_url, params=params, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all image tags
            img_tags = soup.find_all("img")

            # Download images
            for img in img_tags:
                # Stop when we have enough images
                if count >= num_images:
                    break

                # Get image URL
                img_url = img['src']
                try:
                    # Send a request to the image URL
                    img_response = requests.get(img_url)

                    # Save the image
                    with open(f"{save_path}/{count}.jpg", 'wb') as f:
                        f.write(img_response.content)
                    count += 1

                except:
                    # Skip if there's any issue with one image
                    pass

            # Update start index for the next page of results
            params["ijn"] = str(int(params["ijn"]) + 1)

        except Exception as e:
            print("Error:", e)
            break
import cv2

def download_images_with_resize(query, num_images, target_size=(320, 320)):
    """
    Download the first N google images into the "downloads" folder
    of the root directory and resize them to the target size.

    params:
        query (string) : Google image query
        num_images (int) : Number of images to retrieve
        target_size (tuple) : Target size to resize images (width, height)
    """

    query = query.replace(' ', '+')
    base_url = "https://www.google.com/search"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    params = {
        "q": query,
        "tbm": "isch",
        "ijn": "0",  # Start index of the results
    }

    # Create a directory to save images
    folder_name = query.replace('+', '_')
    save_path = os.path.join(DOWNLOADS_PATH, folder_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    count = 0
    while count < num_images:
        try:
            # Send a request to the Google Images URL
            response = requests.get(base_url, params=params, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all image tags
            img_tags = soup.find_all("img")

            # Download images
            for img in img_tags:
                # Stop when we have enough images
                if count >= num_images:
                    break

                # Get image URL
                img_url = img['src']
                try:
                    # Send a request to the image URL
                    img_response = requests.get(img_url)

                    # Resize image to the target size
                    img_array = np.frombuffer(img_response.content, np.uint8)
                    img_cv2 = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    img_resized = cv2.resize(img_cv2, target_size)

                    # Save the resized image
                    cv2.imwrite(f"{save_path}/{count}.jpg", img_resized)
                    count += 1

                except Exception as e:
                    # Skip if there's any issue with one image
                    print("Error downloading image:", e)

            # Update start index for the next page of results
            params["ijn"] = str(int(params["ijn"]) + 1)

        except Exception as e:
            print("Error:", e)
            break

def download_images_full_size(query, num_images, download_folder):
    """
    Download full-size images from Google Images search results.

    Params:
        query (str): Google image query.
        num_images (int): Number of images to download.
        download_folder (str): Folder path to save the downloaded images.
    """

    query = '+'.join(query.split())
    url = f"https://www.google.com/search?q={query}&source=lnms&tbm=isch"

    # Set up request headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36'
    }

    # Send request to Google Images URL
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract image links
    actual_images = []
    for div in soup.find_all("div", {"class": "rg_meta"}):
        metadata = json.loads(div.text)
        link, image_type = metadata["ou"], metadata["ity"]
        actual_images.append((link, image_type))

    print(f"There are a total of {len(actual_images)} images.")

    # Create download folder if it doesn't exist
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # Download images
    for i, (img, img_type) in enumerate(actual_images[:num_images]):
        try:
            img_response = requests.get(img, headers=headers)
            img_content = img_response.content
            img_filename = f"{query}_{i+1}.{img_type}" if img_type else f"{query}_{i+1}.jpg"
            img_path = os.path.join(download_folder, img_filename)
            with open(img_path, 'wb') as f:
                f.write(img_content)
            print(f"Downloaded image {i+1}/{num_images}: {img_filename}")
        except Exception as e:
            print(f"Could not load: {img}")
            print(e)


def display_image(image):
    """
    Display an image in Jupyter Notebook.

    param:
        image: numpy array representing an image, PIL image, or a path to an image
    """
    if isinstance(image, str):
        image_array = cv2.imread(image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        image_array = Image.fromarray(image_array)

    elif isinstance(image, np.ndarray):
        image_array = image

    else:
        raise Exception('image type is not supported.')

    plt.figure(figsize=(12, 8))
    plt.imshow(image_array)
    plt.axis('off')
    plt.show()


def show_array(a, fmt='jpeg'):
    """
    Display array using ipython widget.
    """
    f = io.BytesIO()
    Image.fromarray(a).save(f, fmt)
    display(IPython.display.Image(data=f.getvalue()))


