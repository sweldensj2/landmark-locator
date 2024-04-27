"""
Install dependancies function for Lab-JetsonNanoSetup-PretrainedDeployment.

E6692 Spring 2024

YOU DO NOT NEED TO MODIFY THIS FUNCTION TO COMPLETE THE ASSIGNMENT
"""
import os

DEPENDANCIES = ['facenet_pytorch', 
                'tqdm',
                'BeautifulSoup4']

def install_dependancies():
    """
    Install the dependancies listed above with pip.
    """
    for dependancy in DEPENDANCIES:
        os.system('pip3 install {}'.format(dependancy))
        print("{} installed.".format(dependancy))

    