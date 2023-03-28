import os
import cv2
import numpy as np

def loadImages(dataPath):
    """
    load all Images in the folder and transfer a list of tuples. The first 
    element is the numpy array of shape (m, n) representing the image. 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    # raise NotImplementedError("To be implemented")
    '''
    use os.path.join to combine datapath of folder, then use os.listdir access to all img in the folder.
    use cv2.imread to read each img, and use cv2.IMREAD_GRAYSCALE to convert it into grayscale.
    then append img along with label into dataset[] 
    '''
    dataset = list(tuple())
    for file in os.listdir(os.path.join(dataPath,"face")):
        img = cv2.imread(os.path.join(dataPath,"face",file),cv2.IMREAD_GRAYSCALE)
        temp = ((img),1)
        dataset.append(temp)
    for file in os.listdir(os.path.join(dataPath,"non-face")):
        img = cv2.imread(os.path.join(dataPath,"non-face",file),cv2.IMREAD_GRAYSCALE)
        temp = (img,0)
        dataset.append(temp)
        
    # End your code (Part 1)
    return dataset
