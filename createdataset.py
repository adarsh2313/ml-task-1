import cv2
import os
import numpy as np
import random

def create_images():
    # Creating 1200 16x16 grayscale images with random pixel values between 0 and 128
    
    img = []
    
    for i in range(1200):
        img.append(np.random.randint(0,128,(16,16,1),np.uint8))

    # Drawing a square on the first 400 images (i from 0 to 399)
    random.seed(0)
    for i in range(400):
        s = random.randint(0,8)
        f = random.randint(4,8)
        e = s+f
        img[i] = cv2.rectangle(img[i],(s,s),(e,e),(255),-1)
    
     # Drawing a rectangle on the next 400 images (i from 400 to 799)
    random.seed(2)
    for i in range(400,800):
       s = random.randint(0,6)
       e = random.randint(0,4)
       f = random.randint(8,12)
       img[i] = cv2.rectangle(img[i],(s,f),(f+e,s+e),(255),-1)
    
    # Drawing a circle on the next 400 images (i from 800 to 1200)
    random.seed(5)
    for i in range(800,1200):
       x = random.randint(4,8)
       y = random.randint(4,8)
       radius = random.randint(3,6)
       img[i] = cv2.circle(img[i],(x,y),radius,(255),-1)

    image_dataset = np.zeros((1200,16,16,1))
    for i in range(1200):
        image_dataset[i] = img[i]
    
    return image_dataset