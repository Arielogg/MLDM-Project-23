import numpy as np
import pandas as pd
import os
from PIL import Image
import re

#Change this with your own directories
image_dir = "D:/Datasets/MLDM Project/Data/0.20__2"
new_imagedir = str(image_dir) + '_cropped'

if not os.path.exists(new_imagedir):
    os.makedirs(new_imagedir)

for filename in os.listdir(image_dir):
    if re.match(r'^\d+\.jpg$', filename):
        img = Image.open(os.path.join(image_dir, filename))

        #Center crop dimensions
        width, height = img.size
        left = (width - 512) // 2
        top = (height - 512) // 2
        right = left + 512
        bottom = top + 512

        #Resize the image to 224x224 pixels
        img_cropped = img.crop((left, top, right, bottom)).resize((224, 224))

        #Save the cropped image
        img_cropped.save(os.path.join(new_imagedir, 'cropped' + filename))