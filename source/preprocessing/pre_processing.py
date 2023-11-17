from PIL import Image, ImageFilter
import numpy as np
import os
import cv2

image_dirs = [
    '/Users/arielguerra/Documents/MLDM/S3/MLDM Project/Data/Original/file/2023-07_n1-to-n50-as-jpg/0.20__2',
    '/Users/arielguerra/Documents/MLDM/S3/MLDM Project/Data/Original/file/2023-07_n1-to-n50-as-jpg/0.22__30',
    '/Users/arielguerra/Documents/MLDM/S3/MLDM Project/Data/Original/file/2023-07_n1-to-n50-as-jpg/0.24__30'
]

crop_size = (2560, 1920)
sub_image_size = min(crop_size) // 4

new_dir = '/Users/arielguerra/Documents/MLDM/S3/MLDM Project/Data/MultiTrajectories/'
os.makedirs(new_dir, exist_ok=True)

#Iterating over each trajectory
for i, image_dir in enumerate(image_dirs):
    new_sub_dir = os.path.join(new_dir, f'sub_dir_{i+1}')
    os.makedirs(new_sub_dir, exist_ok=True)

    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(image_dir, filename))
            print('processing ', filename)

            img_cropped = img.crop((0, 0, *crop_size))
            img_cropped = cv2.fastNlMeansDenoisingColored(np.array(img_cropped), None, 10, 10, 7, 21)
            img_cropped = Image.fromarray(img_cropped).convert('L')

            #Creating sub-images
            for j in range(4):
                left = (j % 2) * sub_image_size
                top = (j // 2) * sub_image_size
                right = left + sub_image_size
                bottom = top + sub_image_size
                if right <= crop_size[0] and bottom <= crop_size[1]:
                    img_sub = img_cropped.crop((left, top, right, bottom))
                    new_sub_sub_dir = os.path.join(new_sub_dir, f'cut_{j+1}')
                    os.makedirs(new_sub_sub_dir, exist_ok=True)
                    img_sub.save(os.path.join(new_sub_sub_dir, filename))