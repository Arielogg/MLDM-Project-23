import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import os
import glob

# We set the route for the images
#features_dir_raw = r"../../../generations/original_cropped_generations/20_2/"
#features_dir_raw = r"../../../generations/changemap_generations/cropped_cropped/result_24_cut_4_png/"
features_dir_raw = r"../../../generations/changemap_generations/cropped_resized/result_24_cut_4_png/"
#features_dir_raw = r"../../../generations/deformation_generations/registered/20/"
#features_dir_raw = r"../../../generations/deformation_generations_overfitted/registered/24/"

filelist_raw = sorted(glob.glob(f'{features_dir_raw}*.png'))
if len(filelist_raw) == 0:
    filelist_raw = sorted(glob.glob(f'{features_dir_raw}*.jpg'))

# We open the images and define the stack of images
res = 224
frames = []
for filename in filelist_raw:
    img = Image.open(filename).convert('L')
    frames.append(img)

features = np.stack(frames, axis=0)

# Calculate PSNR function
def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


# We define the ssim across the images in sequence
ssim_across = []
psnr_across = []
for i in range(len(features) - 1):
    fixed_image = np.reshape(features[i], (res, res, 1))
    next_image  = np.reshape(features[i+1], (res, res, 1))
    ssim_skimg = ssim(fixed_image, next_image, channel_axis=-1, data_range=255)
    psnr_skimg = calculate_psnr(fixed_image, next_image)
    ssim_across.append(ssim_skimg)
    psnr_across.append(psnr_skimg)

print(f"Average SSIM in the raw data: {round(np.mean(ssim_across), 4)}")
print(f"Average PSNR in the raw data: {round(np.mean(psnr_across), 4)}")
