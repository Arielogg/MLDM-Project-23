import cv2 as cv

def resize_images(res):
    trajectory_folder = 'data/multi_trajectories'
    new_folder = 'data/patches'
    trajectory_number = '24_30'
    i = 1
    while i <= 50:
        path_read = "../../../" + trajectory_folder + "/" + trajectory_number + "/cut_1/" + str(i) + ".jpg"
        img = cv.imread(path_read)
        resized_image = cv.resize(img, (res, res))
        path_write = "../../../" + new_folder + "/" + str(res) + "/" + str(i) + ".jpg"
        print(path_write)
        cv.imwrite(path_write, resized_image)
        i += 1

patch_sizes = [32, 64, 128, 224, 250, 275, 300, 325, 350, 375, 400, 450, 500, 600, 700, 800, 900, 960]

for patch_size in patch_sizes:
    resize_images(patch_size)