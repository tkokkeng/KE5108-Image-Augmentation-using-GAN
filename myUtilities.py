# Import libraries
import os
import numpy as np
from PIL import Image
from PIL import ImageChops

DATA_PATH = "data/train/stage1_train"
TRANSFORMED_PATH = "data/transformed256/"
IMAGE_RESIZE_TO = 256

# This function get all the train images file sizes.
def get_all_train_images_filesizes(data_path=DATA_PATH):

    # List of all images & masks read
    image_list = []

    # Loop through all the directories
    for root, dirs, files in os.walk(data_path):
        for afile in files:
            # if an image file
            rootdir = root.split('/')
            if rootdir[len(rootdir)-1] == 'images':
                # This is an image file
                im = Image.open(os.path.join(root, afile))
                width, height = im.size
                # Add to the image list.
                image_list.append([afile, [width, height]])
    return image_list

# This function processes all the train images from file.
def load_process_all_train_images(data_path=DATA_PATH, resize_to=IMAGE_RESIZE_TO):

    # Flags
    new_image = False # True if a image file is read
    mask_read = False # True if a mask file is read

    # List of all images & masks read
    image_list = []

    # Loop through all the directories
    for root, dirs, files in os.walk(data_path):
        for afile in files:
            # if an image file
            rootdir = root.split('/')
            if rootdir[len(rootdir)-1] == 'images':
                # This is an image file
                new_image = True
                image_file = afile
                a_image = Image.open(os.path.join(root, afile))
                # Initalise the combined masks in a single image
                all_masks = Image.new('1', a_image.size, 0)
            else:
                # This is not an image file, it is therefore a mask file.
                if new_image:
                    # There is an image file already read, combine image and mask.
                    a_mask = Image.open(os.path.join(root, afile))
                    # Combine the masks
                    a_mask = a_mask.convert(mode='1')
                    all_masks = ImageChops.logical_or(all_masks, a_mask)
                    mask_read = True
                # If no image file was already read, ignore the mask file found.

        if mask_read:
            # Crop and resize.
            cropped_image = center_crop_array(a_image)
            cropped_masks = center_crop_array(all_masks)
            if cropped_image.width > resize_to:
                resized_image = cropped_image.resize((resize_to, resize_to))
                resized_masks = cropped_masks.resize((resize_to, resize_to))
            else:
                resized_image = cropped_image
                resized_masks = cropped_masks

            # Reshape masks to shape (H, W, 1) same number of dimensions as image (H, W, 4) i.e. RGBA.
            masks_data = np.asarray(list(resized_masks.getdata()))
            masks_data = masks_data.reshape((resize_to, resize_to, 1))

            # Combine mask to image as additional channel
            image_data = np.asarray(list(resized_image.getdata())).reshape(((resize_to, resize_to, 4)))
            image_and_mask = np.concatenate((image_data, masks_data), axis=2)

            # Add to the image list.
            image_list.append([image_file, image_and_mask])
            # Reinitialise flags
            new_image = False
            mask_read = False

    return image_list

# This function center crops an array along the larger dimension between width and height.
def center_crop_array(arr):

    half_height = arr.height/2
    half_width = arr.width/2
    cropped_arr = arr

    if half_height < half_width:
        # crop by width
        # cropped_arr = arr[:, round(half_width-half_height):round(half_width+half_height), :]
        cropped_arr = arr.crop((round(half_width-half_height), 0, round(half_width+half_height), arr.height))
    elif half_width < half_height:
        # crop by height
        # cropped_arr = arr[:, round(half_height-half_width):round(half_height+half_width), :]
        cropped_arr = arr.crop((0, round(half_height-half_width), arr.width, round(half_height+half_width)))

    return cropped_arr

# This function loads the preprocessed training image data.
def load_preprocessed_data(data_path=TRANSFORMED_PATH):

    image_list = []

    # Loop through all the directories
    for root, dirs, files in os.walk(data_path):
        for afile in files:
            a_image = np.load(os.path.join(root, afile))
            image_list.append(a_image)
    images = np.asarray(image_list)
    return images
