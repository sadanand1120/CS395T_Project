import numpy as np
from PIL import Image
import os


def get_bmask_cmap():
    cmap = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    return cmap


def extract_and_save_bmask(img_path, overlay_path, img_name, save_dir, cmap=get_bmask_cmap()):
    im_orig = np.array(Image.open(img_path))
    im_overlay = np.array(Image.open(overlay_path))
    diff = im_orig - im_overlay
    diff[diff != 0] = 1
    diff = np.sum(diff, axis=-1)  # suppressing channels
    diff[diff != 0] = 1
    mask = cmap[diff]
    im = Image.fromarray(mask)
    im.save(save_dir + img_name)


images_dir = "../dataset/raw_images/"
overlay_dir = "../dataset/manual_overlays/isRoad/"
binary_safe_masks_dir = "../dataset/bmasks/isRoad/"
all_image_names = os.listdir(images_dir)
all_image_fullpaths = [images_dir + a for a in all_image_names]
all_overlay_fullpaths = [overlay_dir + a for a in all_image_names]

for i in range(len(all_image_names)):
    extract_and_save_bmask(
        all_image_fullpaths[i], all_overlay_fullpaths[i], all_image_names[i], binary_safe_masks_dir)
