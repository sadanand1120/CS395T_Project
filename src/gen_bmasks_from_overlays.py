import numpy as np
from PIL import Image
import os


def extract_and_save_bmask(raw_dir, overlay_dir, save_dir):
    def get_bmask_cmap():
        cmap = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
        return cmap
    all_image_names = os.listdir(raw_dir)
    for i, img in enumerate(all_image_names):
        raw_path = os.path.join(raw_dir, img)
        overlay_path = os.path.join(overlay_dir, img)
        save_path = os.path.join(save_dir, img)
        img_raw = np.array(Image.open(raw_path))
        img_overlay = np.array(Image.open(overlay_path))
        diff = img_raw - img_overlay
        diff[diff != 0] = 1
        diff = np.sum(diff, axis=-1)  # suppressing channels
        diff[diff != 0] = 1
        mask = get_bmask_cmap()[diff]
        img_save = Image.fromarray(mask)
        img_save.save(save_path)
        print(f"Saved {i+1}/{len(all_image_names)} images")


if __name__ == "__main__":
    raw_dir = "dataset/raw_images"
    overlay_dir = "dataset/manual_overlays"
    save_dir = "dataset/bmasks"
    all_classes = os.listdir(overlay_dir)
    for _class_name in all_classes:
        extract_and_save_bmask(raw_dir, os.path.join(overlay_dir, _class_name), os.path.join(save_dir, _class_name))
