import numpy as np
from PIL import Image
import os


def add_1_mid_pix(bmask_dir, save_dir):
    def get_bmask_cmap():
        cmap = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0]], dtype=np.uint8)
        return cmap
    all_image_names = os.listdir(bmask_dir)
    for k, img in enumerate(all_image_names):
        bmask_path = os.path.join(bmask_dir, img)
        save_path = os.path.join(save_dir, img)
        img_bmask = np.array(Image.open(bmask_path))
        arr = np.array(img_bmask[:, :, 0].squeeze()/255, dtype=np.int64)  # 0,1 array
        r, c = arr.shape
        counter = 0
        for i in range(1, r-1):
            for j in range(1, c-1):
                if arr[i, j] == 1 and arr[i-1, j] == 0 and arr[i+1, j] == 0 and arr[i, j-1] == 0 and arr[i, j+1] == 0 and arr[i-1, j-1] == 0 and arr[i-1, j+1] == 0 and arr[i+1, j-1] == 0 and arr[i+1, j+1] == 0:
                    arr[i, j] = 2
                    counter += 1
                elif arr[i, j] == 0 and arr[i-1, j] == 1 and arr[i+1, j] == 1 and arr[i, j-1] == 1 and arr[i, j+1] == 1 and arr[i-1, j-1] == 1 and arr[i-1, j+1] == 1 and arr[i+1, j-1] == 1 and arr[i+1, j+1] == 1:
                    arr[i, j] = 2
                    counter += 1
        mask = get_bmask_cmap()[arr]
        im = Image.fromarray(mask)
        im.save(save_path)
        print(f"Saved {k+1}/{len(all_image_names)} images, had {counter} pixels changed")


if __name__ == "__main__":
    bmask_dir = "dataset/bmasks"
    save_dir = bmask_dir
    all_classes = os.listdir(bmask_dir)
    for _class_name in all_classes:
        print(f"Processing {_class_name}")
        bmask_path = os.path.join(bmask_dir, _class_name)
        save_path = os.path.join(save_dir, _class_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        add_1_mid_pix(bmask_path, save_path)
