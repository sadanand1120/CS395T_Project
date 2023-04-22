import os
import numpy as np
import torch
import torchvision.transforms as tf
from PIL import Image


def store_img_preproc(old_dir, new_dir, transformImg, transformAnn, mode="raw"):
    def get_bmask_cmap():
        cmap = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0]], dtype=np.uint8)
        return cmap

    def image_loader(dir, img_name, toFloat=False):
        out = np.array(Image.open(os.path.join(dir, img_name)))
        if toFloat:
            out = np.array(out, dtype=np.float64)
            out = out / 255.0
        return out

    def myToTensor(x):
        return transformAnn(torch.FloatTensor(x).permute(2, 0, 1))

    def mask1D_to_rgb(mask1D, cmap=get_bmask_cmap()):
        if mask1D.shape[-1] == 1:
            mask1D = mask1D.squeeze()
        return cmap[mask1D]

    def rgb_to_mask1D(rgb, cmap=get_bmask_cmap()):
        ret = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        for i in range(2):
            y = cmap[i]
            ret[(rgb == y).all(2)] = i
        return ret.reshape(rgb.shape[0], rgb.shape[1], 1)

    all_old_names = os.listdir(old_dir)
    for i, img in enumerate(all_old_names):
        new_path = os.path.join(new_dir, img[:-4] + ".pt")
        if mode == "raw":
            _img = image_loader(old_dir, img, toFloat=True)
            _img = transformImg(_img)
            torch.save(_img, new_path)
        elif mode == "bmask":
            _img = rgb_to_mask1D(image_loader(old_dir, img))
            _img = myToTensor(_img).squeeze()
            torch.save(_img, new_path)
        else:
            raise ValueError("Invalid mode")
        print(f"Saved {i+1}/{len(all_old_names)} images")


if __name__ == "__main__":
    height = 1280//2
    width = 1920//2
    transformImg = tf.Compose([tf.ToTensor(), tf.Resize((height, width)), tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    transformAnn = tf.Compose([tf.Resize((height, width))])
    old_dir = "dataset/bmasks"
    new_dir = "opt_dataset/bmasks"
    all_classes = os.listdir(old_dir)
    for _class_name in all_classes:
        print(f"Processing {_class_name}")
        bmask_path = os.path.join(old_dir, _class_name)
        save_path = os.path.join(new_dir, _class_name)
        store_img_preproc(bmask_path, save_path, transformImg, transformAnn, mode="bmask")
