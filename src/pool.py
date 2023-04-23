import numpy as np
import os
from PIL import Image


def _pool2d(A, kernel_size, stride, pool_mode='max', pad_mode='constant', pad_width=0, pad_value=0):
    A = np.pad(A, mode=pad_mode, pad_width=pad_width, constant_values=pad_value)

    output_shape = ((A.shape[0] - kernel_size) // stride + 1, (A.shape[1] - kernel_size) // stride + 1)

    shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
    strides_w = (stride*A.strides[0], stride * A.strides[1], A.strides[0], A.strides[1])

    A_w = np.lib.stride_tricks.as_strided(A, shape_w, strides_w)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(2, 3))
    elif pool_mode == 'avg':
        return A_w.mean(axis=(2, 3))
    else:
        raise ValueError("pool_mode must be 'max' or 'avg'")


def pool2d(bmask, kernel_size, stride, pool_mode='max'):
    if pool_mode == 'max':
        return _pool2d(bmask, kernel_size=kernel_size, stride=stride, pool_mode='max', pad_mode='constant', pad_width=(kernel_size-1)//2, pad_value=0)
    elif pool_mode == 'avg':
        return _pool2d(bmask, kernel_size=kernel_size, stride=stride, pool_mode='avg', pad_mode='constant', pad_width=(kernel_size-1)//2, pad_value=0)
    elif pool_mode == 'min':
        return -_pool2d(-bmask, kernel_size=kernel_size, stride=stride, pool_mode='max', pad_mode='constant', pad_width=(kernel_size-1)//2, pad_value=-1)
    else:
        raise ValueError("pool_mode must be 'max' or 'avg' or 'min'")


def _save_bmask_pool(img_path, img_name, save_dir):
    def get_bmask_cmap():
        cmap = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
        return cmap
    im_orig = np.array(Image.open(img_path))
    arr = np.array(im_orig[:, :, 0].squeeze()/255, dtype=np.int64)  # 0,1 array
    new_arr = pool2d(arr, kernel_size=5, stride=1, pool_mode='max')  # to fill holes
    new_arr = pool2d(new_arr, kernel_size=17, stride=1, pool_mode='min')  # to remove outliers, and smoothen boundaries
    mask = get_bmask_cmap()[new_arr]
    im = Image.fromarray(mask)
    im.save(save_dir + img_name)


if __name__ == "__main__":
    binary_safe_masks_dir = "dataset/safety_bmask/"
    save_dir = binary_safe_masks_dir
    all_image_names = os.listdir(binary_safe_masks_dir)
    all_image_fullpaths = [binary_safe_masks_dir + a for a in all_image_names]
    for i in range(len(all_image_names)):
        _save_bmask_pool(all_image_fullpaths[i], all_image_names[i], save_dir)
