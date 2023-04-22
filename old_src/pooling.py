import numpy as np
from PIL import Image
import os
from numpy.lib.stride_tricks import as_strided


def pool2d(A, kernel_size, stride, padding=0, pool_mode='max', padding_mode='maximum'):
    '''
     2D Pooling

     Parameters:
         A: input 2D array
         kernel_size: int, the size of the window over which we take pool
         stride: int, the stride of the window
         padding: int, implicit zero paddings on both sides of the input
         pool_mode: string, 'max' or 'avg'
     '''
    # Padding
    A = np.pad(A, padding, mode=padding_mode)

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                    (A.shape[1] - kernel_size) // stride + 1)

    shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
    strides_w = (stride*A.strides[0], stride *
                 A.strides[1], A.strides[0], A.strides[1])

    A_w = as_strided(A, shape_w, strides_w)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(2, 3))
    elif pool_mode == 'avg':
        return A_w.mean(axis=(2, 3))


def get_bmask_cmap():
    cmap = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    return cmap


def add_pooling(img_path, img_name, save_dir, method, f, cmap=get_bmask_cmap()):
    im_orig = np.array(Image.open(img_path))
    arr = np.array(im_orig[:, :, 0].squeeze()/255, dtype=np.int64)  # 0,1 array
    r, c = arr.shape
    if method == "min":
        new_arr = -pool2d(-arr, kernel_size=f, stride=1, padding=(f-1)//2, pool_mode='max', padding_mode='maximum')
        # new_arr = pool2d(new_arr, kernel_size=f, stride=1, padding=(f-1)//2, pool_mode='max', padding_mode='minimum')
    elif method == "avg":
        new_arr = pool2d(arr, kernel_size=f, stride=1, padding=(f-1)//2, pool_mode='avg', padding_mode='minimum')
    mask = cmap[new_arr]
    im = Image.fromarray(mask)
    im.save(save_dir + img_name)

# _dir = "isInFrontEntrance"
# binary_safe_masks_dir = f"dataset/bmasks/{_dir}/"
binary_safe_masks_dir = "dataset/safety_bmask/"
save_dir = binary_safe_masks_dir
all_image_names = os.listdir(binary_safe_masks_dir)
all_image_fullpaths = [binary_safe_masks_dir + a for a in all_image_names]

for i in range(len(all_image_names)):
    add_pooling(all_image_fullpaths[i], all_image_names[i], save_dir, method="min", f=9)
