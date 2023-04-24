#!/usr/bin/env python
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import numpy as np
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
from torch.nn import DataParallel as DP
from PIL import Image
import seaborn as sns
import matplotlib.pylab as plt
from scipy.spatial import KDTree

from pool import pool2d


class NeuralPredicates:
    def __init__(self, CKPTS_DIR="ckpts", CKPT_NUM=500):
        self.height = 1280//2
        self.width = 1920//2
        self.ckpts_dir = CKPTS_DIR
        self.ckpt_num = CKPT_NUM
        self._all_classes = os.listdir(self.ckpts_dir)
        self.transformImg = tf.Compose([tf.ToTensor(), tf.Resize((self.height, self.width)), tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.transformAnn = tf.Compose([tf.Resize((self.height, self.width))])
        self.device = self._load_device()
        self.bmask_models = {}
        for c in self._all_classes:
            self.bmask_models[c] = self._load_bmask_model(os.path.join(self.ckpts_dir, c, f"{self.ckpt_num}.pt"))

    def _load_device(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        return device

    def _load_bmask_model(self, cpath):
        Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)  # Load net
        Net.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))  # Change final layer to 2 classes
        Net.backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        Net = DP(Net, device_ids=[0])
        Net = Net.to(self.device)
        ckpt = torch.load(cpath, map_location=self.device)
        Net.load_state_dict(ckpt['state_dict'])
        Net.double()
        Net.eval()
        return Net

    def _process_neural_outputs(self, bmask):
        new_arr = pool2d(bmask, kernel_size=5, stride=1, pool_mode='max')  # to fill holes
        new_arr = pool2d(new_arr, kernel_size=9, stride=1, pool_mode='min')  # to remove outliers, and smoothen boundaries
        return new_arr

    def get_bmask(self, img, model="isAtTurn"):
        """
        Outputs a 0, 1 binary mask for safe pixels
        assumes img in RGB format (PIL)
        """
        def prepare_input_model(img):
            img = np.array(img, dtype=np.float64) / 255.0
            img = self.transformImg(img)
            return img
        img = np.array(img)
        height_ini, width_ini, depth_ini = img.shape
        inp = prepare_input_model(img)
        inp = torch.autograd.Variable(inp, requires_grad=False).to(self.device).unsqueeze(0)
        with torch.no_grad():
            Prd = self.bmask_models[model](inp)['out']  # Run net
        Prd = tf.Resize((height_ini, width_ini))(Prd[0])
        seg = torch.argmax(Prd, 0).detach().cpu().numpy().squeeze()
        return self._process_neural_outputs(seg)

    def binary_to_prob_pool(self, bin):
        return pool2d(bin, kernel_size=25, stride=1, pool_mode='avg')

    def get_closest_dist(self, img, obstacle="static"):
        if obstacle == "static":
            bmask = self.get_bmask(img, model="isStaticObstacle")
        elif obstacle == "dynamic":
            bmask = self.get_bmask(img, model="isDynamicObstacle")
        else:
            raise ValueError("Obstacle type not supported")

        obstacle_pts = np.argwhere(bmask)
        kd = KDTree(obstacle_pts)

        dist = np.zeros_like(bmask)
        r, c = bmask.shape
        for i in range(r):
            for j in range(c):
                dist[i, j], _ = kd.query([i, j])
                print("Done", i, j, "out of", r, c)
        return dist


if __name__ == "__main__":
    neu = NeuralPredicates()
    print("neu loaded")
    img = np.array(Image.open("dataset/raw_images/00025.png"))
    print("image loaded")
    h = neu.get_closest_dist(img, obstacle="static")
    print("h loaded")
    ax = sns.heatmap(h)
    plt.show()
