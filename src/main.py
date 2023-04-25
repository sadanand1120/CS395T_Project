import math
import numpy as np
from pool import pool2d
from recursive_belief_update import RBU
import os
from PIL import Image
from pixel_safety import PixelSafety
import seaborn as sns
from matplotlib import pyplot as plt
from copy import deepcopy


class SpotAzureKinectCameraProjection:
    def __init__(self):
        """
        Frames nomenclature:
        (1) World Coordinates: Attached to the robot at the centre of robot's base (Z up, X forward, Y towards robot's left hand side)
        (2) Camera Pose: Attached at the camera centre, basically translate (1) to that point and then rotate about +Y axis so as to align +X such that it comes straight out of the camera lens
        (3) Camera Coordinates: Attached at the camera centre, basically rotate (2 rotations) (2) such that the new +Y is along old -Z and new +X is along old -Y
        (4) Pixel Coordinaes: 3D to 2D intrinsic matrix brings (3) to (4)
        """
        self.M_intrinsic = self.get_M_intrinsic()
        self.kinect_pose_params = self.get_kinect_pose_params()
        self.M_perspective = self.get_M_perspective()
        self.mats = self.get_projection_mats()

    def get_projection_mats(self):
        T_43 = self.M_intrinsic @ self.M_perspective  # 3x4
        T_32 = self.get_std_rot("X", -math.pi/2) @ self.get_std_rot("Z", -math.pi/2)  # 4x4
        T_21 = self.get_std_rot(*self.kinect_pose_params["rot"]) @ self.get_std_trans(*self.kinect_pose_params["trans"])  # 4x4
        T_41 = T_43 @ T_32 @ T_21  # 3x4
        H_41 = T_41[:, [0, 1, 3]]  # homography for Z=0 plane, 3x3
        H_14 = np.linalg.inv(H_41)  # 3x3
        return {"T_43": T_43, "T_32": T_32, "T_21": T_21, "T_41": T_41, "H_41": H_41, "H_14": H_14}

    def get_M_intrinsic(self):
        # 3x3
        mat = [
            [976.2046, 0, 1020.4302],
            [0, 976.1627, 773.6796],
            [0, 0, 1]
        ]
        return np.array(mat)

    def get_kinect_pose_params(self):
        # metres, radians
        # first did translation, then rotation
        params = {}
        params["cx"] = 24.5 / 100
        params["cy"] = -2 / 100   # initially I measured 7, but -2 seems to be the correct one
        params["cz"] = 76 / 100
        params["trans"] = (params["cx"], params["cy"], params["cz"])
        params["rot"] = ("Y", np.deg2rad(15.2))
        return params

    def get_ordinary_from_homo(self, v):
        # Scales so that last coord is 1 and then removes last coord
        o = v.squeeze()
        o = o/o[-1]
        return o[:-1]

    def get_homo_from_ordinary(self, v):
        o = list(v.squeeze())
        o.append(1)
        return np.array(o)

    def get_std_trans(self, cx, cy, cz):
        # cx, cy, cz are the coords of O_M wrt O_F when expressed in F
        mat = [
            [1, 0, 0, -cx],
            [0, 1, 0, -cy],
            [0, 0, 1, -cz],
            [0, 0, 0, 1]
        ]
        return np.array(mat)

    def get_std_rot(self, axis, alpha):
        # axis is either "X", "Y", or "Z" axis of F and alpha is positive acc to right hand thumb rule dirn
        if axis == "X":
            mat = [
                [1, 0, 0, 0],
                [0, math.cos(alpha), math.sin(alpha), 0],
                [0, -math.sin(alpha), math.cos(alpha), 0],
                [0, 0, 0, 1]
            ]
        elif axis == "Y":
            mat = [
                [math.cos(alpha), 0, -math.sin(alpha), 0],
                [0, 1, 0, 0],
                [math.sin(alpha), 0, math.cos(alpha), 0],
                [0, 0, 0, 1]
            ]
        elif axis == "Z":
            mat = [
                [math.cos(alpha), math.sin(alpha), 0, 0],
                [-math.sin(alpha), math.cos(alpha), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        else:
            raise ValueError("Invalid axis!")
        return np.array(mat)

    def get_M_perspective(self):
        # 3x4
        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        return np.array(mat)


def get_world_from_pixels(p_x, p_y, k):
    loc = k.get_homo_from_ordinary(np.array([p_x, p_y]))
    loc_w = k.get_ordinary_from_homo(k.mats["H_14"] @ loc).squeeze()
    return loc_w  # (w_x, w_y)


def get_global_from_world(rob_x, rob_y, rob_theta, w_x, w_y, k: SpotAzureKinectCameraProjection):
    p_o = np.array([rob_x, rob_y])
    p_w = np.array([w_x, w_y])
    T = (k.get_std_rot("Z", rob_theta).T)[:2, :2]
    return p_o + T @ p_w


def safety_visualize(rob_x, rob_y, rob_theta, rbu: RBU, x_range, y_range):
    arr = np.zeros((int((x_range[1]-x_range[0])/rbu.grid_size+2), int((y_range[1]-y_range[0])/rbu.grid_size+2)))
    x = round(x_range[0], 4)
    y = round(y_range[0], 4)
    i = 0
    j = 0
    while x <= x_range[1]:
        y = round(y_range[0], 4)
        j = 0
        while y <= y_range[1]:
            arr[arr.shape[0] - 1 - i, j] = rbu.safety_belief[(x, y)]
            y = round(y + rbu.grid_size, 4)
            j += 1
        x = round(x+rbu.grid_size, 4)
        i += 1
    h = sns.heatmap(arr)
    plt.show()
    plt.close()


# defining the state space
x_range = (10, 130)
y_range = (-280, -30)

k = SpotAzureKinectCameraProjection()
rbu = RBU(x_range, y_range, grid_size=0.1, alpha=0.7, beta=0.6)
prog_synth_evaluator = PixelSafety({'road_static': 625.0, 'road_dynamic': 345.0, 'sidewalk_static': 625.0, 'sidewalk_dynamic': 345.0})
raw_dir = "dataset/raw_images"
odom_path = "dataset/odom.txt"
all_img_names = os.listdir(raw_dir)
all_img_names = [name[:-4] for name in all_img_names]
all_img_names.sort()

N_x = 2048
N_y = 1536
Y_BOUND = 900
SEQ_DATA = {}  # (timestep: (img_name, (x_glo, y_glo, theta_glo)))

with open(odom_path, "r") as f:
    t = 0
    while True:
        odom_data = f.readline()
        if not odom_data:
            break
        img_name, odom_tup = odom_data.split(": ")
        SEQ_DATA[t] = (img_name, eval(odom_tup[:-1]))
        t += 1

for t in range(len(SEQ_DATA)):
    img_path = os.path.join(raw_dir, SEQ_DATA[t][0]+".png")
    img = np.array(Image.open(img_path))
    rob_x_glo, rob_y_glo, rob_theta_glo = SEQ_DATA[t][1]
    visible_locs = {}  # (x_glo,y_glo): cur_estim
    bmask = prog_synth_evaluator.eval_pixel_safety(img, SEQ_DATA[t][0])
    prob_mask = prog_synth_evaluator.neural_predicates.binary_to_prob_pool(bmask)
    new_prob_mask = deepcopy(prob_mask)
    for p_x in range(N_x):
        for p_y in range(Y_BOUND, N_y):
            print(f"Doing {p_x}, {p_y} out of {N_x}, {N_y}")
            loc_w = get_world_from_pixels(p_x, p_y, k)
            global_w = get_global_from_world(rob_x_glo, rob_y_glo, rob_theta_glo, loc_w[0], loc_w[1], k)
            global_w = tuple(global_w)
            global_w = rbu.round_and_range(*global_w)
            if global_w not in visible_locs.keys():
                visible_locs[global_w] = prob_mask[p_y, p_x]
            else:
                visible_locs[global_w] = max(visible_locs[global_w], prob_mask[p_y, p_x])
    rbu.update(visible_locs)
    for p_x in range(N_x):
        for p_y in range(Y_BOUND, N_y):
            loc_w = get_world_from_pixels(p_x, p_y, k)
            global_w = get_global_from_world(rob_x_glo, rob_y_glo, rob_theta_glo, loc_w[0], loc_w[1], k)
            global_w = tuple(global_w)
            global_w = rbu.round_and_range(*global_w)
            new_prob_mask[p_y, p_x] = rbu.safety_belief[global_w]
    h = sns.heatmap(new_prob_mask)
    plt.savefig(f"dataset/safety_heatmaps/{SEQ_DATA[t][0]}.png")
    plt.close()

    # VIS_BOUND = 3
    # low = (rob_x_glo-VIS_BOUND, rob_y_glo-VIS_BOUND)
    # upp = (rob_x_glo+VIS_BOUND, rob_y_glo+VIS_BOUND)
    # low = rbu.round_and_range(*low)
    # upp = rbu.round_and_range(*upp)
    # safety_visualize(rob_x_glo, rob_y_glo, rob_theta_glo, rbu, x_range=(low[0], upp[0]), y_range=(low[1], upp[1]))


"""
create an RBF object (should have initial bel over all states x,y as 0)
for all timesteps t (basically read img and odom data sequentially):
    visible_locs = {}  # (x_glo,y_glo): cur_estim
    use RBF's PixelSafety object to get bmask of safeness on this image
    apply RBF's PixelSafety's NeuPred's bin_to_prob to get prob_mask
    for all pixels:
        apply Spot's homography to get coords acc to world_frame at robot base
        apply Spot's another transformation to get global coords from world
        add to visible_locs
    pass visible_locs to RBF's update method
"""
