import subprocess
import os
import random
import numpy as np

from PIL import Image

from inference import *

class PixelSafety:
    def __init__(self, dist_mins):
        self.dist_mins = dist_mins

        self.neural_predicates = NeuralPredicates()

        self.accuracy_mul = 1000

    def eval_pixel_safety(self, img):
        """

        :param img:         Raw image to evaluate safety of pixels
        :return:            Array of 1s and 0s corresponding to True/False regarding pixel safety estimate
        """

        is_turn = temp_lib.is_turn(img)
        is_confined_safe = temp_lib.is_confined_safe(img)
        is_dynamic = temp_lib.is_dynamic(img)
        is_static = temp_lib.is_static(img)
        is_in_front = temp_lib.is_in_front(img)
        is_sidewalk = temp_lib.is_sidewalk(img)
        is_road = temp_lib.is_road(img)
        #is_bad_terrain = temp_lib.is_bad_terrain(img)

        static_obs_min_dist = temp_lib.min_dist(img, 'static')
        dynamic_obs_min_dist = temp_lib.min_dist(img, 'dynamic')

        safety_val = np.ones(is_turn.size)

        #Vectorized pixel safety computation
        zero_tensor = np.zeros(1)

        #Update with is_turn
        safety_val = np.where(is_turn >= 1.0, zero_tensor, safety_val)

        #Update with is dynamic
        safety_val = np.where(is_dynamic >= 1.0, zero_tensor, safety_val)

        #Update with is_static
        safety_val = np.where(is_static >= 1.0, zero_tensor, safety_val)

        #Update with is_bad_terrain
        safety_val = np.where(is_road <= 0.0, zero_tensor, safety_val)
        safety_val = np.where(is_sidewalk <= 0.0, zero_tensor, safety_val)

        #Update with is_in_front
        safety_val = np.where(is_in_front >= 1.0, zero_tensor, safety_val)

        #Update with sidewalk condtions
        safety_val = np.where((is_sidewalk >= 1.0 and (static_obs_min_dist < self.dist_mins['sidewalk_static'] or
                                                          dynamic_obs_min_dist < self.dist_mins['sidewalk_dynamic'])),
                                 zero_tensor, safety_val)

        #Update with road conditions
        safety_val = np.where((is_road >= 1.0 and (static_obs_min_dist < self.dist_mins['road_static'] or
                                                          dynamic_obs_min_dist < self.dist_mins['road_dynamic']) and
                                  is_confined_safe <= 0.0),
                                 zero_tensor, safety_val)

        return safety_val

    def get_dist_params(self, training_imgs, num_samples=1000):
        #Encode pixel truth values into constraint sat problem (multiply distances by 1000 (or something) so can work with integers
        #Get examples of true/false based on different locations and distances
        #Use rosette to solve encoding and get returned model

        """
        High level:
        1. Get sample of pixels that aren't turns/obstacles/bad terrain/in front
        2. Get min static/dynamic distances
            a. If meant to be safe encode static < static_dist and dynamic < dynamic_dist
            b. Else encode static > static_dist or dynamic > dynamic_dist
            c. Keep track of largest static/dynamic distances that are lower bounds
            d. Keep track of smallest static/dynamic distances taht are upper bounds
        3. Run Rosette solver
        4. If not unsat save values -- done
        5. If unsat remove constraints saved in 2c and 2d. and rerun iteratively until sat model found
        """

        rosette_encode_str = "#lang rosette/safe\n\n" \
                             "(current-bitwidth #f)\n\n" \
                             "(define-symbolic road_stat road_dyn side_stat side_dyn integer?)\n\n" \
                             "(solve\n" \
                             "\t(assert (and"

        #Loop through training data
        list_of_upper_bounds = []
        for (img, safety) in training_imgs:
            is_road_mask = self.neural_predicates.get_bmask(img, 'isRoad')

            static_obs_min_dist_mask = self.neural_predicates.get_closest_dist(img, 'static') * self.accuracy_mul
            dynamic_obs_min_dist_mask = self.neural_predicates.get_closest_dist(img, 'dynamic') * self.accuracy_mul

            print(static_obs_min_dist_mask)

            for i in range(num_samples):
                #Get pixel location
                pixel_val = self.get_applicable_pixel(img)

                is_road = is_road_mask[pixel_val]
                static_obs_min_dist = static_obs_min_dist_mask[pixel_val]
                dynamic_obs_min_dist = dynamic_obs_min_dist_mask[pixel_val]

                if safety[pixel_val] and is_road:
                    rosette_encode_str += f"\n\t\t(< road_stat {static_obs_min_dist})"
                    rosette_encode_str += f"\n\t\t(< road_dyn {dynamic_obs_min_dist})"
                    list_of_upper_bounds.append(static_obs_min_dist)
                    list_of_upper_bounds.append(dynamic_obs_min_dist)
                elif not safety[pixel_val] and is_road:
                    rosette_encode_str += f"\n\t\t(or (> road_stat {static_obs_min_dist}) (> road_dyn {dynamic_obs_min_dist}))"
                elif safety[pixel_val] and not is_road:
                    rosette_encode_str += f"\n\t\t(< side_stat {static_obs_min_dist})"
                    rosette_encode_str += f"\n\t\t(< side_dyn {dynamic_obs_min_dist})"
                    list_of_upper_bounds.append(static_obs_min_dist)
                    list_of_upper_bounds.append(dynamic_obs_min_dist)
                else:
                    rosette_encode_str += f"\n\t\t(or (> side_stat {static_obs_min_dist}) (> side_dyn {dynamic_obs_min_dist}))"

        list_of_upper_bounds.sort()

        rosette_encode_str += "\t))\n)"

        #Write rosette string to file and evaluate
        with open('temp.rkt', "w") as f:
            f.write(rosette_encode_str)

        script_arg = 'raco test temp.rkt'
        proc = subprocess.run(['C:\\Windows\\System32\\bash.exe', '-l', '-c', script_arg], capture_output=True)

        parse = proc.stdout.decode()

        while parse.find('unsat') != -1:
            #Update rosette string to remove largest lower bound
            largest_lower = str(list_of_upper_bounds.pop(0))

            idxs_to_remove = []
            idx = 0
            for line in rosette_encode_str.split('\n'):
                if line.find(largest_lower):
                    idxs_to_remove.append(idx)

                idx += 1

            new_rosette_encode_str = ""
            idx = 0
            for line in rosette_encode_str.split('\n'):
                if idx not in idxs_to_remove:
                    new_rosette_encode_str += "\n" + line

            #Delete temp.rkt and rewrite
            os.remove('temp.rkt')
            with open('temp.rkt', "w") as f:
                f.write(rosette_encode_str)

            #Rerun
            proc = subprocess.run(['C:\\Windows\\System32\\bash.exe', '-l', '-c', script_arg], capture_output=True)

            parse = proc.stdout.decode()

        #Parse output and update params
        self.set_ints(parse)

    def get_applicable_pixel(self, img):
        """

        :param image:   Image to check validity against
        :return:        Pixel coordinate that is on road or sidewalk, not violating any safety concerns beside distance
        """

        valid_pixel = False
        pixel_loc = None
        while not valid_pixel:
            pixel_x = random.randint(0, img.x_max)
            pixel_y = random.randint(0, img.x_max)

            pixel_loc = (pixel_x, pixel_y)

            valid_pixel = True

            is_turn = temp_lib.is_turn(img, pixel_loc)
            is_confined_safe = temp_lib.is_confined_safe(img, pixel_loc)
            is_dynamic = temp_lib.is_dynamic(img, pixel_loc)
            is_static = temp_lib.is_static(img, pixel_loc)
            is_in_front = temp_lib.is_in_front(img, pixel_loc)
            is_road = temp_lib.is_road(img, pixel_loc)
            is_bad_terrain = temp_lib.is_bad_terrain(img, pixel_loc)

            # Evaluate pixel safety
            if is_turn or is_dynamic or is_static or is_bad_terrain or is_in_front or (is_road and is_confined_safe):
                # These cases are unsafe no matter what
                valid_pixel = False

        return pixel_loc

    def set_ints(self, parse):
        int_list = []
        for sub_str in parse.split(' '):
            if sub_str.isdigit():
                int_list.append(int(sub_str))

        self.dist_mins['road_static'] = int_list[0] / float(self.accuracy_mul)
        self.dist_mins['road_dynamic'] = int_list[1] / float(self.accuracy_mul)
        self.dist_mins['sidewalk_static'] = int_list[2] / float(self.accuracy_mul)
        self.dist_mins['sidewalk_dynamic'] = int_list[3] / float(self.accuracy_mul)


if __name__ == "__main__":
    img = np.array(Image.open('dataset/raw_images/00025.png'))
    gt_safe = np.array(Image.open('dataset/safety_bmask/00025.png'))

    training_set = [(img, gt_safe)]

    #Init module
    evaluator = PixelSafety(None)

    evaluator.get_dist_params(training_set, num_samples=2)
