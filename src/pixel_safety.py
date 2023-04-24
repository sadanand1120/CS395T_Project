import subprocess
import os
import random
import numpy as np

from PIL import Image

from inference import *

class PixelSafety:
    def __init__(self, dist_mins={}):
        if dist_mins == {}:
            self.dist_mins['road_static'] = 0.0
            self.dist_mins['road_dynamic'] = 0.0
            self.dist_mins['sidewalk_static'] = 0.0
            self.dist_mins['sidewalk_dynamic'] = 0.0

        else:
            self.dist_mins = dist_mins

        self.neural_predicates = NeuralPredicates()

        self.accuracy_mul = 1000

    def eval_pixel_safety(self, img):
        """

        :param img:         Raw image to evaluate safety of pixels
        :return:            Array of 1s and 0s corresponding to True/False regarding pixel safety estimate
        """

        is_turn = self.neural_predicates.get_bmask(img, 'isAtTurn')
        is_confined_safe = self.neural_predicates.get_bmask(img, 'isConfinedSafe')
        is_dynamic = self.neural_predicates.get_bmask(img, 'isDynamicObstacle')
        is_static = self.neural_predicates.get_bmask(img, 'isStaticObstacle')
        is_in_front = self.neural_predicates.get_bmask(img, 'isInFrontEntrance')
        is_road = self.neural_predicates.get_bmask(img, 'isRoad')
        is_sidewalk = self.neural_predicates.get_bmask(img, 'isSidewalk')

        #static_obs_min_dist = temp_lib.min_dist(img, 'static')
        #dynamic_obs_min_dist = temp_lib.min_dist(img, 'dynamic')

        static_obs_min_dist = torch.load('dataset/inferred_dist_bmasks/00025_static.pt')
        dynamic_obs_min_dist = torch.load('dataset/inferred_dist_bmasks/00025_dynamic.pt')

        safety_val = np.ones_like(is_turn)

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
        safety_val = np.where(((is_sidewalk >= 1.0) & ((static_obs_min_dist < self.dist_mins['sidewalk_static']) |
                                                     (dynamic_obs_min_dist < self.dist_mins['sidewalk_dynamic']))),
                                 zero_tensor, safety_val)

        #Update with road conditions
        safety_val = np.where(((is_road >= 1.0) & ((static_obs_min_dist < self.dist_mins['road_static']) |
                                                 (dynamic_obs_min_dist < self.dist_mins['road_dynamic'])) &
                               (is_confined_safe <= 0.0)),
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
            is_turn = self.neural_predicates.get_bmask(img, 'isAtTurn')
            is_confined_safe = self.neural_predicates.get_bmask(img, 'isConfinedSafe')
            is_dynamic = self.neural_predicates.get_bmask(img, 'isDynamicObstacle')
            is_static = self.neural_predicates.get_bmask(img, 'isStaticObstacle')
            is_in_front = self.neural_predicates.get_bmask(img, 'isInFrontEntrance')
            is_road_mask = self.neural_predicates.get_bmask(img, 'isRoad')
            is_sidewalk = self.neural_predicates.get_bmask(img, 'isSidewalk')
            is_bad_terrain = is_road_mask + is_sidewalk
            is_bad_terrain = np.where(is_bad_terrain > 0, 0.0, 1.0)

            # static_obs_min_dist_mask = self.neural_predicates.get_closest_dist(img, 'static') * self.accuracy_mul
            # torch.save(static_obs_min_dist_mask, 'dataset/inferred_dist_bmasks/00025_static.pt')
            # dynamic_obs_min_dist_mask = self.neural_predicates.get_closest_dist(img, 'dynamic') * self.accuracy_mul
            # torch.save(dynamic_obs_min_dist_mask, 'dataset/inferred_dist_bmasks/00025_dynamic.pt')

            static_obs_min_dist_mask = torch.load('dataset/inferred_dist_bmasks/00025_static.pt')
            dynamic_obs_min_dist_mask = torch.load('dataset/inferred_dist_bmasks/00025_dynamic.pt')

            print("Done Image Inference")

            for i in range(num_samples):
                #Get pixel location
                (pixel_x, pixel_y) = self.get_applicable_pixel(is_turn, is_confined_safe, is_dynamic, is_static, is_in_front, is_road_mask, is_bad_terrain)

                is_road = is_road_mask[pixel_x, pixel_y]
                static_obs_min_dist = static_obs_min_dist_mask[pixel_x, pixel_y]
                dynamic_obs_min_dist = dynamic_obs_min_dist_mask[pixel_x, pixel_y]

                safety_val = safety[pixel_x, pixel_y][0] and safety[pixel_x, pixel_y][1] and safety[pixel_x, pixel_y][2]

                if safety_val and is_road:
                    rosette_encode_str += f"\n\t\t(< road_stat {static_obs_min_dist})"
                    rosette_encode_str += f"\n\t\t(< road_dyn {dynamic_obs_min_dist})"
                    list_of_upper_bounds.append(static_obs_min_dist)
                    list_of_upper_bounds.append(dynamic_obs_min_dist)
                elif not safety_val and is_road:
                    rosette_encode_str += f"\n\t\t(or (> road_stat {static_obs_min_dist}) (> road_dyn {dynamic_obs_min_dist}))"
                elif safety_val and not is_road:
                    rosette_encode_str += f"\n\t\t(< side_stat {static_obs_min_dist})"
                    rosette_encode_str += f"\n\t\t(< side_dyn {dynamic_obs_min_dist})"
                    list_of_upper_bounds.append(static_obs_min_dist)
                    list_of_upper_bounds.append(dynamic_obs_min_dist)
                else:
                    rosette_encode_str += f"\n\t\t(or (> side_stat {static_obs_min_dist}) (> side_dyn {dynamic_obs_min_dist}))"

        list_of_upper_bounds.sort()

        rosette_encode_str += "\n\t))\n)"

        #Write rosette string to file and evaluate
        with open('temp.rkt', "w") as f:
            f.write(rosette_encode_str)

        proc = subprocess.run(['raco','test','--timeout','60','temp.rkt'], capture_output=True)

        parse = proc.stdout.decode()

        while parse.find('unsat') != -1:
            #Update rosette string to remove smallest upper bound
            largest_lower = str(list_of_upper_bounds.pop(0))
            print(largest_lower)

            idxs_to_remove = []
            idx = 0
            for line in rosette_encode_str.split('\n'):
                if line.find(largest_lower) != -1:
                    idxs_to_remove.append(idx)
                    print('Found: ', idx)

                idx += 1

            new_rosette_encode_str = ""
            idx = 0
            for line in rosette_encode_str.split('\n'):
                if idx not in idxs_to_remove:
                    new_rosette_encode_str += "\n" + line
                else:
                    print('Removing: ', idx)

                idx += 1

            #Delete temp.rkt and rewrite
            os.remove('temp.rkt')
            with open('temp.rkt', "w") as f:
                f.write(new_rosette_encode_str)

            #Update old rosette encoding string to ensure correct lines are removed
            rosette_encode_str = new_rosette_encode_str

            #Rerun
            proc = subprocess.run(['raco','test','--timeout','60','temp.rkt'], capture_output=True)

            parse = proc.stdout.decode()

        #Parse output and update params
        print(parse)
        self.set_ints(parse)

    def get_applicable_pixel(self, is_turn_mask, is_confined_safe_mask, is_dynamic_mask, is_static_mask, is_in_front_mask, is_road_mask,
                                          is_bad_terrain_mask):

        valid_pixel = False
        pixel_loc = None
        while not valid_pixel:
            pixel_x = random.randint(0, np.shape(is_turn_mask)[0]-1)
            pixel_y = random.randint(0, np.shape(is_turn_mask)[1]-1)

            pixel_loc = (pixel_x, pixel_y)

            is_turn = is_turn_mask[pixel_x, pixel_y]
            is_dynamic = is_dynamic_mask[pixel_x, pixel_y]
            is_static = is_static_mask[pixel_x, pixel_y]
            is_bad_terrain = is_bad_terrain_mask[pixel_x, pixel_y]
            is_in_front = is_in_front_mask[pixel_x, pixel_y]
            is_road = is_road_mask[pixel_x, pixel_y]
            is_confined_safe = is_confined_safe_mask[pixel_x, pixel_y]

            valid_pixel = True

            # Evaluate pixel safety
            if is_turn or is_dynamic or is_static or is_bad_terrain or is_in_front or (is_road and is_confined_safe):
                # These cases are unsafe no matter what
                valid_pixel = False

        return pixel_loc

    def set_ints(self, parse):
        int_list = []
        parse = parse.replace('\n', ' ')
        for sub_str in parse.split(' '):
            stripped_sub_str = str(sub_str).strip("()]")
            if stripped_sub_str.isdigit():
                int_list.append(int(stripped_sub_str))

        if len(int_list) == 0:
            print(parse)

        if int_list[0] / float(self.accuracy_mul) > self.dist_mins['road_static']:
            self.dist_mins['road_static'] = int_list[0] / float(self.accuracy_mul)
        if int_list[1] / float(self.accuracy_mul) > self.dist_mins['road_dynamic']:
            self.dist_mins['road_dynamic'] = int_list[1] / float(self.accuracy_mul)
        if len(int_list) >= 4:
            if int_list[2] > 0:
                if int_list[2] / float(self.accuracy_mul) > self.dist_mins['sidewalk_static']:
                    self.dist_mins['sidewalk_static'] = int_list[2] / float(self.accuracy_mul)
            else:
                if self.dist_mins['road_static'] > self.dist_mins['sidewalk_static']:
                    self.dist_mins['sidewalk_static'] = self.dist_mins['road_static']

            if int_list[3] > 0:
                if int_list[3] / float(self.accuracy_mul) > self.dist_mins['sidewalk_dynamic']:
                    self.dist_mins['sidewalk_dynamic'] = int_list[3] / float(self.accuracy_mul)
            else:
                if self.dist_mins['road_dynamic'] > self.dist_mins['sidewalk_dynamic']:
                    self.dist_mins['sidewalk_dynamic'] = self.dist_mins['road_dynamic']
        else:
            if self.dist_mins['road_static'] > self.dist_mins['sidewalk_static']:
                self.dist_mins['sidewalk_static'] = self.dist_mins['road_static']
            if self.dist_mins['road_dynamic'] > self.dist_mins['sidewalk_dynamic']:
                self.dist_mins['sidewalk_dynamic'] = self.dist_mins['road_dynamic']


if __name__ == "__main__":
    img = np.array(Image.open('dataset/raw_images/00025.png'))
    gt_safe = np.array(Image.open('dataset/safety_bmask/00025.png'))

    training_set = [(img, gt_safe)]

    #Init module
    #evaluator = PixelSafety()

    #evaluator.get_dist_params(training_set, num_samples=1000)

    evaluator = PixelSafety({'road_static': 625.0, 'road_dynamic': 345.0, 'sidewalk_static': 625.0, 'sidewalk_dynamic': 345.0})

    #Print parse params
    print("Road Static: ", evaluator.dist_mins['road_static'])
    print("Road Dynamic: ", evaluator.dist_mins['road_dynamic'])
    print("Sidewalk Static: ", evaluator.dist_mins['sidewalk_static'])
    print("Sidewalk Dynamic: ", evaluator.dist_mins['sidewalk_dynamic'])

    gen_safety_mask = evaluator.eval_pixel_safety(img)

    ax = sns.heatmap(gen_safety_mask)
    plt.show()