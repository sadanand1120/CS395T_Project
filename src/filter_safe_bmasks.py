import os

raw_dir = "dataset/raw_images"
bmask_dir = "dataset/safety_bmask"
all_raw_names = os.listdir(raw_dir)
all_bmasks_names = os.listdir(bmask_dir)
all_bmask_fullpaths = [os.path.join(bmask_dir, a) for a in all_bmasks_names]

for i in range(len(all_bmasks_names)):
    if all_bmasks_names[i] not in all_raw_names:
        os.remove(all_bmask_fullpaths[i])
