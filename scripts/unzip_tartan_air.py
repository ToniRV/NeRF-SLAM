#!/usr/bin/env python3

import glob
import os
import os.path as osp
from tqdm import tqdm
import shutil

cur_path = osp.dirname(osp.abspath(__file__))

# Download dataset should be:
# python download_training.py --rgb --depth --only-left --output-dir ./datasets/TartanAir

# Dirs inside datasets/TartanAir must be
# {dataset}/{level}/P***/{depth_left,image_left,pose_left.txt}

LEVELS = ['Easy', 'Hard']

def unzip(tartanair_path='datasets/TartanAir', remove_zip=False):
    datasets_paths = glob.glob(osp.join(tartanair_path, "*"))
    for dataset in tqdm(sorted(datasets_paths)):
        dataset_name = os.path.basename(dataset)
        print("Dataset: %s" % dataset_name)
        for level in LEVELS:
            print("Level: %s"%(level))

            ### Form paths and pass checks
            dataset_level_path = osp.join(dataset, level)
            depth_dataset_level_path = osp.join(dataset_level_path, "depth_left.zip")
            if not osp.exists(depth_dataset_level_path):
                print("Missing Depth zip file for Dataset/Level: %s/%s" %(dataset, level))
                continue
            image_dataset_level_path = osp.join(dataset_level_path, "image_left.zip")
            if not osp.exists(image_dataset_level_path):
                print("Missing Image zip file for Dataset/Level: %s/%s"%(dataset, level))
                continue

            if osp.exists(osp.join(dataset_level_path, dataset_name)) or len(glob.glob(osp.join(dataset_level_path, "P*"))) != 0:
                print("Seems like the dataset was already unzipped? %s" % dataset)
            else:
                ### Unzip dataset
                command = "unzip -q -n %s -d %s"%(depth_dataset_level_path, dataset_level_path)
                print(command)
                os.system(command)
                if remove_zip:
                    os.remove(depth_dataset_level_path)
                command = "unzip -q -n %s -d %s"%(image_dataset_level_path, dataset_level_path)
                print(command)
                os.system(command)
                if remove_zip:
                    os.remove(image_dataset_level_path)

            ### Remove junk directories
            from_ = osp.join(dataset_level_path, "*/*/*/P*")
            if len(glob.glob(from_)) != 0: # We have junk folders
                to_ = dataset_level_path
                command = "mv %s %s"%(from_,to_)
                print(command)
                os.system(command)
                shutil.rmtree(osp.join(dataset_level_path,dataset_name))

import argparse

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=dir_path)
    parser.add_argument('--remove_zip', action="store_true")
    args = parser.parse_args()

    print("Unzipping TartanAir dataset")
    unzip(args.dataset_path, args.remove_zip)




