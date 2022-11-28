#!/usr/bin/env python3

import os
import sys
sys.settrace
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
from tqdm import tqdm

from datasets.data_module import DataModule

def parse_args():
    parser = argparse.ArgumentParser(description="INSTANT SLAM")
    parser.add_argument("--replica_dir", type=str,
                        help="Path to the Replica dataset root dir",
                        default="/home/tonirv/Datasets/Replica/")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    args.parallel_run = True
    args.dataset_dir = None
    args.initial_k = 0 # first frame to load
    args.final_k = -1 # last frame to load, if -1 load all
    args.img_stride = 1 # stride for loading images
    args.stereo = False
    args.buffer = 3000

    transform = "transforms.json"
    dataset_names = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
    for dataset in tqdm(dataset_names):
        args.dataset_dir = os.path.join(args.replica_dir, dataset)
        print(f"Processing dataset: {args.dataset_dir}")
        # Parse dataset and transform to Nerf
        data_provider_module = DataModule("replica", args)
        data_provider_module.initialize_module()
        data_provider_module.dataset.to_nerf_format()
        # Copy transforms.json to its args.dataset_dir folder
        os.replace(transform, os.path.join(args.dataset_dir, transform))
    print('Done...')

