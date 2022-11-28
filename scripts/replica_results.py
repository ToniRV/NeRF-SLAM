#!/usr/bin/env python3

import os
import sys
sys.settrace
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from icecream import ic

import argparse
from tqdm import tqdm
import copy
import torch

from examples.slam_demo import run

def parse_args():
    parser = argparse.ArgumentParser(description="INSTANT SLAM")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    args.parallel_run = True
    args.multi_gpu = True

    args.initial_k = 0
    args.final_k = 2000

    args.stereo = False

    args.weights = "droid.pth"
    args.network = ""
    args.width = 0
    args.height = 0

    args.dataset_dir = None
    args.dataset_name = "nerf"

    args.slam   = True
    args.fusion = 'nerf'
    args.gui    = True

    args.eval   = True

    args.sigma_fusion = True


    args.buffer = 100
    args.img_stride = 1

    # "ours", "ours_w_thresh" or "raw", "no_depth"
    args1 = copy.deepcopy(args)
    args1.mask_type = "ours"

    args2 = copy.deepcopy(args)
    args2.mask_type = "no_depth"

    args3 = copy.deepcopy(args)
    args3.mask_type = "raw"

    args4 = copy.deepcopy(args)
    args4.mask_type = "ours_w_thresh"

    args_to_test = {
        #args1.mask_type: args1,
        #args2.mask_type: args2,
        args3.mask_type: args3,
        #args4.mask_type: args4,
    }


    results = "results.csv"
    datasets = [
        #"/home/tonirv/Datasets/nerf-cube-diorama-dataset/room",
        #"/home/tonirv/Datasets/nerf-cube-diorama-dataset/bluebell",
        #"/home/tonirv/Datasets/nerf-cube-diorama-dataset/book",
        #"/home/tonirv/Datasets/nerf-cube-diorama-dataset/cup",
        #"/home/tonirv/Datasets/nerf-cube-diorama-dataset/laptop"
        #"/home/tonirv/Datasets/Replica/office0",
        #"/home/tonirv/Datasets/Replica/office2",
        #"/home/tonirv/Datasets/Replica/office3",
        #"/home/tonirv/Datasets/Replica/office4",
        #"/home/tonirv/Datasets/Replica/room1",
        #"/home/tonirv/Datasets/Replica/room2",
        # These get stuck not sure why....
        #"/home/tonirv/Datasets/Replica/room0",
        "/home/tonirv/Datasets/Replica/office1",
    ]

    torch.multiprocessing.set_start_method('spawn')
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    for dataset in tqdm(datasets):
        for test_name, test_args in args_to_test.items():
            output = os.path.basename(dataset) + '_nerf_' + test_name + '_' + results
            ic(output)
            test_args.dataset_dir = dataset
            print(f"Processing dataset: {test_args.dataset_dir}")
            try:
                run(test_args)
            except Exception as e:
                print(e)
            print(f"Saving output in: {output}")
            # Copy transforms.json to its args.dataset_dir folder
            os.replace(results, os.path.join(output))
            #torch.cuda.empty_cache()
    print('Done...')

