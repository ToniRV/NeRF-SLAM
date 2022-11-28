#!/usr/bin/env python3

import os
import sys
sys.settrace
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from icecream import ic
import argparse

import numpy as np

import cv2

from datasets.data_module import DataModule

def parse_args():
    parser = argparse.ArgumentParser(description="RealSense Recorder")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    args.parallel_run = False
    args.dataset_dir = "/home/tonirv/Datasets/RealSense"
    args.dataset_name = "real"

    args.initial_k   = 0
    args.final_k     = 0
    args.img_stride  = 1
    args.stereo      = False

    data_provider_module = DataModule(args.dataset_name, args, device="cpu")
    data_provider_module.initialize_module()

    # Start once we press 's'
    print("Waiting to start recorgin, press 's'.")
    cv2.imshow("Click 's' to start; 'q' to stop", np.ones((200,200)))
    while cv2.waitKey(33) != ord('s'):
        continue

    print("Recording")
    data_packets = []
    while cv2.waitKey(33) != ord('q'): # quit
        output = data_provider_module.spin_once("aha")
        data_packets += [output]
    print("Stopping")
    ic(len(data_packets))

    # Write images to disk, and save path with to_nerf() function
    print("Saving to nerf format")
    data_provider_module.dataset.to_nerf_format(data_packets)
    print('Done...')

