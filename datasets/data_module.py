#!/usr/bin/env python3

import colored_glog as log
from pipeline.pipeline_module import MIMOPipelineModule

class DataModule(MIMOPipelineModule):
    def __init__(self, name, args, device="cpu") -> None:
        super().__init__(name, args.parallel_run, args)
        self.device = device
        self.idx = -1

    def get_input_packet(self):
        return True

    def spin_once(self, input):
        log.check(input)
        if self.name == 'real':
            return self.dataset.stream()
        else:
            self.idx += 1
            if self.idx < len(self.dataset):
                return self.dataset[self.idx]
            else:
                print("Stopping data module!")
                super().shutdown_module()
            return None

    def initialize_module(self):
        if self.name == "euroc":
            from datasets.euroc_dataset import EurocDataset
            self.dataset = EurocDataset(self.args, self.device)
        elif self.name == "tum":
            from datasets.tum_dataset import TumDataset
            self.dataset = TumDataset(self.args, self.device)
        elif self.name == "nerf":
            from datasets.nerf_dataset import NeRFDataset
            self.dataset = NeRFDataset(self.args, self.device)
        elif self.name == "replica":
            from datasets.replica_dataset import ReplicaDataset
            self.dataset = ReplicaDataset(self.args, self.device)
        elif self.name == "real":
            from datasets.real_sense_dataset import RealSenseDataset
            self.dataset = RealSenseDataset(self.args, self.device)
        else:
            raise Exception(f"Unknown dataset: {self.name}")
        return super().initialize_module()
