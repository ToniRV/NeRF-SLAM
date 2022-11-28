#!/usr/bin/env python3

from pipeline.pipeline_module import MIMOPipelineModule

class SlamModule(MIMOPipelineModule):
    def __init__(self, name, args, device="cpu"):
        super().__init__(name, args.parallel_run, args)
        self.device = device

    def spin_once(self, input):
        output = self.slam(input)
        if not output or self.slam.stop_condition():
            super().shutdown_module()
        return output
 
    def initialize_module(self):
        if self.name == "VioSLAM":
            from slam.vio_slam import VioSLAM
            self.slam = VioSLAM(self.name, self.args, self.device)
        else:
            raise NotImplementedError
        return super().initialize_module()
