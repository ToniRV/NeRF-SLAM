from pipeline.pipeline_module import MIMOPipelineModule

class FusionModule(MIMOPipelineModule):
    def __init__(self, name, args, device="cpu") -> None:
        super().__init__(name, args.parallel_run, args)
        self.device = device

    def spin_once(self, data_packet):
        output = self.fusion.fuse(data_packet)
        # TODO: if you uncomment this, we never reach gui/fusion loop, but if you comment it never stops.
        if self.fusion.stop_condition():
            print("Stopping fusion module!")
            super().shutdown_module()
        #if not output:
        #    super().shutdown_module()
        return output

    def initialize_module(self):
        self.set_cuda_device() # This needs to be done before importing NerfFusion or TsdfFusion
        if self.name == "tsdf" or self.name == "sigma":
            from fusion.tsdf_fusion import TsdfFusion
            self.fusion = TsdfFusion(self.name, self.args, self.device)
        elif self.name == "nerf":
            from fusion.nerf_fusion import NerfFusion
            self.fusion = NerfFusion(self.name, self.args, self.device)
        else:
            raise NotImplementedError
        return super().initialize_module()

    def get_input_packet(self):
        input = super().get_input_packet(timeout=0.0000000001) # don't block fusion waiting for input
        return input if input is not None else False # so that we keep running, and do not just stop in spin()

    def set_cuda_device(self):
        if self.device == "cpu":
            return

        import os
        if self.device == "cuda:0":
            os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        elif self.device == "cuda:1":
            os.environ['CUDA_VISIBLE_DEVICES'] = "1"
            self.device = "cuda:0" # Since only 1 will be visible...
        else:
            raise NotImplementedError
