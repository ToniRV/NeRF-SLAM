#!/usr/bin/env python3

from pipeline.pipeline_module import MIMOPipelineModule

class GuiModule(MIMOPipelineModule):
    def __init__(self, name, args, device="cpu") -> None:
        super().__init__(name, args.parallel_run, args)
        self.device = device

    def spin_once(self, data_packet):
        # If the queue is empty, queue.get() will block until the queue has data
        output = self.gui.visualize(data_packet)
        #if not output:
        #    super().shutdown_module()
        return output

    def initialize_module(self):
        if self.name == "Open3DGui":
            from gui.open3d_gui import Open3dGui
            self.gui = Open3dGui(self.args, self.device)
        elif self.name == "DearPyGui":
            raise NotImplementedError
        else:
            raise NotImplementedError
        self.gui.initialize()
        return super().initialize_module()

    def get_input_packet(self):
        # don't block rendering waiting for input
        input = super().get_input_packet(timeout=0.000000001)
        # so that we keep running, and do not just stop in spin()
        return input if input is not None else False
