from abc import abstractmethod
import colored_glog as log
from icecream import ic

#from torch.profiler import profile, ProfilerActivity

class PipelineModuleBase:
    def __init__(self, name, parallel_run, args=None, grad=False):
        self.name = name
        self.parallel_run = parallel_run
        self.grad = grad # determines if we are tracing grads or not 
        self.shutdown = False # needs to be atomic
        self.is_initialized = False # needs to be atomic
        self.is_thread_working = False # needs to be atomic
        self.args = args # arguments to init module
        # Callbacks to be called in case module does not return an output.
        self.on_failure_callbacks = []
        self.profile = False # Profile the code for runtime and/or memory

    @abstractmethod
    def initialize_module(self):
        # Allocate memory and initialize variables here so that 
        # we do not need to avoid when running in parallel a:
        # "TypeError: cannot pickle 'XXXX' object" 
        self.is_initialized = True

    @abstractmethod
    def spin(self) -> bool:
        pass

    @abstractmethod
    def shutdown_queues(self):
        pass

    @abstractmethod
    def has_work(self):
        pass

    def shutdown_module(self):
        # TODO shouldn't self.shutdown be atomic? (i.e. thread-safe?)
        if self.shutdown:
            log.warn(f"Module: {self.name} - Shutdown requested, but was already shutdown.")
        log.debug(f"Stopping module {self.name}  and its queues...")
        self.shutdown_queues()
        log.info(f"Module: {self.name} - Shutting down.")
        self.shutdown = True

    def restart(self):
        log.info(f"Module: {self.name} - Resetting shutdown flag to false")
        self.shutdown = False

    def is_working(self):
        return self.is_thread_working or self.hasWork()

    def register_on_failure_callback(self, callback):
        log.check(callback)
        self.on_failure_callbacks.append(callback)

    def notify_on_failure(self):
        for on_failure_callback in self.on_failure_callbacks:
            if on_failure_callback:
                on_failure_callback()
            else:
                log.error(f"Invalid OnFailureCallback for module: {self.name}")

class PipelineModule(PipelineModuleBase):
    def __init__(self, name_id, parallel_run, args=None, grad=False) -> None:
        super().__init__(name_id, parallel_run, args, grad)

    @abstractmethod
    def get_input_packet(self):
        raise

    @abstractmethod
    def push_output_packet(self, output_packet) -> bool:
        raise

    @abstractmethod
    def spin_once(self, input):
        raise

    # Spin is called in a thread.
    def spin(self):
        if self.parallel_run:
            log.info(f'Module: {self.name} - Spinning.')

        if not self.is_initialized:
            self.initialize_module()

        while not self.shutdown:
            self.is_thread_working = False;
            input = self.get_input_packet();
            self.is_thread_working = True;
            if input is not None:
                output = None
                if self.profile:
                    #with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
                    output = self.spin_once(input);
                    #print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
                else:
                    output = self.spin_once(input);
                if output is not None:
                    # Received a valid output, send to output queue
                    if not self.push_output_packet(output):
                        log.warn(f"Module: {self.name} - Output push failed.")
                    else:
                        log.debug(f"Module: {self.name} - Pushed output.")
                else:
                    log.debug(f"Module: {self.name} - Skipped sending an output.")
                    # Notify interested parties about failure.
                    self.notify_on_failure();
            else:
                log.log(2, f"Module: {self.name} - No Input received.")

            # Break the while loop if we are in sequential mode.
            if not self.parallel_run:
                self.is_thread_working = False;
                return True;

        self.is_thread_working = False;
        log.info(f"Module: {self.name} - Successful shutdown.")
        return False;

class MIMOPipelineModule(PipelineModule):
    def __init__(self, name_id, parallel_run, args=None, grad=False):
        super().__init__(name_id, parallel_run, args, grad)
        self.input_queues = {}
        self.output_callbacks = []
        self.output_queues = []

    def register_input_queue(self, name, input_queue):
        self.input_queues[name] = input_queue

    def register_output_callback(self, output_callback):
        self.output_callbacks.append(output_callback)

    def register_output_queue(self, output_queue):
        self.output_queues.append(output_queue)

    # TODO: warn when callbacks take too long
    def push_output_packet(self, output_packet):
        push_success = True
        # Push output to all queues
        for output_queue in self.output_queues:
            try:
                output_queue.put(output_packet)
            except Exception as e:
                log.warn(e)
                push_success = False
        # Push output to all callbacks
        for callback in self.output_callbacks:
            try:
                callback(output_packet)
            except Exception as e:
                log.warn(e)
                push_success = False
        return push_success

    def get_input_packet(self, timeout=0.1):
        inputs = {}
        if self.parallel_run:
            for name, input_queue in self.input_queues.items():
                try:
                    inputs[name] = input_queue.get(timeout=timeout)
                except Exception as e:
                    log.debug(e)
        else:
            for name, input_queue in self.input_queues.items():
                try:
                    inputs[name] = input_queue.get_nowait()
                except Exception as e:
                    log.debug(e)

        if len(inputs) == 0:
            log.debug(f"Module: {self.name} - Input queues didn't return an output.")
            inputs = None
        return inputs
    
    def shutdown_queues(self):
        super().shutdown_queues()
        # This should be automatically called by garbage collector
        # [input_queue.close() for input_queue in self.input_queues]
        # [output_queue.close() for output_queue in self.output_queues]