"""__init__
License: BSD 3-Clause License
Copyright (C) 2021, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""

import os, os.path
import time
import numpy as np
import traceback

import array
import asyncio
import datetime
import json
import websockets
import threading

from mim_data_utils import DataLogger
import matplotlib.pylab as plt

class ThreadHead(threading.Thread):
    def __init__(self, dt, safety_controllers, heads, utils, env=None):
        threading.Thread.__init__(self)

        self.dt = dt
        self.env = env

        if type(heads) != dict:
            self.head = heads # Simple-edge-case for single head setup.
            self.heads = {
                'default': heads
            }
        else:
            self.heads = heads

        self.utils = utils
        for (name, util) in utils:
            self.__dict__[name] = util

        self.ti = 0
        self.streaming = False
        self.streaming_event_loop = None
        self.logging = False
        self.log_writing = False

        self.timing_N = 20000
        self.timing_control = np.zeros(self.timing_N)
        self.timing_utils = np.zeros(self.timing_N)
        self.timing_logging = np.zeros(self.timing_N)

        # Start the websocket thread/server and publish data if requested.
        self.ws_thread = None

        # Read data from the heads / shared memory to have it available for the
        # initial utils and safety-controller run.
        for head in self.heads.values():
            head.read()

        # Run the utils once to make sure the data is available for
        # the safety controller.
        for (name, util) in self.utils:
            util.update(self)

        self.active_controllers = None
        if type(safety_controllers) != list and type(safety_controllers) != tuple:
            safety_controllers = [safety_controllers]
        self.safety_controllers = safety_controllers
        self.switch_controllers(safety_controllers)

    def switch_controllers(self, controllers):
        # Switching the controller changes the fields.
        # Therefore, stopping streaming and logging.
        self.stop_streaming()
        self.stop_logging()

        if type(controllers) != list and type(controllers) != tuple:
            controllers = [controllers]

        # Warmup the controllers and run them once to
        # get all fields propaged and have the data ready
        # for logging / streaming.
        try:
            for ctrl in controllers:
                ctrl.warmup(self)
                ctrl.run(self)

            self.active_controllers = controllers
        except:
            traceback.print_exc()
            print('!!! ThreadHead: Error during controller warmup & run -> Switching to safety controller.')
            self.active_controllers = self.safety_controllers

            for ctrl in self.active_controllers:
                ctrl.warmup(self)
                ctrl.run(self)

    def ws_thread_fn(self):
        print("Hello world from websocket thread.", self)

        async def handle_client(websocket, path):
            while True:
                data = {}
                data['time'] = self.ti / 1000.

                for name, value in self.fields_access.items():
                    val = value['ctrl'].__dict__[value['key']]
                    if type(val) == np.ndarray and val.ndim == 1:
                        type_str = 'd' if val.dtype == np.float64 else 'f'
                        data[name] = str(array.array(type_str, val.data))
                    else:
                        # Fake sending data as an array to the client.
                        data[name] = "array('d', [" + str(val) + "])"

                streaming_json_data = json.dumps(data)

                if self.streaming:
                    await websocket.send(streaming_json_data)
                await asyncio.sleep(0.01)

        # Init an event loop.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Init streaming of data using websockets.
        self.websocket_server = websockets.serve(handle_client, "127.0.0.1", 5678)
        self.streaming_event_loop = asyncio.get_event_loop()
        self.streaming_event_loop.run_until_complete(self.websocket_server)
        self.streaming_event_loop.run_forever()

    def init_log_stream_fields(self):
        fields = []
        fields_access = {}
        for i, ctrl in enumerate(self.active_controllers):
            ctrl_dict = ctrl.__dict__
            for key, value in ctrl_dict.items():
                # Support only single-dim numpy arrays and scalar only.
                if type(value) == float or type(value) == int:
                    field_size = 1
                elif type(value) == np.ndarray and value.ndim == 1:
                    field_size = value.shape[0]
                else:
                    print("  Not logging '%s' as field type '%s' is unsupported" % (
                        key, str(type(value))))
                    continue

                if len(self.active_controllers) == 1:
                    name = key
                else:
                    name = 'ctrl%02d.%s' % (i, key)
                fields.append(name)
                fields_access[name] = {
                    'ctrl': ctrl,
                    'key': key,
                    'size': field_size
                }

        self.fields = fields
        self.fields_access = fields_access

    def start_streaming(self):
        if self.streaming:
            print('!!! ThreadHead: Already streaming data.')
            return
        self.streaming = True

        if self.ws_thread is None:
            self.ws_thread = threading.Thread(target=self.ws_thread_fn)
            self.ws_thread.start()

        # If no logging yet, then setup the fields to log.
        if not self.logging:
            self.init_log_stream_fields()

        print('!!! ThreadHead: Start streaming data.')

    def stop_streaming(self):
        if not self.streaming:
            return

        self.streaming = False

        print('!!! ThreadHead: Stop streaming data.')

    def start_logging(self, log_duration_s=30):
        if self.logging:
            print('ThreadHead: Already logging data.')
            return
        self.log_duration_s = log_duration_s

        # If no logging yet, then setup the fields to log.
        if not self.streaming:
            self.init_log_stream_fields()

        self.data_logger = DataLogger(time.strftime("%Y-%m-%d_%H-%M-%S") + '.mds')

        for name, meta in self.fields_access.items():
            meta['log_id'] = self.data_logger.add_field(name, meta['size'])

        print('!!! ThreadHead: Start logging to file "%s" for %0.2f seconds.' % (
            self.data_logger.filepath, log_duration_s))
        self.logging = True

    def log_data(self):
        if not self.logging:
            return

        # Indicate that writing is happening to the file and that the file
        # should not be clsoed right now.
        self.log_writing = True
        dl = self.data_logger
        dl.begin_timestep()
        for name, meta in self.fields_access.items():
            dl.log(meta['log_id'], meta['ctrl'].__dict__[meta['key']])
        dl.end_timestep()
        self.log_writing = False

        if dl.file_index * self.dt >= self.log_duration_s:
            self.stop_logging()

    def stop_logging(self):
        if not self.logging:
            return

        self.logging = False

        # If there are logs written to the file right now, wait a bit to finish
        # the current logging iteration.
        if self.log_writing:
            time.sleep(10 * self.dt)

        self.data_logger.close_file()
        abs_filepath = os.path.abspath(self.data_logger.filepath)
        print('!!! ThreadHead: Stop logging to file "%s".' % (abs_filepath))
        return abs_filepath


    def plot_timing(self):
        fix, axes = plt.subplots(4, sharex=True, figsize=(8, 10))

        axes[0].plot(self.timing_utils * 1000)
        axes[1].plot(self.timing_control * 1000)
        axes[2].plot(self.timing_logging * 1000)
        axes[3].plot((self.timing_utils + self.timing_control + self.timing_logging) * 1000)

        for ax, title in zip(axes, ['Util', 'Control', 'Logging', 'Total Duration']):
            ax.grid(True)
            ax.set_ylabel('Duration [ms]')
            ax.set_title(title)
            ax.axhline(1., color='red')

        plt.show()

    def run_main_loop(self, sleep=False):
        timing_N = self.timing_N

        # Read data from the heads / shared memory.
        for head in self.heads.values():
            head.read()

        # Process the utils.
        start = time.time()
        try:
            for (name, util) in self.utils:
                util.update(self)
        except:
            traceback.print_exc()
            print('!!! Error with running util "%s" -> Switching to safety controller.' % (name))
            self.switch_controllers(self.safety_controllers)

        self.timing_utils[self.ti % timing_N] = time.time() - start

        # Run the active contollers.
        start = time.time()
        try:
            for ctrl in self.active_controllers:
                ctrl.run(self)
        except:
            traceback.print_exc()
            print('!!! ThreadHead: Error with running controller -> Switching to safety controller.')
            self.switch_controllers(self.safety_controllers)

        self.timing_control[self.ti % timing_N] = time.time() - start

        # Write the computed control back to shared memory.
        for head in self.heads.values():
            head.write()

        # If an env is povided, step it.
        if self.env:
            # Step the simulation multiple times if thread_head is running
            # at a lower frequency.
            for i in range(int(self.dt/self.env.dt)):
                # Need to apply the commands at each timestep of the simulation
                # again.
                for head in self.heads.values():
                    head.sim_step()

                # Step the actual simulation.
                self.env.step(sleep=sleep)

        start = time.time()
        self.log_data()
        self.timing_logging[self.ti % timing_N] = time.time() - start

        # No need to call stream_data or similar. The data is picked-up from
        # the websocket processing thread async.
        self.ti += 1

    def run(self):
        """ Use this method to start running the main loop in a thread. """
        self.run_loop = True
        next_time = time.time() + self.dt
        while self.run_loop:
            if time.time() >= next_time:
                next_time += self.dt
                self.run_main_loop()
            else:
                time.sleep(0.0001)

    def sim_run(self, timesteps, sleep=False):
        """ Use this method to run the setup for `timesteps` amount of timesteps. """
        for i in range(timesteps):
            self.run_main_loop(sleep)