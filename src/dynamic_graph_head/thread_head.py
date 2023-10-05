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
import signal
import sys

from mim_data_utils import DataLogger, DataReader
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

        self.timing_control = 0. 
        self.timing_utils   = 0. 
        self.timing_logging = 0. 
        self.absolute_time  = 0. 
        self.time_start_recording = 0.
        
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
        except KeyboardInterrupt as exp:
            raise exp
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

    def init_log_stream_fields(self, LOG_FIELDS=['all']):
        fields = []
        fields_access = {}
        for i, ctrl in enumerate(self.active_controllers):
            ctrl_dict = ctrl.__dict__
            for key, value in ctrl_dict.items():
                if(key in LOG_FIELDS or LOG_FIELDS==['all']):
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
        
        # init timings logs 
        self.fields_timings                         = ['timing_utils', 'timing_control', 'timing_logging']
        self.fields_access_timing                   = {}
        self.fields_access_timing['timing_utils']   = {'ctrl' : self, 'key' : 'timing_utils', 'size' : 1}
        self.fields_access_timing['timing_control'] = {'ctrl' : self, 'key' : 'timing_control', 'size' : 1}
        self.fields_access_timing['timing_logging'] = {'ctrl' : self, 'key' : 'timing_logging', 'size' : 1}
        self.fields_access_timing['absolute_time']  = {'ctrl' : self, 'key' : 'absolute_time', 'size' : 1}
        
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

    def start_logging(self, log_duration_s=30, log_filename=None, LOG_FIELDS=['all']):
        if self.logging:
            print('ThreadHead: Already logging data.')
            return
        self.log_duration_s = log_duration_s

        # If no logging yet, then setup the fields to log.
        if not self.streaming:
            self.init_log_stream_fields(LOG_FIELDS=LOG_FIELDS)

        if not log_filename:
            log_filename = time.strftime("%Y-%m-%d_%H-%M-%S") + '.mds'

        self.data_logger = DataLogger(log_filename)
        self.log_filename = log_filename
        
        for name, meta in self.fields_access.items():
            meta['log_id'] = self.data_logger.add_field(name, meta['size'])
        
        # log timings
        self.fields_access_timing['timing_utils']['log_id']   = self.data_logger.add_field('timing_utils', self.fields_access_timing['timing_utils']['size'])
        self.fields_access_timing['timing_control']['log_id'] = self.data_logger.add_field('timing_control', self.fields_access_timing['timing_control']['size'])
        self.fields_access_timing['timing_logging']['log_id'] = self.data_logger.add_field('timing_logging', self.fields_access_timing['timing_logging']['size'])
        self.fields_access_timing['absolute_time']['log_id']  = self.data_logger.add_field('absolute_time', self.fields_access_timing['absolute_time']['size'])

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
        # add timings
        for name, meta in self.fields_access_timing.items():
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
        
        # Optionally generate timing plots when user presses ctrl+c key
        print('\n Press Ctrl+C to plot the timings [FIRST MAKE SURE THE ROBOT IS AT REST OR IN A SAFETY MODE] \n')
        print(' Only works if thread_head.plot_timing() is called in the main \n')
        return abs_filepath


    def plot_timing(self):
        signal.signal(signal.SIGINT, lambda sig, frame : print("\n")) 
        signal.pause()
        r = DataReader(self.log_filename)
        N = r.data['absolute_time'].shape[0]
        clock_time = np.linspace(self.dt, N * self.dt, N) 
        absolute_time_to_clock = r.data['absolute_time'].reshape(-1) - clock_time
        fix, axes = plt.subplots(6, sharex=True, figsize=(8, 12))
        axes[0].plot(r.data['timing_utils'] * 1000)
        axes[1].plot(r.data['timing_control'] * 1000)
        axes[2].plot(r.data['timing_logging'] * 1000)
        axes[3].plot((r.data['timing_utils'] + r.data['timing_control'] + r.data['timing_logging']) * 1000)
        axes[4].plot((r.data['absolute_time'][1:] - r.data['absolute_time'][:-1])* 1000)
        axes[5].plot(absolute_time_to_clock * 1000)
        for ax, title in zip(axes, ['Utils', 'Control', 'Logging', 'Total Computation', 'Cycle Duration', "Cumulative Delay (Absolute Time - Clock Time)"]):
            ax.grid(True)
            ax.set_title(title)
            ax.set_ylabel('Duration [ms]')
            if title != "Cumulative Delay (Absolute Time - Clock Time)":
                ax.axhline(1000*self.dt, color='black')
            else:
                ax.axhline(0., color='black')
        signal.signal(signal.SIGINT, lambda sig, frame : sys.exit(0)) 
        print('\n Press Ctrl+C again to close the timing plots and exit. \n')
        plt.show()
        signal.pause()

    def run_main_loop(self, sleep=False):
        self.absolute_time = time.time() - self.time_start_recording
        
        # Read data from the heads / shared memory.
        for head in self.heads.values():
            head.read()

        # Process the utils.
        start = time.time()
        try:
            for (name, util) in self.utils:
                util.update(self)
        except KeyboardInterrupt as exp:
            raise exp
        except:
            traceback.print_exc()
            print('!!! Error with running util "%s" -> Switching to safety controller.' % (name))
            self.switch_controllers(self.safety_controllers)

        self.timing_utils = time.time() - start

        # Run the active contollers.
        start = time.time()
        try:
            for ctrl in self.active_controllers:
                ctrl.run(self)
        except KeyboardInterrupt as exp:
            raise exp
        except:
            traceback.print_exc()
            print('!!! ThreadHead: Error with running controller -> Switching to safety controller.')
            self.switch_controllers(self.safety_controllers)

        self.timing_control = time.time() - start

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
        self.timing_logging = time.time() - start

        # No need to call stream_data or similar. The data is picked-up from
        # the websocket processing thread async.
        self.ti += 1

    def run(self):
        """ Use this method to start running the main loop in a thread. """
        self.run_loop = True
        self.time_start_recording = time.time()
        next_time = 0.
        while self.run_loop:
            t = time.time() - self.time_start_recording - next_time
            if t >= 0:
                self.run_main_loop()
                next_time += self.dt
            else:
                time.sleep(np.core.umath.maximum(-t, 0.00001))


    def sim_run_timed(self, total_sim_time):
        self.run_loop = True
        self.time_start_recording = time.time()
        next_time = self.dt
        while self.run_loop:
            t = time.time() - self.time_start_recording - next_time
            if t >= 0:
                self.run_main_loop()
                next_time += self.dt
            else:
                time.sleep(np.core.umath.maximum(-t, 0.00001))
            if(next_time >= total_sim_time):
                self.run_loop = False
                
    def sim_run(self, timesteps, sleep=False):
        """ Use this method to run the setup for `timesteps` amount of timesteps. """
        for i in range(timesteps):
            self.run_main_loop(sleep)