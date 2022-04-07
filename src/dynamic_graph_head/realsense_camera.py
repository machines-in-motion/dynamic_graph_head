"""
License: BSD 3-Clause License
Copyright (C) 2021, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""

## This gets depth data from realsense with dynamic graph head
## Author : Avadesh Meduri
## Date : 7/04/2022

from multiprocessing import Process, Pipe

import pyrealsense2 as rs
import numpy as np
import cv2
import time

def RealSenseCamera(child_conn, img_height = 640, img_width = 480, frame_rate = 60):

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    config.enable_stream(rs.stream.depth, img_height, img_width, rs.format.z16, frame_rate)
    pipeline.start(config)

    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        child_conn.send((depth_colormap))



class VisionSensor:

    def __init__(self, img_height = 640, img_width = 480, frame_rate = 60):

        self.parent_conn, self.child_conn = Pipe()
        self.subp = Process(target=RealSenseCamera, args=(self.child_conn, img_height, img_width, frame_rate))
        self.subp.start()
        self.image = self.parent_conn.recv()

    def get_image(self):
        
        if self.parent_conn.poll():
            self.image = self.parent_conn.recv()

        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense', self.image)
        # cv2.waitKey(1)
        return self.image
