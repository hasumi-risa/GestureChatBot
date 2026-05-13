# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function

from abc import ABC, abstractmethod



#------------------------------------------------------------------------------
# Gaussian filter parameters
# 
class GaussParams:
    def __init__(self, window_size=31, sigma=3):
        self.window_size = window_size # Kernel size
        self.sigma = sigma # Gaussian curve standard distribution





#------------------------------------------------------------------------------
# Base class for Kinect joint data-to-Labanotation algorithms.
# 
class AlgorithmBase(ABC):
    # Default Gaussian filter parameters
    default_gauss_window_size = 31 # Gaussian filter kernel size
    default_gauss_sigma = 3 # Gaussian filter kernal's standard deviation of distribution


    #------------------------------------------------------------------------------
    # Property gauss_params:  Gaussian filter parameters
    #
    @property
    def gauss_params(self):
        "Gaussian filter parameters property"
        return (self.gauss_window_size, self.gauss_sigma)


    @gauss_params.setter
    def gauss_params(self, gauss_params):
        self.gauss_window_size = gauss_params[0]
        self.gauss_sigma = gauss_params[1]


    #------------------------------------------------------------------------------
    # Property gauss_params_default:  Default Gaussian filter parameters
    #
    @property
    def gauss_params_default(self):
        "Gaussian filter default parameters property"
        return (self.default_gauss_window_size, self.default_gauss_sigma)



    #------------------------------------------------------------------------------
    # Return the default Gaussian filter parameters
    #
    @classmethod
    def get_gauss_params_default(cls):
        return (cls.default_gauss_window_size, cls.default_gauss_sigma)



    #------------------------------------------------------------------------------
    # Initialize this class instance
    # 
    def __init__(self):
        # Gaussian filter parameters
        self.gauss_window_size = self.default_gauss_window_size
        self.gauss_sigma = self.default_gauss_sigma


    #------------------------------------------------------------------------------
    # Convert Kinect joint data frames to Labanotation
    #
    @abstractmethod
    def convertToLabanotation(self, axes, jointD, forceReset):
        pass


    # -----------------------------------------------------------------------------
    # React to mouse button down on canvas
    #
    def onCanvasClick(self, event):
        pass


    # -----------------------------------------------------------------------------
    # React to mouse motion on canvas
    #
    def onCanvasMove(self, event):
        pass


    # -----------------------------------------------------------------------------
    # React to mouse button release on canvas
    #
    def onCanvasRelease(self, event):
        pass


    #------------------------------------------------------------------------------
    # Reset class state
    #
    @abstractmethod
    def reset(self):
        pass


