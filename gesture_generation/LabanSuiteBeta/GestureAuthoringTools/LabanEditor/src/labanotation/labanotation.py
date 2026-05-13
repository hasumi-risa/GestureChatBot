# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function

import settings

from . import alglowerbody
from . import algnaive
from . import algtotal
from . import algparallel
from . import labanProcessor
from . import labanVisualization
from .algorithmbase import AlgorithmBase


class labanotation:
    algorithm = None
    algorithmLowerBody = None
    idEventC = None
    idEventR = None
    idEventM = None

    #------------------------------------------------------------------------------
    # Class initialization
    #
    def __init__(self):
        # print("Initializing Labanotation...")
        pass

    #------------------------------------------------------------------------------
    # convert joint data frames to labanotation using specified algorithm
    #
    def ensureAlgorithmObject(self, algorithm):
        if (self.algorithmLowerBody is None):
            self.algorithmLowerBody = alglowerbody.Algorithm()

        algorithm = algorithm.lower()
        if ((self.algorithm is None) or (self.algorithm.algorithm != algorithm)):
            if (algorithm == 'naive'):
                self.algorithm = algnaive.Algorithm(algorithm)
                self.setupButtonEvents();
            elif (algorithm == 'total'):
                self.algorithm = algtotal.Algorithm(algorithm)
                self.setupButtonEvents();
            elif (algorithm == 'parallel'):
                self.algorithm = algparallel.Algorithm(algorithm)
                self.setupButtonEvents();
            else:
                self.algorithm = None
                self.algorithmLowerBody = None
                # print("Internal Error: Unkown algorithm '" + algorithm + "'.")

    #------------------------------------------------------------------------------
    # convert joint data frames to labanotation using specified algorithm
    #
    def applyAlgorithm(self, ax, axLB, graphLowerBody, jointD, algorithm, forceReset = False):
        self.ensureAlgorithmObject(algorithm)

        if (self.algorithm is None):
            # print("Internal Error: Unkown algorithm '" + algorithm + "'.")
            return (None, None)

        [timeS, labanscript] = self.algorithm.convertToLabanotation(ax, jointD, forceReset)

        self.algorithmLowerBody.setGraph(graphLowerBody)
        [timeS, labanscript] = self.algorithmLowerBody.convertToLabanotation(jointD, timeS, labanscript, forceReset)

        return ([timeS, labanscript])

    #------------------------------------------------------------------------------
    #
    def getGaussianParameters(self, algorithm):
        self.ensureAlgorithmObject(algorithm)

        if (self.algorithm is None):
            # print("Internal Error: Unkown algorithm '" + algorithm + "'.")
            gauss_params = AlgorithmBase.get_gauss_params_default()
        else:
            gauss_params = self.algorithm.gauss_params

        return gauss_params

    #------------------------------------------------------------------------------
    #
    def setGaussianParameters(self, algorithm, gauss_params):
        self.ensureAlgorithmObject(algorithm)
        if (self.algorithm is None):
            return

        self.algorithm.gauss_params = gauss_params

        if (self.algorithmLowerBody is not None):
            self.algorithmLowerBody.gauss_params = gauss_params


    #------------------------------------------------------------------------------
    #
    def setupButtonEvents(self):
        if (settings.application.graphFilter is not None):
            canvas = settings.application.graphFilter.fig.canvas

            if (self.idEventC is not None):
                canvas.mpl_disconnect(self.idEventC)
                self.idEventC = None

            if (self.idEventR is not None):
                canvas.mpl_disconnect(self.idEventR)
                self.idEventR = None

            if (self.idEventM is not None):
                canvas.mpl_disconnect(self.idEventM)
                self.idEventM = None

            if (self.algorithm != None):
                self.idEventC = canvas.mpl_connect('button_press_event', self.algorithm.onCanvasClick)
                self.idEventR = canvas.mpl_connect('button_release_event', self.algorithm.onCanvasRelease)
                self.idEventM = canvas.mpl_connect('motion_notify_event', self.algorithm.onCanvasMove)

    #------------------------------------------------------------------------------
    #
    def labanToScript(self, timeS, all_laban):
        return labanProcessor.toScript(timeS, all_laban)

    #------------------------------------------------------------------------------
    #
    def labanScriptToImage(self, w, h, script):
        return labanVisualization.convertLabanScriptToView(w, h, script);

    #------------------------------------------------------------------------------
    #
    def saveToJSON(self):
        if (self.algorithmLowerBody != None):
            self.algorithmLowerBody.saveToJSON(self.algorithm.labandata)
        elif (self.algorithm != None):
            self.algorithm.saveToJSON()

    #------------------------------------------------------------------------------
    #
    def saveToTXT(self):
        if (self.algorithmLowerBody != None):
            self.algorithmLowerBody.saveToTXT(self.algorithm.timeS, self.algorithm.all_laban)
        elif (self.algorithm != None):
            self.algorithm.saveToTXT()

    #------------------------------------------------------------------------------
    #
    def selectTime(self, time):
        if (self.algorithm != None):
            self.algorithm.selectTime(time)

