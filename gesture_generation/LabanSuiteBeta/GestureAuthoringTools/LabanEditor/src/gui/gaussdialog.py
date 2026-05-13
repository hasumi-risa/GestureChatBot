# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

from __future__ import absolute_import

import matplotlib
import numpy as np

from matplotlib.widgets import Slider

import gui.dialog
import labanotation.tool.wavfilter as wf

from gui.uielement import UIElementEx, WidgetEx
from labanotation.algorithmbase import GaussParams



# -----------------------------------------------------------------------------
# Class GaussDialog
#
# Dialog for setting Gaussian filter parameters.
#
# Properties:
#   gauss_params  The filter parameters in a GaussParam object.  This object
#                     sets the initial values in the dialog and is updated only
#                     only if the user presses the "Ok" button.
#
class GaussDialog(gui.dialog.StandardButtonDialog):
    strColorSlider = 'lightgoldenrodyellow' # Face color of the slider controls



    #------------------------------------------------------------------------------
    # Class object initialization
    #
    # Arguments:
    #   title            Figure window title string
    #   gauss_params     Initial gaussian parameters
    #   windowsizerange  2-tuple with minimum and maximum window size values
    #   sigmarange       2-tuple with minimum and maximum sigma values
    #   sizeInch         Dimensions of the dialog window (inches)
    #   centeronfig      If not None, center the dialog on this figure
    #
    def __init__(self, title='Gaussian Filter Parameters', gauss_params=GaussParams(31, 5), windowsizerange=(1, 41), sigmarange=(1, 20), sizeInch=(4, 3), centeronfig=None, **kwargs):
        super().__init__(title, buttons=('Reset', 'Ok', 'Cancel'), sizeInch=sizeInch, centeronfig=centeronfig, **kwargs)

        self.gauss_params = gauss_params

        self.axGauss = None # Gaussian curve plot axes
        self.lineGaussian = None # Plot of Gaussian curve
        self.rangeSigma = sigmarange # Range of sigma values
        self.rangeWindowSize  = windowsizerange # Range of window sizes



    #------------------------------------------------------------------------------
    # Method createChildren()
    #
    # Create the child UI elements of the dialog
    #
    def createChildren(self):
        super().createChildren()

        dxContent, dyContent = (self._rectContent[2] - self._rectContent[0], self._rectContent[3] - self._rectContent[1])
        dxMarginSlider = self.inchToFigure((0.25, 0))[0]

        # Create axes for Gaussian distribution graph
        self.uielementexGauss = UIElementEx(self.fig)
        self.uielementexGauss.sizeMode = 'edge edge'
        self.addUIElement(self.uielementexGauss)
        self.axGauss = self.uielementexGauss.ax

        xGauss, yGauss = (self._rectContent[0] + self.inchToFigure((0.3, 0))[0], 1.0 - self._rectContent[3])
        self.uielementexGauss.ptFigure = (xGauss, yGauss)

        x, y = (self._rectContent[0], self._rectContent[1])
        x = x + self.inchToFigure((0.75, 0))[0]
        dyInterControl = self.sizeMarginInterControl[1]

        # Create Gaussian curve standard distribution slider
        self.widgetexSliderSigma = WidgetEx(self.fig, Slider, 'Sigma', self.rangeSigma[0], self.rangeSigma[1], valinit=self.gauss_params.sigma, valstep=1)
        self.widgetexSliderSigma.origin = 'bottomleft'
        self.widgetexSliderSigma.sizeMode = 'edge fixed'
        self.widgetexSliderSigma.ptFigure = (x + dxMarginSlider, y)
        self.widgetexSliderSigma.sizeFigure = (dxContent - 2 * dxMarginSlider - x, self.inchToFigure((0, 0.15))[1])
        self.widgetexSliderSigma.ax.set_facecolor(self.strColorSlider)
        self.widgetexSliderSigma.widget.on_changed(self.onSlider_Changed)
        self.addUIElement(self.widgetexSliderSigma)
        y = y + self.widgetexSliderSigma.sizeFigure[1] + dyInterControl

        # Create window size slider
        self.widgetexSliderWindowSize = WidgetEx(self.fig, Slider, 'Window Size', self.rangeWindowSize[0], self.rangeWindowSize[1], valinit=self.gauss_params.window_size, valstep=2)
        self.widgetexSliderWindowSize.origin = 'bottomleft'
        self.widgetexSliderWindowSize.sizeMode = 'edge fixed'
        self.widgetexSliderWindowSize.ptFigure = (x + dxMarginSlider, y)
        self.widgetexSliderWindowSize.sizeFigure = (dxContent - 2 * dxMarginSlider - x, self.inchToFigure((0, 0.15))[1])
        self.widgetexSliderWindowSize.ax.set_facecolor(self.strColorSlider)
        self.widgetexSliderWindowSize.widget.on_changed(self.onSlider_Changed)
        self.addUIElement(self.widgetexSliderWindowSize)
        y = y + self.widgetexSliderWindowSize.sizeFigure[1] + dyInterControl

        # Set graph size after subtracting space for the sliders
        self.uielementexGauss.sizeFigure = (self._rectContent[2] - xGauss, 1.0 - y - self.inchToFigure((0, 0.25))[1] - yGauss)

        # Display a graphical representation of the filter kernel
        self.updateKernelDisplay()



    #------------------------------------------------------------------------------
    # Method getParamsFromControls()
    #
    # Returns the current Gaussian filter parameters from the UI controls
    #
    def getParamsFromControls(self):
            return GaussParams(int(self.widgetexSliderWindowSize.widget.val), int(self.widgetexSliderSigma.widget.val))



    #------------------------------------------------------------------------------
    # Method onPreDialogComplete()
    #
    # Handle the dialog is being completed and approve or deny closing the dialog
    #
    def onPreDialogComplete(self, reason):
        fCanClose = True
        if reason == 'Ok':
            self.gauss_params = self.getParamsFromControls()
            fCanClose = super().onPreDialogComplete(reason)
        elif reason == 'Reset':
            self.widgetexSliderWindowSize.widget.reset()
            self.widgetexSliderSigma.widget.reset()
            fCanClose = False

        return fCanClose



    #------------------------------------------------------------------------------
    # Method onSlider_Changed()
    #
    # Event handler for a slider control's value being changed
    #
    def onSlider_Changed(self, event):
        self.updateKernelDisplay()



    #------------------------------------------------------------------------------
    # Method updateKernelDisplay()
    #
    # Update the visual representation of the Gaussian filter kernel
    #
    def updateKernelDisplay(self):
        # Remove current Gaussian plot lines
        if (self.lineGaussian is not None):
            self.lineGaussian.remove()
            self.lineGaussian = None

        # Create a plot of the filter kernel
        gauss_params = self.getParamsFromControls()

        rgx = np.arange(-gauss_params.window_size / 2.0 + 0.5, gauss_params.window_size / 2.0 + 0.5, 1.0)
        rgy = wf.gaussFilter(gauss_params.window_size, gauss_params.sigma)
        self.lineGaussian, = self.axGauss.plot(rgx, rgy, marker="o", linestyle='-', color='red', lw=1)

        # Set the axes to maximize the view of the curve
        radiusWindow = int(gauss_params.window_size / 2) + 1
        self.axGauss.set_xlim(-radiusWindow, radiusWindow)
        self.axGauss.set_ylim(0, 0.42) # np.max(s))

