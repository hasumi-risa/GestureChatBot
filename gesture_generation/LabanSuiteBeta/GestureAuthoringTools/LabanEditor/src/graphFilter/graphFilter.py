# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import os, math, copy
from collections import OrderedDict
from decimal import Decimal

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from matplotlib.backend_tools import ToolBase, ToolToggleBase
from matplotlib.font_manager import FontProperties
from matplotlib.widgets import Slider, Cursor, Button, RadioButtons

import gui.messagebox
import settings
from gui import uielement
from gui import utilities as utils
from gui.coordconvert import CoordConvert

class graphFilter:
    fig = None
    ax = None

    #------------------------------------------------------------------------------
    # Class initialization
    #
    def __init__(self):
        self.strTitle = 'Filter Graph/Key Frame Editor'

        valueRCParamsToolBarSav = plt.rcParams['toolbar']
        plt.rcParams['toolbar'] = 'toolbar2'
        self.fig, self.ax = plt.subplots()
        plt.rcParams['toolbar'] = valueRCParamsToolBarSav

        self.fig.canvas.set_window_title(self.strTitle)
        self.fig.set_size_inches((settings.screen_cx * 0.65) / self.fig.dpi, (settings.screen_cy * 0.465) / self.fig.dpi)

        self.fig.canvas.mpl_connect('resize_event', self.onresize)
        self.fig.canvas.mpl_connect('close_event', self.onclose)

        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=0.5)

        # default is (0, 0, 1, 1) - [left, bottom, right, top]
        plt.tight_layout(rect=[0.02, 0.03, 0.97, 0.95])

        # create info help button
        strBtnText = 'i'
        btnFontSize = 14
        dxInch, dyInch = CoordConvert.pxToInch(utils.getTextExtentPx(self.fig, strBtnText, FontProperties(size=btnFontSize, weight='bold')), self.fig)
        dxInch = dxInch * 4.0
        dyInch = dyInch * 1.5

        self.widgetexInfo = uielement.WidgetEx(self.fig, Button, strBtnText, color='blue', hovercolor='lightblue')
        self.widgetexInfo.sizeInch = (dxInch, dyInch)
        btn = self.widgetexInfo.widget
        btn.label.set_color('w')
        btn.label.set_fontsize(btnFontSize)
        btn.label.set_fontweight('bold')
        btn.on_clicked(self.onclickInfoButton)

    # -----------------------------------------------------------------------------
    # canvas close event
    #
    def onclose(self, event):
        self.fig = None
        # if user closes this figure, let the main application know and to exit
        settings.application.close()

    #------------------------------------------------------------------------------
    # canvas resize event
    #
    def onresize(self, event):
        # Position the info button
        self.widgetexInfo.recalcRect()

        (xLeft, yBot, xRight, yTop) = utils.getAxesRect(self.ax)
        dyMargin = CoordConvert.inchToFigure((0, 0.05), self.fig)[1]
        self.widgetexInfo.ptFigure = (xRight - self.widgetexInfo.sizeFigure[0], yTop + dyMargin)

        self.widgetexInfo.commitRect()

    # -----------------------------------------------------------------------------
    #
    def onclickInfoButton(self, event):
        gui.messagebox.showinfo(
            "Labanotation",
            "Key Frame (green star icon) Editing Instructions:\n\n\n"
            "  - Right-click and drag on key frame star to *MOVE* along graph line.\n\n"
            "  - Right-click on graph line to *CREATE* a new key frame star.\n\n"
            "  - Double right-click to *DELETE* key frame star.\n\n",
           centeronfig=self.fig)

    # -----------------------------------------------------------------------------
    #
    def updateInputName(self):
        self.fig.canvas.set_window_title(self.strTitle + ' - [' + settings.application.strBeautifiedInputFile + ']')

    #------------------------------------------------------------------------------
    #
    def saveView(self):
        if (self.fig is None):
            return

        filePath = os.path.join(settings.application.outputFolder, settings.application.outputName + '_FilterGraph.png')
        filePath = settings.checkFileAlreadyExists(filePath, fileExt=".png", fileTypes=[('png files', '.png'), ('all files', '.*')])
        if (filePath is None):
            return

        # hide info button so it won't appear in image. Force an immediate redraw
        self.button_ax.set_visible(False)
        self.fig.canvas.draw()

        try:
            self.fig.savefig(filePath, bbox_inches='tight')
            settings.application.logMessage("Filter Graph view was saved to '" + settings.beautifyPath(filePath) + "'")
        except Exception as e:
            strError = e
            settings.application.logMessage("Exception saving Filter Graph view to '" + settings.beautifyPath(filePath) + "': " + str(e))


        # show info button.
        self.button_ax.set_visible(True)
        self.fig.canvas.draw_idle()

    #------------------------------------------------------------------------------
    #
    def selectTime(self, time):
        if (settings.application.labanotation != None):
            settings.application.labanotation.selectTime(time)
            self.fig.canvas.draw_idle()

    #------------------------------------------------------------------------------
    #
