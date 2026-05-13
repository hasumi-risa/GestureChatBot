# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

from __future__ import absolute_import

import copy
import math
import types

from builtins import range
from collections import OrderedDict, namedtuple

import matplotlib.lines
import matplotlib.pyplot as pyplot
import matplotlib.ticker
import matplotlib.widgets as widgets

from matplotlib.backend_bases import MouseButton
from matplotlib.font_manager import FontProperties

import numpy as np
import numpy.linalg

import gui
import gui.dialog
import gui.messagebox
import gui.utilities as utils

from gui.coordconvert import CoordConvert
from gui.gaussdialog import GaussDialog
from gui.dialog import StandardButtonWidgetEx
from gui.uielement import ComboBox, ButtonWidgetEx, TextUIElement, UIElementEx, WidgetEx

import settings

from labanotation.algorithmbase import GaussParams





class Helpers:
    def createLabeledComboBox(fig, strLabel, strActive, mpstrstrIDLabel):
        # Create label text widget
        textuielement = gui.uielement.TextUIElement(fig, strLabel)

        # Create combobox element
        rgLabel = [strLabel for strID, strLabel in mpstrstrIDLabel.items()]
        iActive = 0
        iT = 0
        for strID in mpstrstrIDLabel:
            if strID == strActive:
                iActive = iT
                break;
            iT = iT + 1

        uiex = gui.uielement.ComboBox(fig, rgLabel, index_selected=iActive)


        return (uiex, textuielement)



    def createLabeledRadioButtons(fig, strLabel, strActive, mpstrstrIDLabel):
        # Create label text widget
        textuielement = gui.uielement.TextUIElement(fig, strLabel)
        self.addUIElement(textuielement)

        # Create radio buttons widget
        rgLabel = [strLabel for strID, strLabel in mpstrstrIDLabel.items()]
        iActive = 0
        iT = 0
        for strID in mpstrstrIDLabel:
            if strID == strActive:
                iActive = iT
                break;
            iT = iT + 1

        widgetex = gui.uielement.RadioButtonsWidgetEx(fig, rgLabel, active=iActive)
        self.addUIElement(widgetex)


        return (widgetex, textuielement)



    def layoutLabeledComboBox(pt, widgetexCB, textuielement, sizeMarginInterControl):
        textuielement.ptFigure = pt
        textuielement.commitRect() # TextUIElement changes shape when its position is changed, so we need to commit it's new layout now
        sizeFigure = textuielement.sizeFigure
        x = pt[0] + sizeFigure[0] + sizeMarginInterControl[0]
        y = pt[1]

        y = y + (sizeFigure[1] - widgetexCB.sizeFigure[1]) / 2
        widgetexCB.ptFigure = (x, y)
        x = x + widgetexCB.sizeFigure[0]
        y = y + widgetexCB.sizeFigure[1]

        return (x, y)



    def layoutLabeledRadioButtons(pt, widgetexRB, textuielement):
        textuielement.ptFigure = pt
        textuielement.commitRect() # TextUIElement changes shape when its position is changed, so we need to commit it's new layout now
        x = pt[0]
        y = pt[1] + textuielement.sizeFigure[1]

        widgetexRB.ptFigure = (x, y)
        y = y + widgetexRB.sizeFigure[1]

        return (x, y)




class DialogMinimaDetectionSettings(gui.dialog.StandardButtonDialog):
    class FilterControls:
        @property
        def gauss_params(self):
            return self._gauss_params


        @gauss_params.setter
        def gauss_params(self, params):
            self.set_params(params)


        def __init__(self, fig, label, gauss_params=GaussParams()):
            self.fig = fig
            self.strLabel = label

            self._gauss_params = gauss_params
            self._gaussdialog = None

            self.textuielementLabel = TextUIElement(fig, label + ':')
            self.textuielementValues = TextUIElement(fig, "Window_size = 0, Sigma = 0")
            self.buttonwidgetexChange = ButtonWidgetEx(fig, label="Change")
            self.buttonwidgetexChange.origin = 'topright' # Stick to right side of dialog
            self.buttonwidgetexChange.widget.on_clicked(self.onButtonChange_Clicked)

            self.set_params(gauss_params)



        def close(self):
            if self._gaussdialog is not None:
                self._gaussdialog.close()
                self._gaussdialog = None




        def layout(self, posFigure, sizeMarginInterControl):
            (x, y) = posFigure
            self.textuielementLabel.ptFigure = (x, y)
            y = y + self.textuielementLabel.sizeFigure[1] + sizeMarginInterControl[1]

            xT = x + CoordConvert.inchToFigure((0.1, 0), self.fig)[0]
            self.textuielementValues.ptFigure = (xT, y)
            dy = self.textuielementValues.sizeFigure[1]
            y = y + dy

            self.buttonwidgetexChange.ptFigure = (posFigure[0], y - (dy + self.buttonwidgetexChange.sizeFigure[1]) / 2) # Using the left-edge x for the right-edge is sloppy but sufficient for now--the destination rectangle should be passed in instead

            y = y + sizeMarginInterControl[1]

            return (x, y)



        def onButtonChange_Clicked(self, event):
            if self._gaussdialog is None:
                self._gaussdialog = GaussDialog(title=self.strLabel + ' Gaussian Filter Parameters')
                self._gaussdialog.on_event(self.onGaussDialog_Event)

            self._gaussdialog.gauss_params = self.gauss_params
            self._gaussdialog.show()



        def onGaussDialog_Event(self, event):
            if event.reason == 'Ok':
                self.set_params(self._gaussdialog.gauss_params)


        def set_params(self, gauss_params):
            self._gauss_params = gauss_params
            self.textuielementValues.text.set_text("Window size = " + str(gauss_params.window_size) + ", Sigma = " + str(gauss_params.sigma))






    # -----------------------------------------------------------------------------
    # Initialize object instance
    #
    def __init__(self, settings, centeronfig=None):
        super().__init__(title='Minima Detection Settings', buttons=('Ok', 'Cancel'), sizeInch=(3.5, 2), centeronfig=centeronfig)
        self.gaussdialog = GaussDialog(centeronfig=self.fig)
        self.dictSettings = settings



    # -----------------------------------------------------------------------------
    # Create the child elements
    #
    def createChildren(self):
        super().createChildren()

        self.gaussdialogCoarse = GaussDialog()
        self.gaussdialogFine = GaussDialog()


        # Create Gaussian filter settings
        self.textuielementFilter = TextUIElement(self.fig, "Scale Space Parameters:")
        self.addUIElement(self.textuielementFilter)

        self.filtercontrolsCoarse = self.createFilterControls("Wide", self.dictSettings['wide'])
        self.filtercontrolsFine = self.createFilterControls("Narrow", self.dictSettings['narrow'])

        # Layout children (all use 'top left' origin so we can set and forget)
        x = self._rectContent[0]
        y = 1.0 - self._rectContent[3]

        self.textuielementFilter.ptFigure = (x, y)
        y = y + self.textuielementFilter.sizeFigure[1] + self.sizeMarginInterControl[1]
        x = x + self.inchToFigure((0.2, 0))[0]

        (x, y) = self.filtercontrolsCoarse.layout((x, y), self.sizeMarginInterControl)
        y = y + self.sizeMarginInterControl[1]

        (x, y) = self.filtercontrolsFine.layout((x, y), self.sizeMarginInterControl)



    def createFilterControls(self, strLabel, gauss_params):
        filtercontrols = self.FilterControls(self.fig, strLabel)
        filtercontrols.set_params(gauss_params)
        self.addUIElement(filtercontrols.textuielementLabel)
        self.addUIElement(filtercontrols.textuielementValues)
        self.addUIElement(filtercontrols.buttonwidgetexChange)

        return (filtercontrols)



    def onCanvas_Close(self, event):
        self.filtercontrolsCoarse.close()
        self.filtercontrolsCoarse = None
        self.filtercontrolsFine.close()
        self.filtercontrolsFine = None
        super().onCanvas_Close(event);



    def onPreDialogComplete(self, reason):
        if reason == 'Ok':
            self.dictSettings['wide'] = self.filtercontrolsCoarse.gauss_params
            self.dictSettings['narrow'] = self.filtercontrolsFine.gauss_params

        return super().onPreDialogComplete(reason)






class DialogSupportSettings(gui.dialog.StandardButtonDialog):
    # Mapping from detection algorithm ID to label
    _mpstrstrIDLabelDetectionAlgorithm = OrderedDict([
        ('correlateddirection', 'Feet & Body Correlated Direction'),
        ('kneeangle', 'Knee Angle'),
        ('lowesty', 'Lowest Y'),
        ('lowestposition', 'Lowest Laban Position')])



    # -----------------------------------------------------------------------------
    # Initialize object instance
    #
    def __init__(self, settings, centeronfig=None):
        super().__init__(title='Support Column Settings', buttons=('Ok', 'Cancel'), sizeInch=(3.5, 2), centeronfig=centeronfig)
        self.dictSettings = settings



    # -----------------------------------------------------------------------------
    # Create the child elements
    #
    def createChildren(self):
        super().createChildren()

        # Support detection algorithm combo box
        self.comboboxDetectAlgorithm, self.textuielementDetectAlgorithm = Helpers.createLabeledComboBox(self.fig, 'Detection Algorithm:', self.dictSettings['algorithm'], self._mpstrstrIDLabelDetectionAlgorithm)
        self.addUIElement(self.comboboxDetectAlgorithm)
        self.addUIElement(self.textuielementDetectAlgorithm)

        # Continuous walking checkbutton
        self.widgetexCheckButtons = gui.uielement.CheckButtonsWidgetEx(self.fig, labels=["Target continuous walking"], actives=[self.dictSettings['fContinuousWalking']])
        # self.widgetexCheckButtons = gui.uielement.CheckButtonsWidgetEx(self.fig, labels=["Target continuous walking", "A"], actives=[self.dictSettings['fContinuousWalking'], True])
        self.addUIElement(self.widgetexCheckButtons)

        # Layout children (all use 'top left' origin so we can set and forget)
        x = self._rectContent[0]
        y = 1.0 - self._rectContent[3]

        y = Helpers.layoutLabeledComboBox((x, y), self.comboboxDetectAlgorithm, self.textuielementDetectAlgorithm, self.sizeMarginInterControl)[1]
        y = y + self.sizeMarginInterControl[1]

        self.widgetexCheckButtons.ptFigure = (x, y)


    def onPreDialogComplete(self, reason):
        if reason == 'Ok':
            iitemSelected = self.comboboxDetectAlgorithm.index_selected
            if iitemSelected is not None:
                self.dictSettings['algorithm'] = list(self._mpstrstrIDLabelDetectionAlgorithm.items())[iitemSelected][0]

            self.dictSettings['fContinuousWalking'] = self.widgetexCheckButtons.widget.get_status()[0]

        return super().onPreDialogComplete(reason)






class DialogLowerBodyOptions(gui.dialog.StandardButtonDialog):
    # Mapping from algorithm ID to radio button label
    _mpstrstrIDLabelAlgorithms = OrderedDict([
        ('velocity', 'Foot Velocity'),
        ('parallel', 'Parallel Energy'),
        ('naive', 'Naive')])

    # Mapping from algorithm ID to radio button label
    _mpstrstrIDLabelAlgMotionChange = OrderedDict([
        ('threshold', 'Speed threshold'),
        ('localminima', 'Scale Space speed local minima')])

    # Mapping from dataset ID to radio button label
    _mpstrstrIDLabelDataSet = OrderedDict([
        ('feet', 'Feet'),
        ('hands', 'Hands'),
        ('debug', 'Debug'),
        ('debug2', 'Debug 2')])

    # List of check buttons from self.dictOptions
    _rgcheckbuttoninfo = (
        ('fFilterPosition', 'Use Gaussian filter on position'),
        ('fFilterVelocity', 'Use Gaussian filter on speed'),
        ('fFilterMotionChange', 'Filter motion changes using position delta'),
        ('fCalc3DSpeed',    'Calculate speed using all three axes'),
        ('fMakeYRelative', 'Make lower body Y positions relative to spine joint'),
        )



    # -----------------------------------------------------------------------------
    # Initialize object instance
    #
    def __init__(self, dictOptions, centeronfig=None):
        super().__init__(title='Motion Analysis Options', buttons=('Ok', 'Cancel'), centeronfig=centeronfig)
        self.dictOptions = dictOptions
        self.dialogminima = None
        self.dialogsupport = None

        self._sizeInch = (3.5, 5.5) # Size of the dialog window in inches



    # -----------------------------------------------------------------------------
    # Create the child elements
    #
    def createChildren(self):
        super().createChildren()

        # Data set combo box
        self.comboboxDataSet, self.textuielementDataSet = self.createLabeledComboBox('Data Set:', self.dictOptions['dataset'], self._mpstrstrIDLabelDataSet)

        # Algorithm radio buttons
        self.widgetexRBAlgorithm, self.textuielementAlgorithm = self.createLabeledRadioButtons('Algorithm:', self.dictOptions['algorithm'], self._mpstrstrIDLabelAlgorithms)

        # Motion change algorithm radio buttons
        self.widgetexRBAlgMotionChange, self.textuielementAlgMotionChange = self.createLabeledRadioButtons('Motion Change Algorithm:', self.dictOptions['algMotionChange'], self._mpstrstrIDLabelAlgMotionChange)

        # Check boxes for various options
        rgfActive = []
        rgstrLabels = []
        for checkbuttoninfo in self._rgcheckbuttoninfo:
            rgfActive.append(self.dictOptions[checkbuttoninfo[0]] == True)
            rgstrLabels.append(checkbuttoninfo[1])
        self.widgetexCheckButtons = gui.uielement.CheckButtonsWidgetEx(self.fig, labels=rgstrLabels, actives=rgfActive)
        self.addUIElement(self.widgetexCheckButtons)

        # Minima detection parameters
        widgetexButtonMinimaSettings = StandardButtonWidgetEx(self.fig, "Minima Detection Settings")
        widgetexButtonMinimaSettings.sizeInch = (2, 0.25)
        widgetexButtonMinimaSettings.widget.on_clicked(self.onButtonMinimaSettings_Clicked)
        self.addUIElement(widgetexButtonMinimaSettings)

        # Support column options
        widgetexButtonSupportSettings = StandardButtonWidgetEx(self.fig, "Support Column Settings")
        widgetexButtonSupportSettings.sizeInch = (2, 0.25)
        widgetexButtonSupportSettings.widget.on_clicked(self.onButtonSupportSettings_Clicked)
        self.addUIElement(widgetexButtonSupportSettings)

        # Layout children (all use 'top left' origin so we can set and forget)
        x = self._rectContent[0]
        y = 1.0 - self._rectContent[3]

        y = self.layoutLabeledComboBox((x, y), self.comboboxDataSet, self.textuielementDataSet)[1]
        y = y + self.sizeMarginInterControl[1]

        y = self.layoutLabeledRadioButtons((x, y), self.widgetexRBAlgorithm, self.textuielementAlgorithm)[1]
        y = y + self.sizeMarginInterControl[1]

        y = self.layoutLabeledRadioButtons((x, y), self.widgetexRBAlgMotionChange, self.textuielementAlgMotionChange)[1]
        y = y + self.sizeMarginInterControl[1]

        self.widgetexCheckButtons.ptFigure = (x, y)
        y = y + self.widgetexCheckButtons.sizeFigure[1] + self.sizeMarginInterControl[1]

        widgetexButtonMinimaSettings.ptFigure = (x, y)
        y = y + widgetexButtonMinimaSettings.sizeFigure[1] + self.sizeMarginInterControl[1]

        widgetexButtonSupportSettings.ptFigure = (x, y)



    def createLabeledComboBox(self, strLabel, strActive, mpstrstrIDLabel):
        # Create label text widget
        textuielement = gui.uielement.TextUIElement(self.fig, strLabel)
        self.addUIElement(textuielement)

        # Create combobox element
        rgLabel = [strLabel for strID, strLabel in mpstrstrIDLabel.items()]
        iActive = 0
        iT = 0
        for strID in mpstrstrIDLabel:
            if strID == strActive:
                iActive = iT
                break;
            iT = iT + 1

        uiex = gui.uielement.ComboBox(self.fig, rgLabel, index_selected=iActive)
        self.addUIElement(uiex)


        return (uiex, textuielement)



    def createLabeledRadioButtons(self, strLabel, strActive, mpstrstrIDLabel):
        # Create label text widget
        textuielement = gui.uielement.TextUIElement(self.fig, strLabel)
        self.addUIElement(textuielement)

        # Create radio buttons widget
        rgLabel = [strLabel for strID, strLabel in mpstrstrIDLabel.items()]
        iActive = 0
        iT = 0
        for strID in mpstrstrIDLabel:
            if strID == strActive:
                iActive = iT
                break;
            iT = iT + 1

        widgetex = gui.uielement.RadioButtonsWidgetEx(self.fig, rgLabel, active=iActive)
        self.addUIElement(widgetex)


        return (widgetex, textuielement)



    def layoutLabeledComboBox(self, pt, widgetexCB, textuielement):
        textuielement.ptFigure = pt
        textuielement.commitRect() # TextUIElement changes shape when its position is changed, so we need to commit it's new layout now
        sizeFigure = textuielement.sizeFigure
        x = pt[0] + sizeFigure[0] + self.sizeMarginInterControl[0]
        y = pt[1]

        y = y + (sizeFigure[1] - widgetexCB.sizeFigure[1]) / 2
        widgetexCB.ptFigure = (x, y)
        x = x + widgetexCB.sizeFigure[0]
        y = y + widgetexCB.sizeFigure[1]

        return (x, y)



    def layoutLabeledRadioButtons(self, pt, widgetexRB, textuielement):
        textuielement.ptFigure = pt
        textuielement.commitRect() # TextUIElement changes shape when its position is changed, so we need to commit it's new layout now
        x = pt[0]
        y = pt[1] + textuielement.sizeFigure[1]

        widgetexRB.ptFigure = (x, y)
        y = y + widgetexRB.sizeFigure[1]

        return (x, y)



    def onButtonMinimaSettings_Clicked(self, event):
        if self.dialogminima is None:
            self.dialogminima = DialogMinimaDetectionSettings(self.dictOptions['minimadetection'], centeronfig=self.fig)
        self.dialogminima.show()



    def onButtonSupportSettings_Clicked(self, event):
        if self.dialogsupport is None:
            self.dialogsupport = DialogSupportSettings(self.dictOptions['support'], centeronfig=self.fig)
        self.dialogsupport.show()



    def onCanvas_Close(self, event):
        if self.dialogminima is not None:
            self.dialogminima.close()
            self.dialogminima = None
        super().onCanvas_Close(event)



    def onPreDialogComplete(self, reason):
        fOKToClose = True
        if reason == 'Ok':
            self.saveRadioButtonsValue(self.widgetexRBAlgorithm, 'algorithm', self._mpstrstrIDLabelAlgorithms)
            self.saveRadioButtonsValue(self.widgetexRBAlgMotionChange, 'algMotionChange', self._mpstrstrIDLabelAlgMotionChange)
            self.saveComboBoxValue(self.comboboxDataSet, 'dataset', self._mpstrstrIDLabelDataSet)

            rgfActives = self.widgetexCheckButtons.widget.get_status()
            for icheckbutton in range(0, len(self._rgcheckbuttoninfo)):
                checkbuttoninfo = self._rgcheckbuttoninfo[icheckbutton]
                self.dictOptions[checkbuttoninfo[0]] = rgfActives[icheckbutton]

            if self.dialogminima is not None:
                self.dictOptions['minimadetection'] = self.dialogminima.dictSettings

        return fOKToClose



    def saveComboBoxValue(self, uiexCB, strOption, mpstrstrIDLabel):
        iitemSelected = uiexCB.index_selected
        if iitemSelected is not None:
            self.dictOptions[strOption] = list(mpstrstrIDLabel.items())[iitemSelected][0]



    def saveRadioButtonsValue(self, widgetexRB, strOption, mpstrstrIDLabel):
        strSelected = widgetexRB.widget.value_selected
        for strID, strLabel in mpstrstrIDLabel.items():
            if strLabel == strSelected:
                self.dictOptions[strOption] = strID
                break






class GraphLowerBody:
    strTitleBase = 'Movement Analysis'

    # Selected time indicator style parameters
    alphaTimeSel = 0.5 # Alpha channel value of selected time indicator line
    linewidthTimeSel = 3.0 # Width of the selected time indicator line
    colorTimeSel = 'purple' # Color of selected time indicator

    # Key Frame indicator style parameters
    plotstyleKeyFrame = {
        'color': 'green', # Color of marker
        'linestyle': '', # Don't connect markers with a plot line
        'marker': '*', # Shape of marker
        'markersize': 14, # Size of marker
        }
    colorMovementSpan = 'wheat' # Color of time spans of active movement

    # Generic plot styles
    plotstyleRed = {
        'color': 'red', # Color of plot
        }
    plotstyleRedLight = {
        'color': 'lightpink', # Color of plot
        }
    plotstyleGreen = {
        'color': 'green', # Color of plot
        }
    plotstyleGreenLight = {
        'color': 'lightgreen', # Color of plot
        }
    plotstyleBlue = {
        'color': 'blue', # Color of plot
        }
    plotstyleBlueLight = {
        'color': 'lightblue', # Color of plot
        }

    # Position and velocity plot styles
    plotstylePositionLeft = {
        'color': 'lightblue', # Color of plot
        }
    plotstyleVelocityLeft = plotstyleBlue;
    plotstylePositionRight = {
        'color': 'lightpink', # Color of plot
        }
    plotstyleVelocityRight = plotstyleRed;

    plotstyleLeftNormal = plotstyleVelocityLeft
    plotstyleLeftLight = plotstylePositionLeft
    plotstyleRightNormal = plotstyleVelocityRight
    plotstyleRightLight = plotstylePositionRight

    # X, Y, Z plot styles
    plotstyleX = plotstyleRed;
    plotstyleY = plotstyleGreen;
    plotstyleZ = plotstyleBlue;
    plotstyleVelocityX = plotstyleRedLight;
    plotstyleVelocityY = plotstyleGreenLight;
    plotstyleVelocityZ = plotstyleBlueLight;

    # Local velocity minima indicator style parameters
    plotstyleMovementChange = {
        'color': 'black', # Color of marker
        'linestyle': '', # Don't connect markers with a plot line
        'marker': 'o', # Shape of marker
        'markersize': 8, # Size of marker
        }

    # Scale space style parameters
    colorScaleSpaceInflectionLeft = 'lime'
    colorScaleSpacePeakLeft = 'blue'
    colorScaleSpaceInflectionRight = 'orange'
    colorScaleSpacePeakRight = 'red'

    plotstyleScaleSpaceFilteredNarrowLeft = {
        'color': 'darkcyan', # Color of plot
        }
    plotstyleScaleSpaceFilteredWideLeft = {
        'color': 'cyan', # Color of plot
        }
    plotstyleScaleSpaceFilteredNarrowRight = {
        'color': 'darkred', # Color of plot
        }
    plotstyleScaleSpaceFilteredWideRight = {
        'color': 'lightsalmon', # Color of plot
        }

    # Debug indicators
    plotstyleDebug = {
        'color': 'red', # Color of marker
        'linestyle': '', # Don't connect markers with a plot line
        'marker': 'o', # Shape of marker
        'markersize': 20, # Size of marker
        }

    # View options
    rgviewBase = [
            ["Positions & Speeds", 'displayPositionsAndSpeeds', {}],
            ["Positions",          'displayPositions', {}],
            ["Speeds",             'displaySpeeds', {}],
            ["Body Velocity",      'displayBodyVelocity', {}],
            ["Joint Position Left",  'displayJointPositions', {"fShowRight":False}],
            ["Joint Position Right", 'displayJointPositions', {"fShowRight":True}],
        ]
    rgviewScaleSpace = [ # Only available if scale space minima detection is enabled
            ["Scale Space Minima Both",  'displayScaleSpace', {"fShowLeft":True,  "fShowRight":True}],
            ["Scale Space Minima Left",  'displayScaleSpace', {"fShowLeft":True,  "fShowRight":False}],
            ["Scale Space Minima Right", 'displayScaleSpace', {"fShowLeft":False, "fShowRight":True}],
        ]


    #------------------------------------------------------------------------------
    # Class instance initialization
    #
    def __init__(self):
        self.strTitle = self.strTitleBase # Prominent part of figure title
        self.strInputName = None # Input file name to show in figure title
        self.dictOptions = None # Algorithm's options

        self.rgax = [] # Array of axes we've create to plot data on the figure
        self.rguielement = [] # Array of UIElements on the figure
        self.uielementAxes = None # If not None, the UI element wrapping the main reference axes

        # Layout parameters
        self.marginInchAxes = (0.75, 0.5, 0.8, 0.625) # Axes margins (left, bottom, right, top) (inches)

        # Create figure
        valueRCParamsToolBarSav = matplotlib.rcParams['toolbar']
        matplotlib.rcParams['toolbar'] = 'toolbar2'
        self.fig = pyplot.figure() # Our figure
        matplotlib.rcParams['toolbar'] = valueRCParamsToolBarSav

        # Set UI state
        self.dialoglowerbodyoptions = None # Options dialog
        self.fInfoShowing = False # If true, the info message box is showing
        self.fLButtonIsDown = False # If true, the left mouse button is pressed
        self.line2dTimeSel = None # Selected time marker
        self.proportionaltimeSelected = None # If not None, relative position of selected time marker (0.0 - 1.0)

        # Configure canvas
        canvas = self.fig.canvas
        canvas.set_window_title(self.strTitle)
        self.fig.set_size_inches((settings.screen_cx * 0.65) / self.fig.dpi, (settings.screen_cy * 0.465) / self.fig.dpi)

        # Attach to mouse events
        canvas.mpl_connect('close_event', self.onCanvas_Close)
        canvas.mpl_connect('button_press_event', self.onCanvas_ButtonPress)
        canvas.mpl_connect('button_release_event', self.onCanvas_ButtonRelease)
        canvas.mpl_connect('motion_notify_event', self.onCanvas_MotionNotify)
        canvas.mpl_connect('resize_event', self.onCanvas_Resize)

        # Create info help button
        strBtnText = 'i'
        btnFontSize = 14
        widgetexBtnInfo = WidgetEx(self.fig, widgets.Button, strBtnText, color='blue', hovercolor='lightblue')
        widgetexBtnInfo.origin = 'topright'
        widgetexBtnInfo.ptInch = (0.05, 0.05)
        self.rguielement.append(widgetexBtnInfo)

        dxPx, dyPx, dyPxDescender = utils.getTextExtent2Px(self.fig, strBtnText, FontProperties(size=btnFontSize, weight='bold'))
        dxInch = CoordConvert.pxToInch((dxPx, 0), self.fig)[0]
        widgetexBtnInfo.sizeInch = (dxInch * 4.0, btnFontSize / 72.0 * 1.5)

        btn = widgetexBtnInfo.widget
        btn.label.set_color('w')
        btn.label.set_fontsize(btnFontSize)
        btn.label.set_fontweight('bold')
        btn.on_clicked(self.onBtnInfo_Clicked)

        # Create view drop-down
        xInch = 0.05
        yInch = 0.05
        dxInchIntercontrol = 0.075
        rgstrBase = [view[0] for view in self.rgviewBase]
        rgstrScaleSpace = [view[0] for view in self.rgviewScaleSpace]
        self.comboboxView = ComboBox(self.fig, items=rgstrBase + rgstrScaleSpace, index_selected=0) # Set all items so combo box autosizes to fit everything
        self.comboboxView.ptInch = (xInch, yInch)
        self.comboboxView.on_changed(self.onComboBox_Changed)
        self.rguielement.append(self.comboboxView)
        xInch += self.comboboxView.sizeInch[0] + dxInchIntercontrol
        self.comboboxView.items = rgstrBase # Reset combox to base selections


        # Create options button
        strBtnText = 'Options'
        btnFontSize = 8
        widgetexBtnOptions = ButtonWidgetEx(self.fig, strBtnText, color='lightgray', hovercolor='lightblue')
        self.rguielement.append(widgetexBtnOptions)

        widgetexBtnOptions.ptInch = (xInch, yInch)

        btn = widgetexBtnOptions.widget
        btn.label.set_fontsize(btnFontSize)
        btn.on_clicked(self.onBtnOptions_Clicked)

        # Configure the axes
        self.reset()



    # -----------------------------------------------------------------------------
    # Add key frame markers at the specified time positions
    #
    def addKeyFrameMarkers(self, ax, rgsec, rgsecMoving=None, label="Key Frame", plotstyle=None):
        if (ax is not None):
            # Plot keyframe markers
            cmarker = len(rgsec)
            if plotstyle is None:
                plotstyle = self.plotstyleKeyFrame
            ax.plot(rgsec, [0] * cmarker, label=label, **plotstyle)

            # Plot vertical lines at keyframes
            for i in range(len(rgsec)):
                ax.axvline(rgsec[i], 0, 1, color = plotstyle['color'])

            # Plot moving areas
            if rgsecMoving is not None:
                for pair in rgsecMoving:
                    x = pair[0]
                    x2 = pair[1]
                    ax.axvspan(x, x2, alpha=0.4, color=self.colorMovementSpan)



    # -----------------------------------------------------------------------------
    # Add a plot using the specified plot style
    #
    def addPlot(self, ax, *args, label, plotstyle, **kwargs):
        if ax is not None:
            ax.plot(*args, label=label, **plotstyle, **kwargs)



    # -----------------------------------------------------------------------------
    # Add the plots to display the inflection and peak scale space filtered data
    #
    def addPlotScaledSpace(self, strPrefix, ax, rgsecTimestamp, rgspeed, scaled_space, plotstyleVelocity, plotstyleFilteredWide, plotstyleFilteredNarrow, strColorMarkerInflection, strColorMarkerPeak):
        rgvInflection = scaled_space.inflection
        rgvFiltered = scaled_space.filtered
        rgvFilteredNarrow = scaled_space.rgvFilteredNarrow
        rgvFilteredWide = scaled_space.rgvFilteredWide
        rgvPeak = scaled_space.peak
        if (rgspeed is None) or (rgvInflection is None) or (rgvFiltered is None) or (rgvPeak is None):
            return

        self.addPlot(ax, rgsecTimestamp, rgspeed, label=strPrefix + " Y Speed", plotstyle = plotstyleVelocity)
        self.addPlot(ax, rgsecTimestamp, rgvFilteredWide, label=strPrefix + " Y Speed Wide Gaussian Filter", plotstyle = plotstyleFilteredWide)
        self.addPlot(ax, rgsecTimestamp, rgvFilteredNarrow, label=strPrefix + " Y Speed Narrow Gaussian Filter", plotstyle = plotstyleFilteredNarrow)

        rgx = [rgsecTimestamp[i] for i in rgvInflection]
        rgy = [rgvFilteredWide[i] for i in rgvInflection]
        self.addPlot(ax, rgx, rgy, label = strPrefix + " Velocity Inflection", plotstyle = self.plotstyleMovementChange, markerfacecolor=strColorMarkerInflection)

        rgx = [rgsecTimestamp[i] for i in rgvPeak]
        rgy = [rgvFilteredWide[i] for i in rgvPeak]
        self.addPlot(ax, rgx, rgy, label = strPrefix + " Velocity Peak", plotstyle = self.plotstyleMovementChange, markerfacecolor=strColorMarkerPeak)


    # -----------------------------------------------------------------------------
    # Add a plot to display the scale space filtered data
    #
    def addPlotScaleSpaceFiltered(self, strPrefix, ax, rgsecTimestamp, scaled_space, plotstyleVelocity):
        rgvFiltered = scaled_space.filtered
        rgvFilteredNarrow = scaled_space.rgvFilteredNarrow
        if (rgvFiltered is None) or (rgvFilteredNarrow is None):
            return

        rgx = [rgsecTimestamp[i] for i in rgvFiltered]
        rgy = [rgvFilteredNarrow[i] for i in rgvFiltered]

        for x in rgx:
            ax.axvline(x, 0, 1, color = plotstyleVelocity['color'])

        ax.plot(rgx, rgy, label = strPrefix + "  Velocity Filtered", marker='o', color=(0, 0, 0, 0.5), linestyle='', markersize=8, fillstyle='full', clip_on=False)


    # -----------------------------------------------------------------------------
    # Clear all existing plot axes
    #
    def clearAxes(self):
        self.cursor = None
        self.line2dTimeSel = None
        self.uielementAxes = None

        for ax in self.rgax:
            ax.clear()
            ax.remove()
            ax = None
        self.rgax = []

        self.fig.canvas.draw_idle() # We changed the figure so matplotlib needs to redraw it when we're done


    # -----------------------------------------------------------------------------
    # Create the primary axes for a new plot
    #
    def createPrimaryAxes(self):
        # Get position and size for axes (using top-left origin)
        xFig, yFig = CoordConvert.inchToFigure((self.marginInchAxes[0], self.marginInchAxes[3]), self.fig)
        dxFig, dyFig = CoordConvert.inchToFigure((self.marginInchAxes[2], self.marginInchAxes[1]), self.fig)
        dxFig = 1 - dxFig - xFig
        dyFig = 1 - dyFig - yFig
        self.uielementAxes = UIElementEx(self.fig, sizeFigure=(dxFig, dyFig))
        self.uielementAxes.ptFigure = (xFig, yFig)
        self.uielementAxes.sizeMode = 'edge edge'

        self.uielementAxes.recalcRect()
        self.uielementAxes.commitRect()

        if self.cursor is None:
            self.cursor = gui.Cursor(self.uielementAxes.ax, color='red', linewidth=0.5, useblit=True)
            self.cursor.on_hit_test(self.onCursor_HitTest)


        return self.uielementAxes.ax


    # -----------------------------------------------------------------------------
    # Change the graph to show the scale-space minima analysis data
    #
    def displayScaleSpace(self, fShowLeft, fShowRight):
        self.clearAxes()

        if self.data is None:
            return

        rgsecTimestamp = self.data.rgsecTimestamp
        if rgsecTimestamp is None:
            return

        axPosition = self.createPrimaryAxes()
        axPosition.set_xlim(0, rgsecTimestamp[-1])
        self.rgax.append(axPosition)

        axVelocity = axPosition.twinx() # Second joint velocity axes with a second Y axis and sharing the main X (time) axis
        self.rgax.append(axVelocity)

        xaxis = axPosition.get_xaxis()
        xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.2f}'))
        xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.2))
        xaxis.set_label_text('Time (seconds)')

        axPosition.set_yticks([])

        #if self.dictOptions['fCalc3DSpeed']:
        #    axPosition.get_yaxis().set_label_text('Change in Position (meters)')
        #else:
        #    axPosition.get_yaxis().set_label_text('Position (meters)')
        axVelocity.get_yaxis().set_label_text('Speed (meters per second)')

        # Set the legend
        rgartistLegend = []
        if fShowLeft:
            rgartistLegend.append(matplotlib.lines.Line2D([0], [0], color=self.plotstyleVelocityLeft['color'], label='Left Speed'))
        if fShowRight:
            rgartistLegend.append(matplotlib.lines.Line2D([0], [0], color=self.plotstyleVelocityRight['color'], label='Right Speed'))
        if fShowLeft:
            rgartistLegend.append(matplotlib.lines.Line2D([0], [0], color=self.plotstyleScaleSpaceFilteredWideLeft['color'], label='Left Gaussian Wide'))
        if fShowRight:
            rgartistLegend.append(matplotlib.lines.Line2D([0], [0], color=self.plotstyleScaleSpaceFilteredWideRight['color'], label='Right Gaussian Wide'))
        if fShowLeft:
            rgartistLegend.append(matplotlib.lines.Line2D([0], [0], color=self.plotstyleScaleSpaceFilteredNarrowLeft['color'], label='Left Gaussian Narrow'))
        if fShowRight:
            rgartistLegend.append(matplotlib.lines.Line2D([0], [0], color=self.plotstyleScaleSpaceFilteredNarrowRight['color'], label='Right Gaussian Narrow'))

        if fShowLeft and fShowRight:
            kwargsInflection = {
                        'markerfacecolor':self.colorScaleSpaceInflectionLeft,
                        'markerfacecoloralt':self.colorScaleSpaceInflectionRight,
                        'fillstyle':'left'}
            kwargsPeak = {
                        'markerfacecolor':self.colorScaleSpacePeakLeft,
                        'markerfacecoloralt':self.colorScaleSpacePeakRight,
                        'fillstyle':'left'}
        elif fShowLeft:
            kwargsInflection = {'markerfacecolor':self.colorScaleSpaceInflectionLeft}
            kwargsPeak = {'markerfacecolor':self.colorScaleSpacePeakLeft}
        elif fShowRight:
            kwargsInflection = {'markerfacecolor':self.colorScaleSpaceInflectionRight}
            kwargsPeak = {'markerfacecolor':self.colorScaleSpacePeakRight}
        else:
            kwargsInflection = None
            kwargsPeak = None

        if kwargsInflection is not None:
            rgartistLegend.append(matplotlib.lines.Line2D([0], [0], color='w', label='Inflection',
                        marker=self.plotstyleMovementChange['marker'],
                        markersize=self.plotstyleMovementChange['markersize'],
                        **kwargsInflection))
        if kwargsPeak is not None:
            rgartistLegend.append(matplotlib.lines.Line2D([0], [0], color='w', label='Peak',
                        marker=self.plotstyleMovementChange['marker'],
                        markersize=self.plotstyleMovementChange['markersize'],
                        **kwargsPeak))

        rgartistLegend.append(matplotlib.lines.Line2D([0], [0], color='w', label='Filtered',
                    marker=self.plotstyleMovementChange['marker'],
                    markersize=self.plotstyleMovementChange['markersize'],
                    markerfacecolor=self.plotstyleMovementChange['color']))
        axVelocity.legend(handles=rgartistLegend, bbox_to_anchor=(0, 1), loc=3, ncol=7)

        # Plot the scaled space data except for the final minima
        if fShowLeft:
            self.addPlotScaledSpace(
                "Left",
                axVelocity,
                rgsecTimestamp,
                [abs(v) for v in self.data.left.velocity],
                self.data.minimadetection.left.scaled_space,
                self.plotstyleVelocityLeft,
                self.plotstyleScaleSpaceFilteredWideLeft,
                self.plotstyleScaleSpaceFilteredNarrowLeft,
                self.colorScaleSpaceInflectionLeft,
                self.colorScaleSpacePeakLeft)

        if fShowRight:
            self.addPlotScaledSpace(
                "Right",
                axVelocity,
                rgsecTimestamp,
                [abs(v) for v in self.data.right.velocity],
                self.data.minimadetection.right.scaled_space,
                self.plotstyleVelocityRight,
                self.plotstyleScaleSpaceFilteredWideRight,
                self.plotstyleScaleSpaceFilteredNarrowRight,
                self.colorScaleSpaceInflectionRight,
                self.colorScaleSpacePeakRight)

        # Fix the Y axis limits
        axVelocity.set_ylim((0, axVelocity.get_ylim()[1]))

        # Plot the final minima
        if fShowLeft:
            self.addPlotScaleSpaceFiltered(
                "Left",
                axVelocity,
                rgsecTimestamp,
                self.data.minimadetection.left.scaled_space,
                self.plotstyleVelocityLeft)

        if fShowRight:
            self.addPlotScaleSpaceFiltered(
                "Right",
                axVelocity,
                rgsecTimestamp,
                self.data.minimadetection.right.scaled_space,
                self.plotstyleVelocityRight)


    # -----------------------------------------------------------------------------
    # Change the graph to show the position and speed plots
    #
    def displayJointPositions(self, fShowRight):
        self.clearAxes()

        if self.data is None:
            return

        rgsec = self.data.rgsecTimestamp

        axPosition = self.createPrimaryAxes()
        axPosition.set_xlim(0, rgsec[-1])
        axPosition.get_yaxis().set_label_text('Position (meters)')
        self.rgax.append(axPosition)

        axVelocity = axPosition.twinx() # Second joint velocity axes with a second Y axis and sharing the main X (time) axis
        axVelocity.get_yaxis().set_label_text('Speed (meters per second)')
        self.rgax.append(axVelocity)

        xaxis = axPosition.get_xaxis()
        xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.2f}'))
        xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.2))
        xaxis.set_label_text('Time (seconds)')

        # Add a legend
        rgartistLegend = [
                matplotlib.lines.Line2D([0], [0], color=self.plotstyleX['color'], label='X Position'),
                matplotlib.lines.Line2D([0], [0], color=self.plotstyleY['color'], label='Y Position'),
                matplotlib.lines.Line2D([0], [0], color=self.plotstyleZ['color'], label='Z Position'),
                matplotlib.lines.Line2D([0], [0], color=self.plotstyleVelocityX['color'], label='X Velocity'),
                matplotlib.lines.Line2D([0], [0], color=self.plotstyleVelocityY['color'], label='Y Velocity'),
                matplotlib.lines.Line2D([0], [0], color=self.plotstyleVelocityZ['color'], label='Z Velocity'),
                matplotlib.lines.Line2D([0], [0], color='w', label='Key Frame',
                    marker=self.plotstyleKeyFrame['marker'],
                    markerfacecolor=self.plotstyleKeyFrame['color'],
                    markersize=self.plotstyleKeyFrame['markersize']),
                ]
        axVelocity.legend(handles=rgartistLegend, bbox_to_anchor=(0, 1), loc=3, ncol=7)

        rgsec = self.data.rgsecTimestamp
        rgxyz = self.data.right.rgxyz if fShowRight else self.data.left.rgxyz

        # Plot positions
        self.addPlot(axPosition, rgsec, [rgxyz[i][0] for i in range(0, len(self.data.left.rgxyz))], label = "X Position", plotstyle = self.plotstyleX)
        self.addPlot(axPosition, rgsec, [rgxyz[i][1] for i in range(0, len(self.data.left.rgxyz))], label = "Y Position", plotstyle = self.plotstyleY)
        self.addPlot(axPosition, rgsec, [rgxyz[i][2] for i in range(0, len(self.data.left.rgxyz))], label = "Z Position", plotstyle = self.plotstyleZ)

        # Plot speed
        rgspeed = [0] + [((rgxyz[i][0] - rgxyz[i - 1][0]) / (rgsec[i] - rgsec[i - 1]))  for i in range(1, len(rgxyz))]
        self.addPlot(axVelocity, rgsec, rgspeed, label="X Speed", plotstyle = self.plotstyleVelocityX)
        rgspeed = [0] + [((rgxyz[i][1] - rgxyz[i - 1][1]) / (rgsec[i] - rgsec[i - 1]))  for i in range(1, len(rgxyz))]
        self.addPlot(axVelocity, rgsec, rgspeed, label="X Speed", plotstyle = self.plotstyleVelocityY)
        rgspeed = [0] + [((rgxyz[i][2] - rgxyz[i - 1][2]) / (rgsec[i] - rgsec[i - 1]))  for i in range(1, len(rgxyz))]
        self.addPlot(axVelocity, rgsec, rgspeed, label="X Speed", plotstyle = self.plotstyleVelocityZ)

        ## Plot keyframe candidates
        #rgx = [rgsec[itime] for itime in self.data.left.rgiMovementChanges]
        #rgy = [rgspeedLeft[itime] for itime in self.data.left.rgiMovementChanges]
        #self.addPlot(axVelocity, rgx, rgy, label = "Left Velocity Minima", plotstyle = self.plotstyleMovementChange, markerfacecolor=self.plotstyleVelocityLeft['color'])

        #rgx = [rgsec[itime] for itime in self.data.right.rgiMovementChanges]
        #rgy = [rgspeedRight[itime] for itime in self.data.right.rgiMovementChanges]
        #self.addPlot(axVelocity, rgx, rgy, label = "Right Velocity Minima", plotstyle = self.plotstyleMovementChange, markerfacecolor=self.plotstyleVelocityRight['color'])

        if len(self.data.keyframe.rgsec) > 0:
            self.addKeyFrameMarkers(axVelocity, self.data.keyframe.rgsec, self.data.keyframe.rgpairsecFIsMoving)



    # -----------------------------------------------------------------------------
    # Change the graph to show the position and speed plots
    #
    def displayPositionsAndSpeeds(self):
        self.clearAxes()

        if self.data is None:
            return

        rgsec = self.data.rgsecTimestamp

        axPosition = self.createPrimaryAxes()
        axPosition.set_xlim(0, rgsec[-1])
        self.rgax.append(axPosition)

        axVelocity = axPosition.twinx() # Second joint velocity axes with a second Y axis and sharing the main X (time) axis
        axVelocity.get_yaxis().set_label_text('Speed (meters per second)')
        self.rgax.append(axVelocity)


        xaxis = axPosition.get_xaxis()
        xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.2f}'))
        xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.2))
        xaxis.set_label_text('Time (seconds)')

        # Add a legend
        rgartistLegend = [
                matplotlib.lines.Line2D([0], [0], color=self.plotstylePositionLeft['color'], label='Left Y Position'),
                matplotlib.lines.Line2D([0], [0], color=self.plotstylePositionRight['color'], label='Right Y Position'),
                matplotlib.lines.Line2D([0], [0], color=self.plotstyleVelocityLeft['color'], label='Left Y Velocity'),
                matplotlib.lines.Line2D([0], [0], color=self.plotstyleVelocityRight['color'], label='Right Y Velocity'),
                matplotlib.lines.Line2D([0], [0], color='w', label='Movement Changes',
                    marker=self.plotstyleMovementChange['marker'],
                    markerfacecolor=self.plotstyleVelocityLeft['color'],
                    markerfacecoloralt=self.plotstyleVelocityRight['color'],
                    markersize=self.plotstyleMovementChange['markersize'],
                    fillstyle='left'),
                matplotlib.lines.Line2D([0], [0], color='w', label='Key Frame',
                    marker=self.plotstyleKeyFrame['marker'],
                    markerfacecolor=self.plotstyleKeyFrame['color'],
                    markersize=self.plotstyleKeyFrame['markersize']),
                matplotlib.patches.Patch(label="Movement",
                    edgecolor=self.plotstyleKeyFrame['color'],
                    facecolor=self.colorMovementSpan,
                    fill=True),
                ]
        axVelocity.legend(handles=rgartistLegend, bbox_to_anchor=(0, 1), loc=3, ncol=7)

        # Plot position
        if self.dictOptions['fCalc3DSpeed']:
            axPosition.get_yaxis().set_label_text('Change in Position (meters)')
            self.addPlot(axPosition, rgsec, self.data.left.position, label = "Left Change in Position", plotstyle = self.plotstylePositionLeft)
            self.addPlot(axPosition, rgsec, self.data.right.position, label = "Right Change in Position", plotstyle = self.plotstylePositionRight)
        else:
            axPosition.get_yaxis().set_label_text('Y Position (meters)')
            self.addPlot(axPosition, rgsec, self.data.left.position, label = "Left Y Position", plotstyle = self.plotstylePositionLeft)
            self.addPlot(axPosition, rgsec, self.data.right.position, label = "Right Y Position", plotstyle = self.plotstylePositionRight)

        # Plot speed
        rgspeedLeft = [abs(v) for v in self.data.left.velocity]
        self.addPlot(axVelocity, rgsec, rgspeedLeft, label="Left Y Speed", plotstyle = self.plotstyleVelocityLeft)
        rgspeedRight = [abs(v) for v in self.data.right.velocity]
        self.addPlot(axVelocity, rgsec, rgspeedRight, label="Right Y Speed", plotstyle = self.plotstyleVelocityRight)

        # Plot keyframe candidates
        rgx = [rgsec[itime] for itime in self.data.left.rgiMovementChanges]
        rgy = [rgspeedLeft[itime] for itime in self.data.left.rgiMovementChanges]
        self.addPlot(axVelocity, rgx, rgy, label = "Left Velocity Minima", plotstyle = self.plotstyleMovementChange, markerfacecolor=self.plotstyleVelocityLeft['color'])

        rgx = [rgsec[itime] for itime in self.data.right.rgiMovementChanges]
        rgy = [rgspeedRight[itime] for itime in self.data.right.rgiMovementChanges]
        self.addPlot(axVelocity, rgx, rgy, label = "Right Velocity Minima", plotstyle = self.plotstyleMovementChange, markerfacecolor=self.plotstyleVelocityRight['color'])

        if len(self.data.keyframe.rgsec) > 0:
            self.addKeyFrameMarkers(axVelocity, self.data.keyframe.rgsec, self.data.keyframe.rgpairsecFIsMoving)



    # -----------------------------------------------------------------------------
    # Change the graph to show the position plots
    #
    def displayPositions(self):
        self.clearAxes()

        if self.data is None:
            return

        rgsec = self.data.rgsecTimestamp

        axPosition = self.createPrimaryAxes()
        axPosition.set_xlim(0, rgsec[-1])
        self.rgax.append(axPosition)

        xaxis = axPosition.get_xaxis()
        xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.2f}'))
        xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.2))
        xaxis.set_label_text('Time (seconds)')

        # Add a legend
        rgartistLegend = [
                matplotlib.lines.Line2D([0], [0], color=self.plotstyleVelocityLeft['color'], label='Left Y Position'),
                matplotlib.lines.Line2D([0], [0], color=self.plotstyleVelocityRight['color'], label='Right Y Position'),
                matplotlib.lines.Line2D([0], [0], color='w', label='Movement Changes',
                    marker=self.plotstyleMovementChange['marker'],
                    markerfacecolor=self.plotstyleVelocityLeft['color'],
                    markerfacecoloralt=self.plotstyleVelocityRight['color'],
                    markersize=self.plotstyleMovementChange['markersize'],
                    fillstyle='left'),
                matplotlib.lines.Line2D([0], [0], color='w', label='Key Frame',
                    marker=self.plotstyleKeyFrame['marker'],
                    markerfacecolor=self.plotstyleKeyFrame['color'],
                    markersize=self.plotstyleKeyFrame['markersize']),
                matplotlib.patches.Patch(label="Movement",
                    edgecolor=self.plotstyleKeyFrame['color'],
                    facecolor=self.colorMovementSpan,
                    fill=True),
                ]
        axPosition.legend(handles=rgartistLegend, bbox_to_anchor=(0, 1), loc=3, ncol=7)

        # Plot position
        if self.dictOptions['fCalc3DSpeed']:
            axPosition.get_yaxis().set_label_text('Change in Position (meters)')
            self.addPlot(axPosition, rgsec, self.data.left.position, label = "Left Change in Position", plotstyle = self.plotstyleVelocityLeft)
            self.addPlot(axPosition, rgsec, self.data.right.position, label = "Right Change in Position", plotstyle = self.plotstyleVelocityRight)
        else:
            axPosition.get_yaxis().set_label_text('Y Position (meters)')
            self.addPlot(axPosition, rgsec, self.data.left.position, label = "Left Y Position", plotstyle = self.plotstyleVelocityLeft)
            self.addPlot(axPosition, rgsec, self.data.right.position, label = "Right Y Position", plotstyle = self.plotstyleVelocityRight)

        # Plot keyframe candidates
        rgx = [rgsec[itime] for itime in self.data.left.rgiMovementChanges]
        rgy = [self.data.left.position[itime] for itime in self.data.left.rgiMovementChanges]
        self.addPlot(axPosition, rgx, rgy, label = "Left Velocity Minima", plotstyle = self.plotstyleMovementChange, markerfacecolor=self.plotstyleVelocityLeft['color'])

        rgx = [rgsec[itime] for itime in self.data.right.rgiMovementChanges]
        rgy = [self.data.right.position[itime] for itime in self.data.right.rgiMovementChanges]
        self.addPlot(axPosition, rgx, rgy, label = "Right Velocity Minima", plotstyle = self.plotstyleMovementChange, markerfacecolor=self.plotstyleVelocityRight['color'])

        if len(self.data.keyframe.rgsec) > 0:
            self.addKeyFrameMarkers(axPosition, self.data.keyframe.rgsec, self.data.keyframe.rgpairsecFIsMoving)



    # -----------------------------------------------------------------------------
    # Change the graph to show the speed plots
    #
    def displaySpeeds(self):
        self.clearAxes()

        if self.data is None:
            return

        rgsec = self.data.rgsecTimestamp

        axVelocity = self.createPrimaryAxes()
        self.rgax.append(axVelocity)

        axVelocity.set_xlim(0, rgsec[-1])

        xaxis = axVelocity.get_xaxis()
        xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.2f}'))
        xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.2))
        xaxis.set_label_text('Time (seconds)')

        yaxis = axVelocity.get_yaxis()
        yaxis.tick_right()
        yaxis.set_label_position("right")
        yaxis.set_label_text('Speed (meters per second)')

        # Add a legend
        rgartistLegend = [
                matplotlib.lines.Line2D([0], [0], color=self.plotstyleVelocityLeft['color'], label='Left Y Velocity'),
                matplotlib.lines.Line2D([0], [0], color=self.plotstyleVelocityRight['color'], label='Right Y Velocity'),
                matplotlib.lines.Line2D([0], [0], color='w', label='Movement Changes',
                    marker=self.plotstyleMovementChange['marker'],
                    markerfacecolor=self.plotstyleVelocityLeft['color'],
                    markerfacecoloralt=self.plotstyleVelocityRight['color'],
                    markersize=self.plotstyleMovementChange['markersize'],
                    fillstyle='left'),
                matplotlib.lines.Line2D([0], [0], color='w', label='Key Frame',
                    marker=self.plotstyleKeyFrame['marker'],
                    markerfacecolor=self.plotstyleKeyFrame['color'],
                    markersize=self.plotstyleKeyFrame['markersize']),
                matplotlib.patches.Patch(label="Movement",
                    edgecolor=self.plotstyleKeyFrame['color'],
                    facecolor=self.colorMovementSpan,
                    fill=True),
                ]
        axVelocity.legend(handles=rgartistLegend, bbox_to_anchor=(0, 1), loc=3, ncol=7)

        # Plot speed
        rgspeedLeft = [abs(v) for v in self.data.left.velocity]
        self.addPlot(axVelocity, rgsec, rgspeedLeft, label="Left Y Speed", plotstyle = self.plotstyleVelocityLeft)
        rgspeedRight = [abs(v) for v in self.data.right.velocity]
        self.addPlot(axVelocity, rgsec, rgspeedRight, label="Right Y Speed", plotstyle = self.plotstyleVelocityRight)

        # Plot keyframe candidates
        rgx = [rgsec[itime] for itime in self.data.left.rgiMovementChanges]
        rgy = [rgspeedLeft[itime] for itime in self.data.left.rgiMovementChanges]
        self.addPlot(axVelocity, rgx, rgy, label = "Left Velocity Minima", plotstyle = self.plotstyleMovementChange, markerfacecolor=self.plotstyleVelocityLeft['color'])

        rgx = [rgsec[itime] for itime in self.data.right.rgiMovementChanges]
        rgy = [rgspeedRight[itime] for itime in self.data.right.rgiMovementChanges]
        self.addPlot(axVelocity, rgx, rgy, label = "Right Velocity Minima", plotstyle = self.plotstyleMovementChange, markerfacecolor=self.plotstyleVelocityRight['color'])

        if len(self.data.keyframe.rgsec) > 0:
            self.addKeyFrameMarkers(axVelocity, self.data.keyframe.rgsec, self.data.keyframe.rgpairsecFIsMoving)



    # -----------------------------------------------------------------------------
    # Change the graph to show the speed plots
    #
    def displayBodyVelocity(self):
        self.clearAxes()

        if self.data is None:
            return

        rgsec = self.data.rgsecTimestamp

        axVelocity = self.createPrimaryAxes()
        self.rgax.append(axVelocity)

        axVelocity.set_xlim(0, rgsec[-1])

        xaxis = axVelocity.get_xaxis()
        xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.2f}'))
        xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.2))
        xaxis.set_label_text('Time (seconds)')

        yaxis = axVelocity.get_yaxis()
        yaxis.tick_right()
        yaxis.set_label_position("right")
        yaxis.set_label_text('Speed (meters per second)')

        # Plot speed
        rgB = [math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) for v in self.data.rgvecVelocityBody]
        axVelocity.plot(rgsec, rgB, label="Body Speed", color='Black')

        rgY = [math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) for v in self.data.left.rgvec]
        self.addPlot(axVelocity, rgsec, rgY, label="Left Foot Speed", plotstyle=self.plotstyleLeftLight)

        rgD = [np.dot(np.linalg.norm(self.data.rgvecVelocityBody[i]), np.linalg.norm(self.data.left.rgvec[i])) for i in range(0, len(self.data.rgvecVelocityBody))]
        self.addPlot(axVelocity, rgsec, rgD, label="Left Correlation", plotstyle=self.plotstyleLeftNormal)

        rgY = [math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) for v in self.data.right.rgvec]
        self.addPlot(axVelocity, rgsec, rgY, label="Right Foot Speed", plotstyle=self.plotstyleRightLight)

        rgD = [np.dot(np.linalg.norm(self.data.rgvecVelocityBody[i]), np.linalg.norm(self.data.right.rgvec[i])) for i in range(0, len(self.data.rgvecVelocityBody))]
        self.addPlot(axVelocity, rgsec, rgD, label="Right Correlation", plotstyle=self.plotstyleRightNormal)


    #------------------------------------------------------------------------------
    # Display the specified graph view
    #
    def doDisplay(self, strDisplay):
        # Get the matching view entry, if any
        view = None
        for viewT in self.rgviewBase:
            if strDisplay == viewT[0]:
                view = viewT
                break
        if view is None:
            for viewT in self.rgviewScaleSpace:
                if strDisplay == viewT[0]:
                    view = viewT
                    break
        if view is None:
            raise ValueError('Unrecognized display name "{0}"'.format(strDisplay))

        # Invoke the method that creates the view
        fnViewMethod = getattr(self, view[1])
        fnViewMethod(**view[2])

        # Draw the selected time indicator and draw everything
        self.synchronizeSelectedTime()
        self.fig.canvas.draw_idle()


    # -----------------------------------------------------------------------------
    # Get the primary axes mainly for reference purposes
    #
    def getReferenceAxes(self):
        if self.uielementAxes is not None:
            return self.uielementAxes.ax
        elif self.rgax:
            return self.rgax[0]

        return None



    # -----------------------------------------------------------------------------
    # React to clicked event on info button
    #
    def onBtnInfo_Clicked(self, event):
        if not self.fInfoShowing:
            self.fInfoShowing = True
            gui.messagebox.showinfo('Lower Body Data', 'This window displays the data involved in generating the Labanotation script for the lower body.', centeronfig = self.fig)
            self.fInfoShowing = False


    # -----------------------------------------------------------------------------
    # React to clicked event on options button
    #
    def onBtnOptions_Clicked(self, event):
        if (self.dialoglowerbodyoptions is None):
            self.dialoglowerbodyoptions = DialogLowerBodyOptions(self.dictOptions, centeronfig=self.fig)
            self.dialoglowerbodyoptions.on_event(self.onDialogLowerBodyOptions_Event)
        self.dialoglowerbodyoptions.show()



    #------------------------------------------------------------------------------
    # React to a button press event on the graph canvas
    #
    def onCanvas_ButtonPress(self, event):
        if (event.xdata is None) or (event.ydata is None):
            return

        # Ignore button presses when a toolbar mode is on
        if (self.fig.canvas.toolbar is not None) and (self.fig.canvas.toolbar.mode is not None) and (self.fig.canvas.toolbar.mode != ''):
            return

        # Left mouse button click?
        if (event.button == MouseButton.LEFT) and (event.inaxes in self.rgax):
            if event.dblclick:
                pass
            elif self.data is not None:
                self.fLButtonIsDown = True

                # Map mouse X-position to [0..1] relative to X (time) axis
                ax = self.getReferenceAxes()
                proportionTime = event.xdata / self.data.rgsecTimestamp[-1]
                # print("event.xdata = ", event.xdata, ", proportionTime = ", proportionTime)

                # Call application so that other graphs can be updated as well
                settings.application.selectTime(proportionTime)



    #------------------------------------------------------------------------------
    # React to a button release event on the graph canvas
    #
    def onCanvas_ButtonRelease(self, event):
        if (event.button == MouseButton.LEFT) and self.fLButtonIsDown:
            self.fLButtonIsDown = False



    # -----------------------------------------------------------------------------
    # React to canvas close event
    #
    def onCanvas_Close(self, event):
        self.fig = None
        # if user closes this figure, let the main application know and to exit
        settings.application.close()



    #------------------------------------------------------------------------------
    # React to a mouse move event on the graph canvas
    #
    def onCanvas_MotionNotify(self, event):
        if (not self.fLButtonIsDown or event.xdata is None):
            return

        if self.fLButtonIsDown:
            ax = self.getReferenceAxes()
            if ax is not None:
                # Map mouse X position to [0..1] relative to X (time) axis
                [xLow, xHigh] = ax.get_xlim()
                proportionTime = event.xdata / (xHigh - xLow)

                # Call application so that other graphs can be updated as well
                settings.application.selectTime(proportionTime)



    #------------------------------------------------------------------------------
    # React to the graph canvas resize event
    #
    def onCanvas_Resize(self, event):
        # Update the UI element rectangles
        if self.uielementAxes is not None:
            self.uielementAxes.recalcRect()

        for uielement in self.rguielement:
            uielement.recalcRect()

        # Update the UI element axes
        if self.uielementAxes is not None:
            self.uielementAxes.commitRect()

        for uielement in self.rguielement:
            uielement.commitRect()



    #------------------------------------------------------------------------------
    # React to the view combobox changing value
    #
    def onComboBox_Changed(self, event):
        self.doDisplay(event.label)


    def onCursor_HitTest(self, hittestevent):
        if (hittestevent.event.inaxes in self.rgax):
            hittestevent.is_hit = True


    #------------------------------------------------------------------------------
    # React to the dialog event
    #
    def onDialogLowerBodyOptions_Event(self, event):
        if event.reason == 'Ok':
            self.dictOptions = self.dialoglowerbodyoptions.dictOptions

            pyplot.show(block=False) # Since applyAlgorithm can take a while, make sure the dialog window has closed first
            settings.application.applyAlgorithm()



    #------------------------------------------------------------------------------
    # Reset the graph to initial empty state
    #
    def reset(self):
        self.data = None
        self.proportionaltimeSelected = None
        self.clearAxes()
        self.updateFigureTitle()
        self.comboboxView.index_selected = 0



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
    # Set the selected time
    #
    def selectTime(self, proportionaltime):
        self.proportionaltimeSelected = proportionaltime
        ax = self.getReferenceAxes()
        if (ax is not None) and (self.data is not None):
            # Move selected time marker to new time position
            #secTime = proportionaltime * ax.get_xlim()[1]
            secTime = proportionaltime * self.data.rgsecTimestamp[-1]
            if (self.line2dTimeSel is None):
                self.line2dTimeSel = ax.axvline(secTime, 0, 1, alpha = self.alphaTimeSel, color = self.colorTimeSel, linewidth = self.linewidthTimeSel)
                ax.add_line(self.line2dTimeSel)
            else:
                self.line2dTimeSel.set_xdata([secTime, secTime])

            # Draw the updated marker
            self.fig.canvas.draw_idle()



    # -----------------------------------------------------------------------------
    # Sets the data from the algorithm that we'll display
    #
    def setData(self, data):
        self.data = data

        itemCur = self.comboboxView.item_selected

        rgstrView = [view[0] for view in self.rgviewBase]
        if (self.data.minimadetection.left.scaled_space is not None) and (self.data.minimadetection.left.scaled_space is not None):
            rgstrView += [view[0] for view in self.rgviewScaleSpace]

        try:
            index = rgstrView.index(itemCur)
        except:
            index = 0

        self.comboboxView.items = rgstrView
        self.comboboxView.index_selected = index
        self.doDisplay(rgstrView[index])



    # -----------------------------------------------------------------------------
    # Ensure the selected time marker is at the right position
    #
    def synchronizeSelectedTime(self):
        if self.proportionaltimeSelected is not None:
            self.selectTime(self.proportionaltimeSelected)



    # -----------------------------------------------------------------------------
    # Synchronize the figure title with the current state
    #
    def updateFigureTitle(self):
        str = self.strTitle
        if (self.dictOptions is not None):
            str = str + ' (' + self.dictOptions['dataset'] + ')'
        if self.strInputName is not None:
            str = str + ' - [' + self.strInputName + ']'
        self.fig.canvas.set_window_title(str)


    # -----------------------------------------------------------------------------
    # Handle a change to the input data file name
    #
    def updateInputName(self):
        self.strInputName = settings.application.strBeautifiedInputFile
        self.updateFigureTitle()
