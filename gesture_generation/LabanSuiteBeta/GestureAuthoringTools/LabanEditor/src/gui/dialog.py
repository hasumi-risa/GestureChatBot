# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

from __future__ import absolute_import

import math
import sys

import matplotlib as plt
import matplotlib.pyplot as pyplot
import matplotlib.widgets as widgets

from matplotlib.font_manager import FontProperties

from . import utilities as utils
from .coordconvert import CoordConvert
from .uielement import WidgetEx
from .window import Window



# -----------------------------------------------------------------------------
# Class DialogEvent
#
# Event parameters to Dialog.on_event event.
#
# Fields:
#   reason  The reason for the dialog event.  The standard dialog returns the
#               string of the dialog action button pressed ('OK', 'Cancel', etc.)
#
class DialogEvent:
    def __init__(self):
        self.reason = None # The reason for the dialog event



# -----------------------------------------------------------------------------
# Class Dialog
#
# Base class for standard UI dialogs.  Contains logic for geometry, handling
# UIElement objects, dialog completion, and hooks for subclasses.  Does not
# handle content, except for specifying the default rectangle for content,
# which is left entirely to subclasses like StandardButtonDialog.
#
class Dialog:
    @property
    def fIsShown(self):
        "Dialog is shown property"
        return (self.__fIsShown and (self.fig is not None))


    @property
    def geometry(self):
        rc = (0, 0, 0, 0)
        if (self._window is not None):
            rc = self._window.geometry
        return rc

    @geometry.setter
    def geometry(self, geometry):
        if (self._window is not None):
            self._window.geometry = geometry
            dpi = self.fig.dpi
        else:
            dpi = plt.rcParams['figure.dpi']

        self._geometry = geometry
        self._sizeInch = (geometry[2] / dpi, geometry[3] / dpi)


    @property
    def sizeInch(self):
        return self._sizeInch


    @sizeInch.setter
    def sizeInch(self, sizeInch):
        self._sizeInch = sizeInch
        if (self._window is not None):
            sizePx = self.fig.dpi_scale_trans.transform(sizeInch)
            geometry = self.geometry
            self.geometry = (geometry[0], geometry[1], sizePx[0], sizePx[1])


    @property
    def title(self):
        return (self._strTitle)

    @title.setter
    def title(self, str):
        self._strTitle = str


    @property
    def centeronfig(self):
        return (self._figCenterOn)


    @centeronfig.setter
    def centeronfig(self, figure):
        self._figCenterOn = figure
        if self.fIsShown:
            self.centerOnTarget()



    def __init__(self, title='Dialog', sizeInch=(4, 3), centeronfig=None):
        # Layout parameters
        self.sizeMarginContent =  0 # Margin between the edges of the dialog and the dialog content (figure units)
        self.sizeMarginInterControl =  0 # Margin between controls (figure units)

        # Data
        self.fig = None # The dialog's figure
        self.ax = None # The dialog's content axes

        # Protected Data
        self._callbackregistry = plt.cbook.CallbackRegistry() # Subscribers to our generated events
        self._dialogevent = DialogEvent() # Event to pass to self._fnCallbackEvent
        self._figCenterOn = centeronfig # If not None, center this dialog on this figure
        self._geometry = None # Initial window geometry
        self._rectContent = (0,0,1,1) # Dialog content goes into this rectangle (figure coordinates)
        self._rguielement = [] # Array of UIElements on this dialog
        self._sizeInch = sizeInch # Size of the dialog window in inches
        self._strTitle = title # The dialog window's title
        self._window = None # Helper for the figure's window

        # Private Data
        self.__fIsShown = False # If true, the dialog's figure is showing




    def addUIElement(self, uielement):
        if (uielement not in self._rguielement):
            self._rguielement.append(uielement)



    def centerOnFigure(self, figure = None):
        if figure is None:
            figure = self._figCenterOn
        if figure is not None:
            try:
                geomTarget = Window(figure).geometry
                self.centerOnPoint((geomTarget[0] + geomTarget[2] / 2, geomTarget[1] + geomTarget[3] / 2))
            except:
                pass



    def centerOnPoint(self, pt):
        try:
            windowUs = Window(self.fig)
            geomUs = windowUs.geometry

            xNew = pt[0] - geomUs[2] / 2
            yNew = pt[1] - geomUs[3] / 2

            windowUs.geometry = (xNew, yNew, geomUs[2], geomUs[3])
        except:
            pass



    def close(self):
        if (self.fig is not None):
            self.__fIsShown = False

            fig = self.fig
            self.fig = None
            pyplot.close(fig)



    def createChildren(self):
        pass



    def create(self):
        # Create a figure for the dialog and disable the toolbar
        toolbarSav = plt.rcParams['toolbar']
        plt.rcParams['toolbar'] = 'None'
        self.fig = pyplot.figure()
        plt.rcParams['toolbar'] = toolbarSav

        # Set the figure size and set the window title
        self.fig.set_size_inches(self._sizeInch)
        canvas = self.fig.canvas
        canvas.set_window_title(self._strTitle)
        self._window = Window(self.fig)

        # Set window geometry if we have one (note that if self._figCenterOn is not None we will set the position again later)
        if self._geometry is not None:
            self._window.geometry = self._geometry

        # Attach to canvas events
        canvas.mpl_connect('close_event', self.onCanvas_Close)
        canvas.mpl_connect('resize_event', self.onCanvas_Resize)

        # Create the axes for the dialog
        self.ax = self.fig.add_axes([0, 0, 1.0, 1.0], label='dialog')
        self.ax.set_axis_on()

        # Layout everything
        self.updateLayoutParams()
        self.layoutContentRect()
        self.createChildren()
        self.doLayout()



    def completeDialog(self, reason):
        if (self.fig is not None):
            reasonSav = self._dialogevent.reason
            self._dialogevent.reason = reason
            if (self.onPreDialogComplete(reason)):
                pyplot.close(self.fig)
            else:
                self._dialogevent.reason = reasonSav



    def doLayout(self):
        if (self._rguielement is not None):
            # Update UIElement's figure coordinates
            for uielement in self._rguielement:
                uielement.recalcRect()

            self.layoutContentRect()
            self.layoutChildren()

            # Update the UIElement's axes
            for uielement in self._rguielement:
                uielement.commitRect()



    def figureToInch(self, pairFigureValue):
        return (CoordConvert.figureToInch(pairFigureValue, self.fig))



    def figureToPx(self, pairFigureValue):
        return (CoordConvert.figureToPx(pairFigureValue, self.fig))



    def inchToFigure(self, pairInchValue):
        return (CoordConvert.inchToFigure(pairInchValue, self.fig))



    def inchToPx(self, pairInchValue):
        return (CoordConvert.inchToPx(pairInchValue, self.fig))



    def pxToFigure(self, pairPxValue):
        return (CoordConvert.pxToFigure(pairPxValue, self.fig))



    def pxToInch(self, pairFigureValue):
        return (CoordConvert.pxToInch(pairFigureValue, self.fig))



    def layoutChildren(self):
        pass



    def layoutContentRect(self):
        # Content covers the entire dialog minus the margins
        self._rectContent = (
            self.sizeMarginContent[0],
            self.sizeMarginContent[1],
            1.0 - 2.0 * self.sizeMarginContent[0],
            1.0 - 2.0 * self.sizeMarginContent[1])



    def on_event(self, fnCallback):
        return self._callbackregistry.connect('event', fnCallback)



    def onCanvas_Close(self, event):
        self.__fIsShown = False
        self._rguielement = []
        self._window = None
        self.ax = None
        self.fig = None

        # Notify callback about the dialog closure
        self._callbackregistry.process('event', self._dialogevent)



    def onCanvas_Resize(self, event):
        self._sizeInch = self.fig.get_size_inches()
        self.updateLayoutParams()
        self.doLayout()



    def onPreDialogComplete(self, reason):
        return (True)



    def removeUIElement(self, uielement):
        self._rguielement.remove(uielement)



    def show(self):
        self._dialogevent.reason = 0
        if (self.fig is None):
            self.create()

        if (self.fig is not None):
            self.fig.show()
            self.centerOnFigure()
            self.__fIsShown = True



    def updateLayoutParams(self):
        self.sizeMarginContent =  self.inchToFigure((0.1, 0.1))
        self.sizeMarginInterControl =  self.inchToFigure((0.1, 0.1))






# -----------------------------------------------------------------------------
# Class StandardButtonWidgetEx
#
# This class defines a WidgetEx-wrapped Button widget for standard dialog action
# buttons (like "OK" and "Cancel" for StandardButtonDialog, for instance) so
# that they all have a consistent style and size.
#
class StandardButtonWidgetEx(WidgetEx):
    sizeInchMarginText = (0.1, 0.06) # Margin between text and edge of a standard button
    fontsize = 9 # Font size of a standard button
    strColor = 'lightgray' # Button face color when button is idle
    strColorHover = 'lightblue' # Button face color when mouse is hovering in the button



    def __init__(self, figure, strText):
        super().__init__(figure, widgets.Button, label=strText, color=self.strColor, hovercolor=self.strColorHover)
        self.widget.label.set_fontsize(self.fontsize)

        # Set our size
        self.sizeInch = self.calcSizeInchOfTextButton(strText, self.fontsize)



    def calcSizeInchOfTextButton(self, str, fontsize):
        sizeInchText = self.pxToInch(utils.getTextExtentPx(self.ax.figure, str, FontProperties(size=fontsize)))
        return (sizeInchText[0] + 2 * self.sizeInchMarginText[0], sizeInchText[1] + 2 * self.sizeInchMarginText[1])


    def calcSizeInchOfTextButton2(self, str, fontsize):
        sizeInchText = self.pxToInch(utils.getTextExtent2Px(self.ax.figure, str, FontProperties(size=fontsize))[0:2])
        return (sizeInchText[0] + 2 * self.sizeInchMarginText[0], sizeInchText[1] + 2 * self.sizeInchMarginText[1])





# -----------------------------------------------------------------------------
# Class StandardButtonDialog
#
# This class defines a dialog that has a row of standard buttons at the button
# and content above the buttons.  The standard buttons use the same style but
# the number of buttons and their text are specified by the caller.
#
# The content is defined by subclasses (like MessageBox.)
#
class StandardButtonDialog(Dialog):
    sizeInchStdBtnMinDefault = (0.75, 0.25) # Dialog buttons should be at least this wide (inches)


    # Properties

    @property
    def buttons(self):
        "List of standard buttons shown on the dialog"
        return (self._rgstrStdBtn)


    @buttons.setter
    def buttons(self, rgstr):
        if (type(rgstr) is not list) and (type(rgstr) is not tuple):
            self._rgstrStdBtn = [rgstr]
        else:
            self._rgstrStdBtn = rgstr

        if (self.fIsShown):
            self.ensureStdBtnWidgets() # Update widgets for buttons that were changed
            self.doLayout() # Relayout the changed buttons
            self.fig.canvas.draw_idle() # Redraw the canvas to show the changed buttons



    def __init__(self, title=None, buttons=('OK'), buttonsnoclose=(), centeronfig=None, **kwargs):
        super().__init__(title=title, centeronfig=centeronfig, **kwargs)

        self.sizeInchStdBtnMin = None

        self._rgwidgetexStdBtn = [] # Widgets for the standard buttons in left-to-right order of appearance
        self._rgstrStdBtn = [] # The names of the standard buttons to show
        self._rectStdBtnArea = (0, 0, 0, 0) # Rectangle containing the standard buttons at the bottom of the dialog (figure coordinates)
        self._rgstrNoClose = buttonsnoclose # The names of the standard buttons that don't complete the dialog

        # Set list of standard buttons to show in left-to-right order of appearance
        if (type(buttons) is not list) and (type(buttons) is not tuple):
            self._rgstrStdBtn = [buttons]
        else:
            self._rgstrStdBtn = buttons



    def createChildren(self):
        super().createChildren()
        self.ensureStdBtnWidgets()
        self.layoutContentRect() # layoutContentRect() is called by Dialog before createChildren(), but our content rectangle is affected by the standard buttons we create

        # Position standard action buttons in a row starting at the lower-right corner of the button area
        xFigure = 1.0 - self._rectStdBtnArea[2] # Remember, the button origins are "bottom-right"
        yFigure = self._rectStdBtnArea[1]

        dyButtonArea = self._rectStdBtnArea[3] - self._rectStdBtnArea[1]
        for widgetex in reversed(self._rgwidgetexStdBtn): # We're laying out right-to-left
            sizeT = widgetex.sizeFigure
            yBtn = yFigure + (dyButtonArea - sizeT[1]) / 2.0 # Center button vertically with tallest button
            widgetex.ptFigure = (xFigure, yBtn)

            xFigure = xFigure + sizeT[0] + self.sizeMarginInterControl[0] # Update x position for next button (remember, the button origins are "bottom-right")



    def createStdBtnWidget(self, strStdBtn):
        widgetex = StandardButtonWidgetEx(self.fig, strStdBtn)
        dxInch, dyInch = widgetex.sizeInch
        dxInch = dxInch if (dxInch >= self.sizeInchStdBtnMin[0]) else self.sizeInchStdBtnMin[0]
        dyInch = dyInch if (dyInch >= self.sizeInchStdBtnMin[1]) else self.sizeInchStdBtnMin[1]
        widgetex.sizeInch = (dxInch, dyInch)

        return (widgetex)


    def ensureStdBtnWidgets(self):
        dictButtonKeep = {}
        rgwidgetexRemove = []

        # Create lists of existing button widgets to keep that are in the requested set and buttons to remove ones that are not
        for widgetex in self._rgwidgetexStdBtn:
            strLabel = widgetex.widget.label.get_text()
            if (strLabel in self._rgstrStdBtn):
                dictButtonKeep[strLabel] = widgetex
            else:
                rgwidgetexRemove.append(widgetex)

        # Remove the widgets no longer needed
        for widgetex in rgwidgetexRemove:
            self.removeUIElement(widgetex) # Remove widget from dialog
            widgetex.ax.remove() # Remove the widget's axes from the figure
        rgwidgetexRemove = None

        # Rebuild list of widgets, creating new ones as needed
        self._rgwidgetexStdBtn = []
        for str in self._rgstrStdBtn:
            widgetex = dictButtonKeep.get(str)
            if widgetex is not None:
                # Reuse existing button
                self._rgwidgetexStdBtn.append(widgetex)
            else:
                # Create a new standard button
                widgetex = self.createStdBtnWidget(str)
                widgetex.origin = 'bottomright' # Buttons are stacked against the bottom-right edge
                widgetex.widget.on_clicked(self.onStdBtn_Clicked) # Attach event handler
                self._rgwidgetexStdBtn.append(widgetex) # Remember the widget
                self.addUIElement(widgetex) # Add widget to the dialog



    def handleStdBtnClicked(self, strButton):
        if strButton not in self._rgstrNoClose:
            # Complete the dialog with this button as the completion code
            self.completeDialog(strButton)
        else:
            # Notify callback of the button press but don't complete the dialog
            self._dialogevent.reason = strButton
            if (self._fnCallbackEvent is not None):
                self._fnCallbackEvent(self._dialogevent)



    def layoutContentRect(self):
        super().layoutContentRect()

        # Determine the tallest standard button
        dyFigure = 0
        for widgetex in self._rgwidgetexStdBtn:
            if (widgetex is not None):
                dy = widgetex.sizeFigure[1]
                if (dy > dyFigure):
                    dyFigure = dy

        # Position button area at the bottom of the dialog
        (xLeftContent, yBotContent, xRightContent, yTopContent) = self._rectContent
        self._rectStdBtnArea = (xLeftContent, yBotContent, xRightContent, yBotContent + dyFigure)

        # Move content rect so it starts above the button area
        yBotContent = yBotContent + dyFigure + 2 * self.sizeMarginInterControl[1] # Add twice inter-control gap so content isn't placed right up against the buttons
        self._rectContent = (xLeftContent, yBotContent, xRightContent, yTopContent)



    def onCanvas_Close(self, event):
        self._rgwidgetexStdBtn = []
        super().onCanvas_Close(event)



    def onStdBtn_Clicked(self, event):
        # Identify the standard button and handle the click event
        for widgetex in self._rgwidgetexStdBtn:
            if (widgetex.ax == event.inaxes):
                self.handleStdBtnClicked(widgetex.widget.label.get_text())
                break


    def updateLayoutParams(self):
        super().updateLayoutParams()

        # If minimum button size isn't set yet, set it now
        if (self.sizeInchStdBtnMin is None):
            # Combine fixed minimum horizontal size with font-derived vertical size
            self.sizeInchStdBtnMin = [
                self.sizeInchStdBtnMinDefault[0],
                self.pxToInch(
                    utils.getTextExtent2Px(
                        self.fig,
                        "GgJjPpQqYy09",
                        fontproperties=FontProperties(size=StandardButtonWidgetEx.fontsize))[0:2]
                    )[1] + 2 * StandardButtonWidgetEx.sizeInchMarginText[1],
                ]
