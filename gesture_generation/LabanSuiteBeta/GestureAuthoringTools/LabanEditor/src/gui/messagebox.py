# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

from __future__ import absolute_import

from builtins import range

import math
import os
import sys

import matplotlib
import matplotlib.cbook as cbook
import matplotlib.pyplot as pyplot

from matplotlib.font_manager import FontProperties

from . import utilities as utils
from .dialog import StandardButtonDialog
from .uielement import UIElementEx, TextUIElement
from .window import Window


_fWinsoundIsLoaded = False
try:
    import winsound
    _fWinsoundIsLoaded = True
except:
    pass



# -----------------------------------------------------------------------------
# Class MessageBox
#
# This class defines a message box dialog that has a text message, an optional
# icon, and a row of buttons at the bottom.
#
# A message box is shown via run() or show().  The run() method blocks
# until the message box is dismissed and it returns the button pressed or
# None if the dialog window was dismissed outside the standard buttons.
# The show() method returns immediately but the on_event() event must be
# used to obtain how the message box was dismissed.
#
# On Windows OS, an optional "message beep" can be played when the message
# box is run.
#
class MessageBox(StandardButtonDialog):
    # Class-Wide Public data
    fontsizeDefault = 9 # Default font size
    sizeInchMessageBoxMin = (3, 2) # Minimum size of the message box (inches)
    sizeInchMessageBoxMax = (10, 10) # Maximum size of the message box (inches)

    # Class-Wide Protected data
    _mpstrentryIcon = { # Names of icon image source files and image objects
        'info': ['Info.png', None],
        'question': ['Question.png', None],
        'warning': ['Warning.png', None],
        'error': ['Error.png', None],
        }

    if _fWinsoundIsLoaded:
        _mpstrbeepidIcon = { # winsound message beep ID's for icons that beep
            'question': winsound.MB_ICONQUESTION,
            'warning': winsound.MB_ICONEXCLAMATION,
            'error': winsound.MB_ICONEXCLAMATION,
            }



    @classmethod
    def playMessageBeep(cls, icon):
        if _fWinsoundIsLoaded and (icon is not None):
            beepid = cls._mpstrbeepidIcon.get(icon)
            if beepid is not None:
                try:
                    winsound.MessageBeep(beepid)
                except:
                    pass



    @property
    def result(self):
        return (_self._dialogEvent.reason)



    def __init__(self, title="Message", text="", buttons=('Ok'), showicon=None, centeronfig=None):
        super().__init__(title=title, buttons=buttons, centeronfig=centeronfig)

        self.strText = text
        self.textuielement = None
        self.uielementIcon = None

        self._fIsBlocking = False
        self._sizeInch = self.sizeInchMessageBoxMin

        if (showicon is None) or (showicon not in self._mpstrentryIcon):
            self.icon = None
        else:
            self.icon = showicon



    def centerOnCurrentScreen(self):
            # Center on the screen we're on
            try:
                geomScreen = Window(self.fig).get_screen_geometry()
                self.centerOnPoint((geomScreen[0] + geomScreen[2] / 2, geomScreen[1] + geomScreen[3] / 2))
            except:
                pass



    def centerOnFigure(self, figure = None):
        # For a message box, center the dialog at the center of the screen if we don't have a target figure
        if figure is None:
            figure = self._figCenterOn
        if figure is not None:
            super().centerOnFigure(figure)
        else:
            self.centerOnCurrentScreen()



    def create(self):
        super().create()

        # Measure the text and size the dialog to fit
        dxInch, dyInch = self.pxToInch(utils.getTextExtentPx(self.fig, self.strText, FontProperties(size=self.fontsizeDefault)))
        xInchContent, yInchContent = self.figureToInch((self._rectContent[0], self._rectContent[1]))
        dxMarginInchContent, dyMarginInchContent = self.figureToInch((1.0 - self._rectContent[2], 1.0 - self._rectContent[3]))
        dxInch = xInchContent + dxInch + dxMarginInchContent
        dyInch = yInchContent + dyInch + dyMarginInchContent

        if (self.uielementIcon is not None):
            dxInch = dxInch + self.uielementIcon.sizeInch[0] + self.sizeMarginInterControl[0] # Add space for icon
            dyInch = dyInch + self.uielementIcon.sizeInch[1] / 2 - self._dyInchAscent # Center first text line baseline on icon

        if (dxInch > self.sizeInchMessageBoxMax[0]):
            dxInch = self.sizeInchMessageBoxMax[0]
        elif (dxInch < self.sizeInchMessageBoxMin[0]):
            dxInch = self.sizeInchMessageBoxMin[0]

        if (dyInch > self.sizeInchMessageBoxMax[1]):
            dyInch = self.sizeInchMessageBoxMax[1]
        elif (dyInch < self.sizeInchMessageBoxMin[1]):
            dyInch = self.sizeInchMessageBoxMin[1]

        self.sizeInch = (dxInch, dyInch)


    def createChildren(self):
        super().createChildren()

        # Create message text UI element
        self.textUIElement = TextUIElement(self.fig, self.strText, self.fontsizeDefault, horizontalalignment='left', verticalalignment='top', wrap=True)
        self.addUIElement(self.textUIElement)

        # Create icon UI element if an icon is specified
        image = self.getIconImage(self.icon) if self.icon is not None else None
        if image is None:
            self.uielmentIcon = None
        else:
            self.uielementIcon = UIElementEx(self.fig)
            self.uielementIcon.ax.imshow(image)
            self.uielementIcon.ax.axis('off')
            self.uielementIcon.sizeInch = (0.4, 0.4)
            self.addUIElement(self.uielementIcon)
            self.uielementIcon.origin = 'topleft'
            self.uielementIcon.ptFigure = (self._rectContent[0], 1.0 - self._rectContent[3])

            # Measure the font ascent size so we can align the text to the icon later
            dyInch, dyInchDescent = self.pxToInch(utils.getTextExtent2Px(self.fig, 'GgJjPpQqYz', FontProperties(size=self.fontsizeDefault))[1:])
            self._dyInchAscent = dyInch - dyInchDescent



    def getIconImage(self, icon):
        image = None
        if (icon != None):
            iconentry = self._mpstrentryIcon.get(icon)
            if iconentry is not None:
                image = iconentry[1]
            if image is None:
                try:
                    # Get path to images subdirectory relative to this module's file
                    path = os.path.dirname(os.path.abspath(__file__))
                    path = os.path.join(path, 'images')
                    path = os.path.join(path, iconentry[0])

                    # Load the image file
                    with cbook.get_sample_data(path) as image_file:
                        image = pyplot.imread(image_file)
                        iconentry[1] = image

                except:
                    print("gui.MessageBox: Failed to load message box icon from file '", path, '"')

        return (image)



    def layoutChildren(self):
        super().layoutChildren()

        # Get upper-left corner of content area
        x = self._rectContent[0]
        y = 1.0 - self._rectContent[3]

        # Adjust text position to accomodate the icon
        if (self.uielementIcon is not None):
            dxT, dyT = self.uielementIcon.sizeFigure
            x = x + dxT + self.sizeMarginInterControl[0]
            y = y + dyT / 2 - self.inchToFigure((0, self._dyInchAscent))[1]

        # Position the text
        self.textUIElement.ptFigure = (x, y)



    def onCanvas_Close(self, event):
        canvas = self.fig.canvas
        super().onCanvas_Close(event)

        if (self._fIsBlocking):
            canvas.stop_event_loop()



    def run(self):
        self.show()
        self._fIsBlocking = True
        self.fig.canvas.start_event_loop() # Run blocking event loop until onCanvas_Close() calls canvas.stop_event_loop()
        self._fIsBlocking = False

        return (self._dialogevent.reason)



    def show(self):
        fIsShown = self.fIsShown
        super().show()
        if not fIsShown:
            MessageBox.playMessageBeep(self.icon)






# -----------------------------------------------------------------------------
# Function showinfo()
#
# Show a message box for an informational message with an "Ok" button.  This
# function blocks until the message box is dismissed.
#
# Return:
#   (Return)  Always returns "Ok"
#
def showinfo(title, text, centeronfig=None):
    MessageBox(title=title, text=text, centeronfig=centeronfig, showicon='info').run()
    return 'Ok'



# -----------------------------------------------------------------------------
# Function showerror()
#
# Show a message box for an error message with an "Ok" button.  This
# function blocks until the message box is dismissed and returns "Ok".
#
# Return:
#   (Return)  Always returns "Ok"
#
def showerror(title, text, centeronfig=None):
    MessageBox(title=title, strText=text, centeronfig=centeronfig, showicon='error').run()
    return 'Ok'



# -----------------------------------------------------------------------------
# Function showquestion()
#
# Show a message box for a question message with "Yes" and "No" buttons.  This
# function blocks until the message box is dismissed and returns the button
# pressed.
#
# Return:
#   (Return)  Returns "Yes" or "No" depending on the button pressed
#
def showquestion(title, text, centeronfig=None):
    button = MessageBox(title=title, text=text, centeronfig=centeronfig, showicon='question', buttons=('Yes', 'No')).run()
    if button is None:
        button = 'No'
    return button



# -----------------------------------------------------------------------------
# Function showwarning()
#
# Show a message box for a warning message with an "Ok" button.  This
# function blocks until the message box is dismissed.
#
# Return:
#   (Return)  Always returns "Ok"
#
def showwarning(title, text, centeronfig=None):
    MessageBox(title=title, text=text, centeronfig=centeronfig, showicon='warning').run()
    return 'Ok'
