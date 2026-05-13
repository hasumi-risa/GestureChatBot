# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

from __future__ import absolute_import

import sys

import matplotlib.cbook as cbook
import matplotlib as plt
import matplotlib.widgets as widgets

from . import utilities



class ScrollBar(widgets.AxesWidget):
    @property
    def value(self):
        return (self._value)

    @value.setter
    def value(self, value):
        self.setValue(value)


    @property
    def value_page(self):
        return self._value_page
    @value_page.setter
    def value_page(self, value_page):
        if value_page < 1:
            raise ValueError("Page size must be greater than 0")
        if value_page != self._value_page:
            self._value_page = value_page
            self.update()


    @property
    def value_range(self):
        return (self._value_range)

    @value_range.setter
    def value_range(self, value_range):
        if value_range[0] > value_range[1]:
            raise ValueError("Range minimum value value_range[0] is greater than maximum value value_range[1]")

        self._value_range = value_range
        if self.orientation == 'horizontal':
            self.ax.set_xlim(value_range)
        else:
            self.ax.set_ylim(value_range)

        value = self._value
        if value > (value_range[1] - self.value_page):
            value = value_range[1] - self.value_page
        if value < value_range[0]:
            value = value_range[0]
        self._setValue(value)


    def __init__(self, ax, orientation='vertical', value=0, value_range=(0, 100), value_page=10):
        super().__init__(ax)


        self._callbackregistry = cbook.CallbackRegistry()
        self._value = value
        self._value_range = []
        self._value_page = value_page
        self.fMouseDragIsActive = False
        self.orientation = orientation
        self.value_offset_mouse_drag = 0
        self.value_range = value_range

        # Turn off tick marks and labels
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Set axes to value range
        if orientation == 'horizontal':
            ax.set_xlim(value_range)
        else:
            ax.set_ylim(value_range)

        # Connect to mouse events
        self.connect_event('button_press_event', self.onEvent_ButtonPress)
        self.connect_event('button_release_event', self.onEvent_ButtonRelease)
        self.connect_event('motion_notify_event', self.onEvent_MotionNotify)

        # Add thumb
        if orientation == 'horizontal':
           self.poly = ax.axvspan(0, 1, 0, 1, color='gray')
        else:
           self.poly = ax.axhspan(0, 1, 0, 1, color='gray')

        # Update the thumb to the correct location and size
        self.update()


    def on_changed(self, fn):
        return (self._callbackregistry.connect('changed', fn))



    def disconnect(self, connectionid):
        self._callbackregistry.disconnect(connectionid)



    def onEvent_ButtonPress(self, event):
        if self.ignore(event) or (event.button != 1) or (event.inaxes != self.ax):
            return

        value = event.xdata if self.orientation == 'horizontal' else (self._value_range[1] - event.ydata)
        if (value >= self.value) and (value < (self._value + self._value_page)):
            # Start dragging the thumb
            self.fMouseDragIsActive = True
            self.value_offset_mouse_drag = self._value - value
            event.canvas.grab_mouse(self.ax)
        elif value < self.value:
            # Page up
            self.setValue(self._value - self._value_page)
        elif value > self.value:
            # Page down
            self.setValue(self._value + self._value_page)


    def onEvent_ButtonRelease(self, event):
        if self.ignore(event) or (event.button != 1):
            return

        if self.fMouseDragIsActive:
            event.canvas.release_mouse(self.ax)
            self.fMouseDragIsActive = False


    def onEvent_MotionNotify(self, event):
        if self.ignore(event) or not self.fMouseDragIsActive:
            return

        # Update thumb drag
        value = event.xdata if self.orientation == 'horizontal' else (self._value_range[1] - event.ydata)
        value = value + self.value_offset_mouse_drag

        self.setValue(value)


    def setValue(self, value):
        if self._setValue(value) and self.eventson:
            self._callbackregistry.process('changed', self._value)



    def update(self):
        xy = self.poly.xy
        value = self._value
        value_page = self._value_page
        if self.orientation == 'horizontal':
            xy[0] = (value, 0)
            xy[1] = (value, 1)
            xy[2] = (value + value_page, 1)
            xy[3] = (value + value_page, 0)
        else:
            value = self._value_range[1] - value
            xy[0] = (0, value)
            xy[1] = (1, value)
            xy[2] = (1, value - value_page)
            xy[3] = (0, value - value_page)
        xy[4] = xy[0]
        self.poly.xy = xy

        if self.drawon:
            self.ax.figure.canvas.draw()



    def _setValue(self, value):
        # Don't set value past last page
        if (value + self._value_page) > self._value_range[1]:
            value = self._value_range[1] - self._value_page

        # Don't set value before minimum range
        if value < self._value_range[0]:
            value = self._value_range[0]

        if (abs(value - self._value) > sys.float_info.epsilon):
            self._value = value
            self.update()
            return True
        else:
            return False

