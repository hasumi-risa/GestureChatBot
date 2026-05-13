# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

from __future__ import absolute_import

import re

import matplotlib
import matplotlib.pyplot as plt



# -----------------------------------------------------------------------------
# class Window
#
# This helper class abstracts window operations on a figure to hide the
# different logic needed for different backends.
#
class Window:
    # -----------------------------------------------------------------------------
    # Property geometry
    #
    # This property specifies the figure window's geometry in screen pixel
    # coordinates (left, top, width, height.)
    #
    # Note that the backend may override or ignore the setting of this property;
    # for example, the new geometry might be ignored if the geometry is off-screen.
    #
    @property
    def geometry(self):
        geometry = (0, 0, 0, 0)
        window, backend = self._get_window_backend()
        if (backend == 'tkagg'):
            str = window.wm_geometry()
            # Parse the Tk-style geometry specification.  Note that if x (or y) is negative, TkAgg
            # returns "+-X" with an extra "+" to indicate it's relative to the left (top) edge of the desktop
            match = re.search('=?(\d+)x(\d+)\+?((\+|-)(\d+))((\+|-)(\d+))', str)
            if match:
                rgStr = match.group(3, 6, 1, 2)
                geometry = [int(s) for s in rgStr]

        elif backend.startswith('wx'):
            pt = window.GetPosition()
            size = window.GetSize()
            geometry = (pt[0], pt[1], size[0], size[1])

        elif backend.startswith('qt'):
            geometry = window.geometry().getRect()

        return geometry


    @geometry.setter
    def geometry(self, geometry):
        if (self.figure is not None):
            window, backend = self._get_window_backend()
            if (backend == 'tkagg'):
                # Note that like when getting the geometry, there are permanent "+" signs on x and y so they'll
                # be relative to the left/top desktop edges even if they are negative
                str = "%dx%d+%d+%d" % (geometry[2], geometry[3], geometry[0], geometry[1])
                window.wm_geometry(str)
                pass

            elif backend.startswith('wx'):
                window.SetPosition(int(geometry[0]), int(geometry[1]))
                window.SetSize(int(geometry[2]), int(geometry[3]))
                pass

            elif backend.startswith('qt'):
                window.setGeometry(int(geometry[0]), int(geometry[1]), int(geometry[2]), int(geometry[3]))



    # -----------------------------------------------------------------------------
    # Initialize object of this class
    #
    # Arguments:
    #   figure  Wrap this figure's window
    #
    def __init__(self, figure):
        self.figure = figure



    # -----------------------------------------------------------------------------
    # Method get_screen_geometry()
    #
    # Returns the geometry (left, top, width, height) of the screen containing the
    # the current figure.  If a figure spans multiple screens, the screen picked is
    # dependant on the backend and windowing system but is usually the one
    # containing the majority of the figure.
    #
    # Returns:
    #   geometry    Geometry of the screen containing the current figure.
    #
    def get_screen_geometry(self):
        # This is inelegant but effective in determining the working area size.
        # There doesn't seem to be a better backend-independent way of doing this.

        # Create a figure at the same location as the current figure
        figProbe = plt.figure()
        windowProbe = Window(figProbe)
        figProbe.show() # Make sure the window is created
        windowProbe.geometry = self.geometry

        # Make the figure full screen and get the geometry
        figProbe.canvas.manager.full_screen_toggle()
        geometry = windowProbe.geometry
        plt.close(figProbe)

        return geometry



    # -----------------------------------------------------------------------------
    # Method _get_window_backend()
    #
    # Returns the window and name of the matplotlib backend of the figure.
    #
    # Returns:
    #   window    Matplotlib window object
    #   backend   Name of the matplotlib backend in all lower-case
    #
    def _get_window_backend(self):
        strBackend = ""
        window = None
        if self.figure is not None:
            window = self.figure.canvas.manager.window
            strBackend = matplotlib.get_backend().lower()

        return (window, strBackend)
