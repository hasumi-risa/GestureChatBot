# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import copy

import matplotlib.cbook
import matplotlib.widgets


# --------------------------------------------------------------------------------------------
# Class Cursor
#
# The built-in Cursor class only handles one axes at a time and doesn't work properly when
# there are overlapping axes like a second axis twinned to the first.  This class enables the
# built-in Cursor class to handle multiple axes.
# --------------------------------------------------------------------------------------------
class Cursor(matplotlib.widgets.Cursor):
    # --------------------------------------------------------------------------------------------
    # Class HitTestEvent
    #
    # This event is sent when an input event is received to determine whether the event should be
    # considered to have occurred in this cursor's axes.  If so, the event handler should set
    # is_hit to True.  If not, the event handler should leave is_hit untouched to avoid overriding
    # a True result from another connected event handler.
    # --------------------------------------------------------------------------------------------
    class HitTestEvent:
        def __init__(self, event):
            self.event = event
            self.is_hit = False



    #------------------------------------------------------------------------------
    # Instance initialization
    #
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._callbackregistry = matplotlib.cbook.CallbackRegistry()



    #------------------------------------------------------------------------------
    # Method disconnect()
    #
    # Removes a function from the connected event.
    #
    # Arguments:
    #   connectionid    The ID returned by an event connection method identifying
    #                       the function to remove
    #
    def disconnect(self, connectionid):
        self._callbackregistry.disconnect(connectionid)



    #------------------------------------------------------------------------------
    # Method on_hit_test()
    #
    # Connect a function to the hit_test event.
    #
    # This event is triggered when this cursor receives an input event and the
    # function determines whether the event is in this cursor's axes.  The function
    # receives a HitTestEvent and sets HitTestEvent.is_hit to True if the class
    # should try to map the event to cursor's axes.
    #
    # When no functions are attached, this class by default tries to map all input
    # events to its axes.
    #
    #
    # Arguments:
    #   fn  Function to add
    #
    # Returns:
    #   connectionid    ID the new connection; pass to disconnect() to remove the
    #                       function from the list
    #
    def on_hit_test(self, fn):
        return (self._callbackregistry.connect('hittest', fn))



    #------------------------------------------------------------------------------
    # Method onmove()
    #
    # Handle a motion-notify input event.
    #
    # Arguments:
    #   event  MouseEvent object
    #
    def onmove(self, event):
        # When there are overlapping axes like a second axis twinned to the first,
        # event.inaxes is the second axes instead of the first.  To get around this,
        # if when the event was generated for another axes that shares at least one axis
        # with our axes, we'll map the mouse motion event to our axes.
        if self._callbackregistry.callbacks:
            hittestevent = self.HitTestEvent(event)
            self._callbackregistry.process('hittest', hittestevent)
            fMapEvent = hittestevent.is_hit
        else:
            fMapEvent = True # By default, always try to map the event to our axes

        # Try to map the event to our axes if needed
        if fMapEvent and (event.inaxes != self.ax):
            x, y = self.ax.transAxes.inverted().transform((event.x, event.y))

            if ((x >= 0) and (x < 1) and (y >= 0) and (y < 1)):
                event = copy.copy(event)
                event.inaxes = self.ax
                event.xdata, event.ydata = self.ax.transData.inverted().transform((event.x, event.y))

        super().onmove(event)




