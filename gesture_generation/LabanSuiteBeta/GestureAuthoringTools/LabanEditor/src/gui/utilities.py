# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

from __future__ import absolute_import

from matplotlib.font_manager import FontProperties



#------------------------------------------------------------------------------
# Get the rectangle for the specifies axes
#
# Arguments:
#   ax      Axes object to examine
#
# Returns:
#   (Return)  Rectangle of the axes (xLeft, yBottom, xRight, yTop)
def getAxesRect(ax):
    bbox = ax.get_position()
    (xLeft, yBot) = bbox.min
    (xRight, yTop) = bbox.max
    return (xLeft, yBot, xRight, yTop)



#------------------------------------------------------------------------------
# Get the rectangle for the specified text in pixels
#
# The dimensions includes the font descent even if the text doesn't have any
# descenders.
#
# Arguments:
#   fig             Figure in whose coordinates we'll return
#   str             String to measure
#   fontproperties  A FontProperties() object specifying the format of the string
#
# Returns:
#   (Return)        2-tuple with the total width and total height in pixels
#
def getTextExtentPx(fig, str, fontproperties=FontProperties()):
    # Get the renderer needed to measure the text
    renderer = fig.canvas.get_renderer()
    if renderer is None:
        fig.canvas.draw() # If we don't have a rendered, we need to draw the figure first
        renderer = fig.canvas.get_renderer()

    # Create a text artist with the text to measure
    text = fig.text(0.0, 0.0, str, color='r', fontproperties=fontproperties, wrap=False)

    # Get the text metrics in pixels
    bbox = text.get_window_extent(renderer)
    text.remove()

    return (bbox.width, bbox.height)



#------------------------------------------------------------------------------
# Get the rectangle for the specified text in pixels
#
# This function extends getTextExtentPx() to return the font descent height.
#
#
# Arguments:
#   fig             Figure in whose coordinates we'll return
#   str             String to measure
#   fontproperties  A FontProperties() object specifying the format of the string
#
# Returns:
#   (Return)        3-tuple with the total width, total height, and font descend
#                       in pixels
#
def getTextExtentDescentPx(fig, str, fontproperties=FontProperties()):
    # Get a renderer needed to measure the text
    renderer = fig.canvas.get_renderer()
    if renderer is None:
        fig.canvas.draw() # If we don't have a rendered, we need to draw the figure first
        renderer = fig.canvas.get_renderer()

    # Create a text artist with the text to measure
    text = fig.text(0.0, 0.0, str, color='r', fontproperties=fontproperties, wrap=False)

    # Get the text metrics in pixels
    bbox = text.get_window_extent(renderer) # Note that this includes the font descent even though the text doesn't have any
    dyPxDescent = renderer.get_text_width_height_descent(str, text._fontproperties, False)[2] # get_text_width_height_descent() doesn't handle newlines
    text.remove()

    return (bbox.width, bbox.height, dyPxDescent)



#------------------------------------------------------------------------------
# Get the rectangle for the specified text in pixels
#
# This function returns the width, height, and font descent height in pixels of
# a string.  The returned font descent height is zero and the text height does
# not include descender height if the text does not include any glyphs with
# descenders.
#
# It is faster than getTextExtentPx()/getTextExtentPxEx(), but does not handle
# multiline text (text with embedded newlines.)
#
#
# Arguments:
#   fig             Figure in whose coordinates we'll return
#   str             String to measure
#   fontproperties  A FontProperties() object specifying the format of the string
#
# Returns:
#   (Return)    3-tuple with the total width, total height, and font descend
#               in pixels
#
def getTextExtent2Px(fig, str, fontproperties=FontProperties()):
    # Get a renderer needed to measure the text
    renderer = fig.canvas.get_renderer()
    if renderer is None:
        fig.canvas.draw() # If we don't have a rendered, we need to draw the figure first
        renderer = fig.canvas.get_renderer()

    # Get the text metrics in pixels
    dx, dy, dyPxDescent = renderer.get_text_width_height_descent(str, fontproperties, False) # get_text_width_height_descent() doesn't handle newlines

    return (dx, dy, dyPxDescent)



#------------------------------------------------------------------------------
# Get the rectangle for the specified layout
#
# Arguments:
#   layout      Layout to convert (xLeft, yBottom, width, height)
#
# Returns:
#   (Return)  Rectangle of the layout (xLeft, yBottom, xRight, yTop)
def layoutToRect(layout):
    xLeft = layout[0]
    yBot = layout[1]
    return (xLeft, yBot, xLeft + layout[2], yBot + layout[3])



#------------------------------------------------------------------------------
# Get the layout for the specified rectangle
#
# Arguments:
#   rect    Rect to convert (xLeft, yBottom, xRight, yTop)
#
# Returns:
#   (Return)  Layout of the rect (xLeft, yBottom, width, height)
def rectToLayout(rect):
    return (rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1])






##------------------------------------------------------------------------------
## Class CallbackList
##
## This class manages a list of functions.  This is commonly used to handle
## functions registered for an event.
##
## The connect() and disconnect() methods adds and removes functions.  To check
## whether there are any or how many functions are connected, use The Python
## bool() and len() functions on a class instance.  To call the connected
## functions, call the class instance directly or use the call() method.
##
#class CallbackList:
#    connectionidNext = 0 # Next connection ID


#    #------------------------------------------------------------------------------
#    # Instance initialization
#    #
#    def __init__(self):
#        self.mpconnectionidfn = {} # Mapping from connection ID's to callback functions



#    #------------------------------------------------------------------------------
#    # Method __bool__
#    #
#    # Support for Python bool()--returns whether functions are connected
#    #
#    # Returns:
#    #   (Return)  True if there are connected functions, false otherwise
#    #
#    def __bool__(self):
#        return bool(self.mpconnectionidfn)



#    #------------------------------------------------------------------------------
#    # Method call
#    #
#    # Support for calling this instance directly which calls the connected
#    # functions.
#    #
#    # Arguments:
#    #   param   Parameter to pass to each function
#    #
#    def __call__(self, param):
#        self.call(param)



#    #------------------------------------------------------------------------------
#    # Method __len__
#    #
#    # Support for Python len()--returns the number of connected functions.
#    #
#    # Returns:
#    #   (Return)  The number of connected functions
#    #
#    def __len__(self):
#        return len(self.mpconnectionidfn)



#    #------------------------------------------------------------------------------
#    # Method call
#    #
#    # Call the connected functions.
#    #
#    # Arguments:
#    #   param   Parameter to pass to each function
#    #
#    def call(self, param):
#        for connectionid, fn in self.mpconnectionidfn.items():
#            fn(param)



#    #------------------------------------------------------------------------------
#    # Method connect
#    #
#    # Add a function to the list of connected functions.
#    #
#    # Arguments:
#    #   fn  Function to add
#    #
#    # Returns:
#    #   connectionid    ID the new connection; pass to disconnect() to remove the
#    #                       function from the list
#    #
#    def connect(self, fn):
#        connectionid = self.connectionidNext
#        self.connectionidNext += 1
#        self.mpconnectionidfn[connectionid] = fn

#        return (connectionid)


#    #------------------------------------------------------------------------------
#    # Method disconnect
#    #
#    # Removes a function from the list of connected functions.
#    #
#    # Arguments:
#    #   connectionid    The ID returned by connect() identifying the function to
#    #                       remove
#    def disconnect(self, connectionid):
#        try:
#            del self.mpconnectionidfn[connectionid]
#        except KeyError:
#            pass
