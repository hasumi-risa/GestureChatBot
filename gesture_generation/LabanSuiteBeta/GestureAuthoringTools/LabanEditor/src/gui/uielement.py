# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

from __future__ import absolute_import

import sys
from contextlib import ExitStack

import matplotlib
import matplotlib.cbook

from . import utilities as utils
from . import widgets as guiwidgets
from .coordconvert import CoordConvert



# -----------------------------------------------------------------------------
# Class UIElement
#
# This class enables an element on a figure to have configurable position and
# size behavior including:
#   * Supporting both inches and figure coordinates to specify size and position
#   * Positioning relative to the top-left, top-right, bottom-right or bottom-
#     left corner of the figure
#   * Fixed, figure-base, or edge-based size and position as the parent
#     figure is resized.
#
# Furthermore, unlike standard matplotlib behavior, objects of this class default
# to an origin "topleft" which is more like standard UI coordinate systems and
# much easier to handle when positioning and sizing UI elements.
#
class UIElement:
    # -----------------------------------------------------------------------------
    # Property origin
    #
    # This property specifies the location of the origin (0, 0) and direction of
    # the axes when positioning the element (via self.ptFigure and self.ptInch.)
    #
    # It also specifies which corner of the element is position at ptFigure/Inch.
    # For instance, "topleft" means the upper-left corner of the element is
    # positioned relative to the upper-left corner of the figure (point (0.0, 1.0)
    # in "bottomleft" origin figure coordinates), with x increasing right and y
    # increasing down.
    #
    # Note that the element's size given by self.sizeFigure and self.sizeInch
    # are normally positive and isn't affected by the axis direction used to
    # position the element.  (For instance, with the origin set to "topleft"
    # where y increases down self.sizeInch[1] is still +0.5 for a 1/2-inch tall
    # element.)
    #
    # Matplotlib usually uses "bottomleft" for positioning items in a figure
    # or axes but that's quite inconvenient for general UI like dialog boxes or
    # tool windows where "topleft" is more convenient and familiar.
    #
    # Values:
    #   'bottomleft'   (Default) bottomleft of figure
    #   'bottomright'  bottomright of figure
    #   'topleft'      Upper-left corner of figure
    #   'topright'     Upper-right corner of figure
    #
    # 'bottomleft' has the highest performance.
    #
    @property
    def origin(self):
        return self._origin


    @origin.setter
    def origin(self, strOrigin):
        self._origin = strOrigin

        # Update the position-inch attribute and sizing edge offsets to reflect the new origin
        self._refreshPtInch()
        self._updateDInchSizeEdge()



    # -----------------------------------------------------------------------------
    # Property ptFigure
    #
    # This property specifies the element's position in figure coordinates
    # (0.0 - 1.0, 0.0 - 1.0) relative to the current origin (see self.origin
    # property.)
    #
    # This property mirrors ptInch (but in figure coordinates.)
    #
    @property
    def ptFigure(self):
        return self._convertBetweenOriginBottomLeft(self._ptFigure)

    @ptFigure.setter
    def ptFigure(self, pt):
        self._ptFigure = self._convertBetweenOriginBottomLeft(pt)
        self._refreshPtInch() # Update location in inches now in case figure is resized



    # -----------------------------------------------------------------------------
    # Property ptInch
    #
    # This property specifies the element's position in inches relative to the
    # current origin (see self.origin property.)
    #
    # This property mirrors ptFigure (but in inches.)
    #
    @property
    def ptInch(self):
        return self._ptInch

    @ptInch.setter
    def ptInch(self, pt):
        self._ptInch = pt
        self._ptFigure = self._convertBetweenOriginBottomLeft(self.inchToFigure(pt))



    # -----------------------------------------------------------------------------
    # Property ptMode
    #
    # This property specifies how the element's position changes with the figure.
    #
    # A single keyword sets the same positioning behavior both horizontally and
    # vertically.  Two keywords separated by a space sets the horizontal and
    # vertical positionng behavior independently.  E.g., "fixed" sets a both fixed
    # horizontal and vertical position while "fixed edge" sets a fixed horizontal
    # position and an edge-relative vertical position.
    #
    # Valid keywords are:
    #   fixed       The element maintains a fixed position in the figure
    #   figure      The element position is a set number of figure units
    #   edge        The positioning edge maintains a fixed distance from the
    #                   corresponding figure edge (same as fixed)
    #
    # 'figure' has the highest performance.
    #
    # A "positioning edge" is the edge used to position the element.  For instance,
    # if the element's origin is set to "topleft", the top and left edges of the
    # element are the positioning edges.
    #
    @property
    def ptMode(self):
        return self._refmodeToStr(self._refmodePt[0]) + ' ' + self.RefModeToStr(self._refmodePt[1])

    @ptMode.setter
    def ptMode(self, strMode):
        rgstr = strMode.split()
        if len(rgstr) == 1:
            rgstr.append(rgstr[0]) # Single keyword, duplicate for both horizontal and vertical
        if len(rgstr) == 2:
            self._refmodePt = (self._strToRefMode(rgstr[0]), self._strToRefMode(rgstr[1]))



    # -----------------------------------------------------------------------------
    # Property sizeFigure
    #
    # This property specifies the element's size in figure coordinates
    # (0.0 - 1.0, 0.0 - 1.0).  Note than an element's size is always positive
    # regardless of the current origin.
    #
    # This property mirrors sizeInch (but in figure coordinates.)
    #
    @property
    def sizeFigure(self):
        return self._sizeFigure

    @sizeFigure.setter
    def sizeFigure(self, size):
        if (size[0] < 0) or (size[1] < 0):
            size = list(size)
            if size[0] < 0:
                size[0] = 0
            if size[1] < 0:
                size[1] = 0

        # When the origin is not "bottomleft", self._ptFigure is affected by the element size so we have to save and restore the origin-relative position so the element does not move
        ptFigureSav = self.ptFigure
        self._sizeFigure = size
        self._sizeInch = self.figureToInch(size) # Update location in inches now in case figure is resized
        self.ptFigureSav = ptFigureSav

        self._updateDInchSizeEdge()



    # -----------------------------------------------------------------------------
    # Property sizeInch
    #
    # This property specifies the element's size in inches.  Note than an element's
    # size is always positive regardless of the current origin.
    #
    # This property mirrors sizeFigure (but in inches.)
    #
    @property
    def sizeInch(self):
        return self._sizeInch

    @sizeInch.setter
    def sizeInch(self, size):
        if (size[0] < 0) or (size[1] < 0):
            size = list(size)
            if size[0] < 0:
                size[0] = 0
            if size[1] < 0:
                size[1] = 0

        # When the origin is not "bottomleft", self._ptFigure is affected by the element size so we have to save and restore the origin-relative position so the element does not move
        ptFigureSav = self.ptFigure
        self._sizeInch = size
        self._sizeFigure = self.inchToFigure(size)
        self.ptFigure = ptFigureSav

        self._updateDInchSizeEdge()



    # -----------------------------------------------------------------------------
    # Property sizeMode
    #
    # This property specifies how the element's size changes with the figure.
    #
    # A single keyword sets the same sizing behavior both horizontally and
    # vertically.  Two keywords separated by a space sets the horizontal and
    # vertical sizing behavior independently.  E.g., "fixed" sets a both fixed
    # horizontal and vertical size while "fixed edge" sets a fixed horizontal size
    # and an edge-relative vertical size.
    #
    # Valid keywords are:
    #   fixed       The element maintains a fixed screen size
    #   figure      The element size is a set number of figure units
    #   edge        The sizing edge maintains a fixed distance from the corresponding
    #                   figure edge
    #
    # 'figure' has the highest performance.
    #
    # A "sizing edge" is the edge opposite the edge used to position the element.
    # For instance, if the element's origin is set to "topleft", the bottom and right
    # edges of the element are the sizing edges.
    #
    @property
    def sizeMode(self):
        return self._refmodeToStr(self._refmodeSize[0]) + ' ' + self.RefModeToStr(self._refmodeSize[1])

    @sizeMode.setter
    def sizeMode(self, strMode):
        rgstr = strMode.split()
        if len(rgstr) == 1:
            rgstr.append(rgstr[0]) # Single keyword, duplicate for both horizontal and vertical
        if len(rgstr) == 2:
            self._refmodeSize = (self._strToRefMode(rgstr[0]), self._strToRefMode(rgstr[1]))
            self._updateDInchSizeEdge()



    ## -----------------------------------------------------------------------------
    ## Enumeration DirtyState
    ##
    ## The internal element position and size attributes (self._pt* and self._size*)
    ## are only updated as necessary for better performance.  The position and size
    ## each have a pair of attributes, one for figure units and another for inches.
    ## These enumerated values specify which need to be updated from the other
    ##
    #class DirtyState:
    #    CLEAN = 0 # _pt/sizeFigure/Inch are clean
    #    DIRTYFIGURE = 1 # _pt/sizeFigure needs to be recalculated from _pt/sizeInch
    #    DIRTYINCH = 2 # _pt/sizeInch needs to be recalculated from _pt/sizeFigure


    # -----------------------------------------------------------------------------
    # Enumeration RefMode
    #
    # The reference mode of the position coordinate or size value that determines
    # how value changes as the parent figure changes (see self.ptMode and
    # self.sizeMode.)
    #
    class RefMode:
        FIXED = 0 # Position/size is based on figure-independent values (pt/sizeInch)
        FIGURE = 1 # Position/size is based on figure coordinates (pt/sizeInch)
        EDGE = 2 # Position: same as FIXED; Size: figure-independent offset from edge of figue (_dinchSizeEdge)

    # Mapping from origin for pt to origin for size for RefMode.Edge
    _mporiginoriginSize = {
        'bottomleft': 'topright',
        'bottomright': 'topleft',
        'topleft': 'bottomright',
        'topright': 'bottomleft',
        }



    # -----------------------------------------------------------------------------
    # Initialize object instance
    #
    # Arguments:
    #   fig     Reference figure for this element
    #   sizeInch    Initial size or None for default (inches)
    #   sizeFigure  Initial size or None for default (figure units)
    #
    #   Both sizeInch and sizeFigure cannot be specified.
    #
    def __init__(self, fig, sizeInch=None, sizeFigure=None):
        if (sizeInch is not None) and (sizeFigure is not None):
            raise ValueError("Both sizeInch and sizeFigure cannot be specified at the same time")

        self.fig = fig # The figure this element is on
        self._origin = 'topleft' # Origin used for positioning this element

        if sizeInch is None:
           sizeInch = self.figureToInch(sizeFigure) if sizeFigure is not None else (0, 0)
        if sizeFigure is None:
            sizeFigure = self.inchToFigure(sizeInch) if sizeInch is not None else (0, 0)

        self._sizeInch = sizeInch # Size of element (inches)
        self._sizeFigure = sizeFigure # Size of element (figure coordinates)
        self._ptInch = (0, 0) # Location of the element relative to the parent figure's axes (figure inches, w/respect to self._origin)
        self._ptFigure = (0, 1.0 - sizeFigure[1]) # Location of the element relative to the parent figure's axes (figure coordinates, ignoring self._origin)

        self._refmodePt = (self.RefMode.FIXED, self.RefMode.FIXED)
        self._refmodeSize = (self.RefMode.FIXED, self.RefMode.FIXED)
        self._dinchSizeEdge = (0, 0) # Dist from figure edge for element's sizing edges in RefMode.EDGE (inches)


    # -----------------------------------------------------------------------------
    # commitRect
    #
    # The element is updated with any changes to ptFigure/Inch and sizeFigure/Inch.
    #
    # Subclasses typically override this method to set the new position and size
    # into the wrapped matplotlib object from self._ptFigure and self._sizeFigure.
    #
    def commitRect(self):
        pass



    # -----------------------------------------------------------------------------
    # figureToInch
    #
    # Converts a pair of values in figure coordinates to the corresponding values
    # in inches.
    #
    def figureToInch(self, pairFigureValue):
        return CoordConvert.figureToInch(pairFigureValue, self.fig)



    # -----------------------------------------------------------------------------
    # figureToInch
    #
    # Converts a pair of values in figure coordinates to the corresponding values
    # in pixels.
    #
    def figureToPx(self, pairFigureValue):
        return (CoordConvert.figureToPx(pairFigureValue, self.ax.figure))



    # -----------------------------------------------------------------------------
    # inchToFigure
    #
    # Converts a pair of values in inches to the corresponding values in figure
    # coordinates.
    #
    def inchToFigure(self, pairInchValue):
        return CoordConvert.inchToFigure(pairInchValue, self.fig)



    # -----------------------------------------------------------------------------
    # inchToPx
    #
    # Converts a pair of values in inches to the corresponding values in pixels.
    #
    def inchToPx(self, pairInchValue):
        return (CoordConvert.inchToPx(pairInchValue, self.fig))



    # -----------------------------------------------------------------------------
    # pxToFigure
    #
    # Converts a pair of values in pixels to the corresponding values in figure
    # coordinates.
    #
    def pxToFigure(self, pairPxValue):
        return (CoordConvert.pxToFigure(pairPxValue, self.fig))



    # -----------------------------------------------------------------------------
    # pxToInch
    #
    # Converts a pair of values in pixels to the corresponding values in inches.
    #
    def pxToInch(self, pairPxValue):
        return (CoordConvert.pxToInch(pairPxValue, self.fig))



    # -----------------------------------------------------------------------------
    # recalcRect
    #
    # The element's pt/sizeFigure are updated to handle any changes to the
    # parent figure's size to keep the element at the position and size specified
    # by pt/sizeInch.
    #
    # The element's axes aren't updated until commitRect() is called; until then
    # the element will move and resize with the figure.
    #
    def recalcRect(self):
        fHaveNewPtFigure = False
        fHaveNewSizeFigure = False
        fPtInchIsDirty = False
        fSizeInchIsDirty = False

        # We can use the origin-relative position as if it were at the lower-left origin to can calculate the
        # new element position and size movement the same way regardless of the origin.  Note that this is not
        # the same as self._ptFigure which is the lower-left corner of the element relative to the bottomleft
        # origin regardless of the current origin.
        ptFigureNew = list(self.ptFigure)
        sizeFigureNew = list(self._sizeFigure) #_sizeFigure/Inch might be tuples and we need modifyable lists

        # Calculate positions needed by the current positioning mode
        if (self._refmodePt[0] != self.RefMode.FIGURE) or (self._refmodePt[1] != self.RefMode.FIGURE):
            ptFigureFixed = self.inchToFigure(self._ptInch)

        # Update the position
        for i in (0, 1):
            mode = self._refmodePt[i]

            if mode != self.RefMode.FIGURE: # For position, FIXED and EDGE modes have the same effect
                ptFigureNew[i] = ptFigureFixed[i]
                fHaveNewPtFigure = True
            if mode != self.RefMode.FIXED:
                fPtInchIsDirty = True


        # Calculate sizes needed by the current sizing mode
        if (self._refmodeSize[0] == self.RefMode.FIXED) or (self._refmodeSize[1] == self.RefMode.FIXED):
            sizeFigureFixed = self.inchToFigure(self._sizeInch)
        if (self._refmodeSize[0] == self.RefMode.EDGE) or (self._refmodeSize[1] == self.RefMode.EDGE):
            dfigure = self.inchToFigure(self._dinchSizeEdge)
            edgeFigure = [1 - dfigure[i] for i in (0, 1)] # Positions of the sizing edges in EDGE mode (figure units)

        # Update the size
        for i in (0, 1):
            mode = self._refmodeSize[i]

            if mode == self.RefMode.FIXED:
                sizeFigureNew[i] = sizeFigureFixed[i]
                fHaveNewSizeFigure = True
            elif mode == self.RefMode.FIGURE:
                fSizeInchIsDirty = True
            elif mode == self.RefMode.EDGE:
                dfigure = edgeFigure[i] - ptFigureNew[i]
                sizeFigureNew[i] = dfigure if (dfigure > 0) else 0
                fHaveNewSizeFigure = True
                fSizeInchIsDirty = True

        # Set the new size
        if fHaveNewSizeFigure:
            self._sizeFigure = sizeFigureNew

        # Update size in inches if necessary
        if fSizeInchIsDirty:
            self._sizeInch = self.figureToInch(sizeFigureNew)

        # Any change to the element size will need a corresponding change in self._ptFigure when
        # the origin is not "bottomleft"
        if fHaveNewPtFigure or (fHaveNewSizeFigure and (self.origin != 'bottomleft')):
            self.ptFigure = ptFigureNew
        elif fPtInchIsDirty:
            self._refreshPtInch()



    # -----------------------------------------------------------------------------
    # _convertBetweenOriginBottomLeft
    #
    # This function returns the equivalent coordinates for an element position
    # relative to the current origin to the "bottomleft" origin and vice-versa so
    # that the element (relative to the new origin) is at the same position in the
    # figure.
    #
    # The conversion is symmetric so calling this method toggles the coordinates
    # between the current origin and "bottomleft".  Passing the result from this
    # method to this method again returns the original coordinates.
    #
    #
    # Arguments:
    #   ptFigure  Position of the element relative to the current origin or
    #                 "bottomleft" origin (figure units)
    #
    # Returns:
    #   ptFigure  Equivalent position of the element relative to the other origin
    #                 (figure units)
    #
    def _convertBetweenOriginBottomLeft(self, ptFigure):
        return self._convertBetweenOriginBottomLeftCore(self.origin, ptFigure)



    # -----------------------------------------------------------------------------
    # _convertBetweenOriginBottomLeftCore
    #
    # This function returns the equivalent coordinates for an element position
    # relative to the current origin to the "bottomleft" origin and vice-versa so
    # that the element (relative to the new origin) is at the same position in the
    # figure.
    #
    # The conversion is symmetric so calling this method toggles the coordinates
    # between the specified origin and "bottomleft".  Passing the result from this
    # method to this method again returns the original coordinates.
    #
    #
    # Arguments:
    #   origin    Position is converted to or from this origin
    #   ptFigure  Position of the element relative to the origin (figure units)
    #
    # Returns:
    #   ptFigure  Equivalent position of the element relative to the specified
    #                 or "bottomleft" origin (figure units)
    #
    def _convertBetweenOriginBottomLeftCore(self, origin, ptFigure):
        if origin == 'bottomright':
            ptFigureRet = (
                1.0 - (ptFigure[0] + self._sizeFigure[0]),
                ptFigure[1])
        elif origin == 'topleft':
            ptFigureRet = (
                ptFigure[0],
                1.0 - (ptFigure[1] + self._sizeFigure[1]))
        elif origin == 'topright':
            ptFigureRet = (
                1.0 - (ptFigure[0] + self._sizeFigure[0]),
                1.0 - (ptFigure[1] + self._sizeFigure[1]))
        else: # if (self._origin == 'bottomleft'):
            ptFigureRet = ptFigure

        return ptFigureRet



    # -----------------------------------------------------------------------------
    # Method _refmodeToStr
    #
    # Converts a RefMode enumerated value to the corresponding keyword string.
    #
    def _refmodeToStr(self, RefMode):
        if RefMode == self.RefMode.FIXED:
            return 'fixed'
        elif RefMode == self.RefMode.FIGURE:
            return 'figure'
        elif RefMode == self.RefMode.EDGE:
            return 'edge'



    # -----------------------------------------------------------------------------
    # _refreshPtInch
    #
    # Updates self.ptInch by converting self._ptFigure to origin-relative figure
    # coordinates.
    #
    def _refreshPtInch(self):
        # Note that self._ptFigure is always relative to the "bottomleft" origin
        # while self._ptInch is always relative to the current origin
        self.ptInch = self.figureToInch(self._convertBetweenOriginBottomLeft(self._ptFigure))



    # -----------------------------------------------------------------------------
    # Method _strToRefMode
    #
    # Converts a string to the corresponding RefMode enumerated value.
    #
    def _strToRefMode(self, str):
        if (str == 'figure') or (str == 'fig'):
            return self.RefMode.FIGURE
        elif (str == 'edge'):
            return self.RefMode.EDGE
        else: # elif (str == 'fixed'):
            return self.RefMode.FIXED



    # -----------------------------------------------------------------------------
    # Method _updateDInchSizeEdge
    #
    # Updates the distances between the element and the figure edge used for the
    # 'edge' positioning and sizing mode.
    def _updateDInchSizeEdge(self):
        if (self._refmodeSize[0] == self.RefMode.EDGE) or (self._refmodeSize[1] == self.RefMode.EDGE):
            sizeInch = self.figureToInch((1, 1))
            self._dinchSizeEdge = ((sizeInch[0] - self._ptInch[0] - self._sizeInch[0], sizeInch[1] - self._ptInch[1] - self._sizeInch[1]))







# -----------------------------------------------------------------------------
# Class UIElementAx
#
# This class wraps a matplotlib axes object into a UIElement, allowing the axes
# to have an absolute size and position.
#
class UIElementAx(UIElement):
    # -----------------------------------------------------------------------------
    # Initialize object instance
    #
    # Arguments:
    #   ax      The element's axes to wrap
    #   sizeInch    Initial size or None for default (inches)
    #   sizeFigure  Initial size or None for default (figure units)
    #
    def __init__(self, ax, sizeInch=None, sizeFigure=None):
        super().__init__(ax.figure, sizeInch=sizeInch, sizeFigure=sizeFigure)

        self.ax = ax # Element's axes we're managing


    # -----------------------------------------------------------------------------
    # commitRect
    #
    # The element is updated with any changes to ptFigure/Inch and sizeFigure/Inch
    #
    def commitRect(self):
        super().commitRect()
        if self.ax is not None:
            self.ax.set_position(
                [
                    self._ptFigure[0],
                    self._ptFigure[1],
                    self._sizeFigure[0],
                    self._sizeFigure[1]
                ])



    def remove(self):
        if self.ax is not None:
            self.ax.remove()
            self.ax = None






# -----------------------------------------------------------------------------
# Class UIElementEx
#
# This class extends UIElementAx by creating an axes object for the caller,
# simplifying UIElementAx creation.
#
class UIElementEx(UIElementAx):
    _serialnumberAxisLabel = 1 # Serial number to make axes unique to prevent matplotlib from reusing existing axes


    # -----------------------------------------------------------------------------
    # Initialize object instance
    #
    # Arguments:
    #   figure  Figure where this UI element is to appear
    #   sizeInch    Initial size or None for default size (inches)
    #   sizeFigure  Initial size or None for default size (figure units)
    #
    def __init__(self, figure, sizeInch=None, sizeFigure=None):
        # Create unique axes for the widget
        snAxisLabel = UIElementEx._serialnumberAxisLabel
        UIElementEx._serialnumberAxisLabel = UIElementEx._serialnumberAxisLabel + 1
        ax = figure.add_axes([0,0,1,1], label='uielementex_'+str(snAxisLabel))

        # Initialize the superclass
        super().__init__(ax, sizeInch, sizeFigure)






# -----------------------------------------------------------------------------
# Class WidgetEx
#
# This class extends UIElement and wraps a matplotlib widget enabling
# positioning and sizing based on inches rather than figure coordinates.  This
# allows the widget to not change size or position as the parent figure is
# resized.
#
# WidgetEx creates the widget's axes for the caller, simplifying widget.
#
class WidgetEx(UIElementEx):
    # -----------------------------------------------------------------------------
    # Initialize object instance
    #
    # Arguments:
    #   figure      Figure where this UI element is to appear
    #   classWidget The class object of the widget to create (not the widget object)
    #   sizeInch    Initial size or None for default (inches)
    #   sizeFigure  Initial size or None for default (figure units)
    #   args        Positional arguments to classWidget without the "ax" axes argument
    #   kwargs      Keyword arguments to classWidget
    #
    def __init__(self, figure, classWidget, *args, sizeInch=None, sizeFigure=None, **kwargs):
        # Initialize the superclass and get a new axes object
        super().__init__(figure, sizeInch=sizeInch, sizeFigure=sizeFigure)

        # Create the widget
        self.widget = classWidget(self.ax, *args, **kwargs)






# -----------------------------------------------------------------------------
# Class ButtonWidgetEx
#
# This class wraps a Button widget as a WidgetEx.  It allows the Button widget
# to have absolute positioning and sizing.
#
# By default the positioning origin is "topleft" which is more suitable for
# generic UI dialogs unlike the matplotlib's usual "bottomleft".
#
class ButtonWidgetEx(WidgetEx):
    # -----------------------------------------------------------------------------
    # Initialize class instance
    #
    # Arguments:
    #   fig         Figure to contain this element
    #   fontsize    Size of font to draw the button's text
    #   label       Text to show in the button
    #   color       Color the button when idle
    #   hovercolor  Color of the button when the cursor is in the button
    #   padInch     Additional space between text and button edges (left, bottom,
    #                   right, top) (inches)
    #   sizeInch    Initial size or None for automatic size (inches)
    #   sizeFigure  Initial size or None for automatic size (figure units)
    #
    def __init__(self, fig, label="", fontsize=8, color='lightgray', hovercolor='lightblue', padInch=(0.1, 0.075, 0.1, 0.075), sizeInch=None, sizeFigure=None, *args, **kwargs):
        super().__init__(fig, matplotlib.widgets.Button, *args, label=label, color=color, hovercolor=hovercolor, **kwargs)

        self.widget.label.set_fontsize(fontsize)
        self.padInch = padInch

        if (sizeInch is None) and (sizeFigure is None):
            self.sizeInch = self.calcButtonSizeInch(label)


    def calcButtonSizeInch(self, str):
        dxInch, dyInch = self.pxToInch(utils.getTextExtentPx(self.fig, str, self.widget.label.get_fontproperties()))
        padInch = self.padInch
        return (dxInch + padInch[0] + padInch[2], dyInch + padInch[1] + padInch[3])

    def calcButtonSizeInch2(self, str):
        dxInch, dyInch = self.pxToInch(utils.getTextExtent2Px(self.fig, str, self.widget.label.get_fontproperties())[0:2])
        padInch = self.padInch
        return (dxInch + padInch[0] + padInch[2], dyInch + padInch[1] + padInch[3])





# -----------------------------------------------------------------------------
# Class TextUIElement
#
# This class wraps a figure Text artist as a UIElement.
#
# By default the positioning origin is "topleft" which is more suitable for
# generic UI dialogs unlike the matplotlib's usual "bottomleft".
#
class TextUIElement(UIElement):
    def __init__(self, fig, strText="", fontsize=8, fontweight='normal', sizeInch=None, sizeFigure=None, **kwargs):
        super().__init__(fig, sizeInch=sizeInch, sizeFigure=sizeFigure)

        # If alignment is not specified, set it to so that the (x, y) position is
        # the bottom-left corner of the bounding box
        if ('va' not in kwargs) and ('verticalalignment' not in kwargs):
            kwargs['verticalalignment'] = 'bottom'
        if ('ha' not in kwargs) and ('horizontalalignment' not in kwargs):
            kwargs['horizontalalignment'] = 'left'

        self.text = matplotlib.text.Text(fontsize=fontsize, fontweight=fontweight, text=strText, **kwargs)
        fig.add_artist(self.text)

        # Set "automatic" size
        if (sizeInch is None) and (sizeFigure is None):
            bbox = self.text.get_window_extent(self.fig.canvas.get_renderer())
            self.sizeInch = self.fig.dpi_scale_trans.inverted().transform((bbox.width, bbox.height))



    def commitRect(self):
        super().commitRect()

        if self.text is not None:
            self.text.set_x(self._ptFigure[0])
            self.text.set_y(self._ptFigure[1])



    # -----------------------------------------------------------------------------
    # _convertBetweenOriginBottomLeft
    #
    # This override adds handling when text vertical alignment is 'baseline'.
    #
    def _convertBetweenOriginBottomLeft(self, ptFigure):
        if (self.text.get_verticalalignment() != 'baseline') or self._origin.startswith('bottom'):
            return super()._convertBetweenOriginBottomLeft(ptFigure)

        # When vertical alignment is 'baseline', ignore the height of the text when calculating the position from the top of the figure
        elif self._origin == 'topright':
            return (
                1.0 - (ptFigure[0] + self._sizeFigure[0]),
                1.0 - ptFigure[1])
        else: # if (self._origin == 'topleft'):
            return (
                ptFigure[0],
                1.0 - ptFigure[1])






# -----------------------------------------------------------------------------
# Class CheckButtonsWidgetEx
#
# This class wraps a CheckButtons widget as a WidgetEx UIElement.  It allows
# the CheckButtons widget to have absolute positioning and sizing as well as
# buttons that remain squares no matter the aspect ratio of the figure.
#
# By default the positioning origin is "topleft" which is more suitable for
# generic UI dialogs unlike the matplotlib's usual "bottomleft".
#
class CheckButtonsWidgetEx(WidgetEx):
    def __init__(self, fig, *args, **kwargs):
        super().__init__(fig, matplotlib.widgets.CheckButtons, *args, **kwargs)

        self.ax.set_frame_on(False)

        # Set our "automatic" size we have the desired button size spacing.
        #
        # The CheckButtons widget spaces the buttons to fit the axes.  We prefer to
        # have a consistent spacing so we'll size the widget axes so that the buttons are
        # spaced the same as four buttons in a one inch tall widget.
        #
        # The width of the widget affects left margin and space between button and text;
        # the height affects the top and bottom margins and vertical space between buttons.
        #
        # With more than one button, the CheckButtons widget divides the axes vertically
        # into (# buttons + 1) rectangles of space with the top and bottom margins using
        # one-half rectangle of space each.
        #
        # With one (or zero) buttons, the CheckButtons widgets assigns the button 1/4 of
        # the vertical space (Why?  I don't know.)
        cbutton = len(self.widget.rectangles)
        if cbutton > 1:
            dxyInch = ((cbutton + 1) / 4)
        else:
            dxyInch = 1
        self.sizeInch = (dxyInch, dxyInch) # Unlike RadioButtons, we can't seem to set the buttons sized directly so we must set the widget size to a square






# -----------------------------------------------------------------------------
# Class RadioButtonsWidgetEx
#
# This class wraps a RadioButtons widget as a WidgetEx UIElement.  It allows
# the RadioButtons widget to have absolute positioning and sizing as well as
# buttons that remain circles no matter the aspect ratio of the figure.
#
# By default the positioning origin is "topleft" which is more suitable for
# generic UI dialogs unlike the matplotlib's usual "bottomleft".
#
class RadioButtonsWidgetEx(WidgetEx):
    def __init__(self, fig, *args, **kwargs):
        super().__init__(fig, matplotlib.widgets.RadioButtons, *args, **kwargs)

        self.sizeInchButton = (0.13, 0.13) # Dimensions of buttons (inches)

        self.ax.set_frame_on(False)

        # Set our "automatic" size we have the desired button size spacing.
        #
        # The RadioButtons widget spaces the buttons to fit the axes.  We prefer to
        # have a consistent spacing so we'll size the widget axes so that the buttons are
        # spaced the same as five buttons in a one inch tall widget.
        #
        # RadioButtons seem to divide the axes vertically into (# buttons + 1) rectangles
        # of space using one-half rectangle of space as top and bottom margins.
        cbutton = len(self.widget.circles)
        self.sizeInch = (1.25, (cbutton + 1) / 5) # Width affects left margin and space between button and text; height affects the top and bottom margins and vertical space between buttons



    def commitRect(self):
        super().commitRect()

        # Set the circle height/width.  Normally they are set in axes coordinates so the
        # circle distorts with as the axes and figure change size.  We'll set them to
        # dimensions specified in inches by calculating the mapping from inches to axes
        # coordinates.
        if (self.fig is not None) and (self.ax is not None):
            sizeFigure = self.inchToFigure(self.sizeInchButton)
            bboxAxes = self.ax.get_position()
            dxCircle =  sizeFigure[0] / bboxAxes.width
            dyCircle =  sizeFigure[1] / bboxAxes.height
            for circle in self.widget.circles:
                circle.width = dxCircle
                circle.height = dyCircle






# -----------------------------------------------------------------------------
# Class ScrollBar
#
# This class wraps a ScrollBar widget as a WidgetEx UIElement.  It allows
# the widget to have absolute positioning and sizing.
#
# By default the positioning origin is "topleft" which is more suitable for
# generic UI dialogs unlike the matplotlib's usual "bottomleft".
#
class ScrollBar(WidgetEx):
    dinchWidthDefault = 0.15 # Default scrollbar widget (inches)


    def __init__(self, fig, *args, **kwargs):
        super().__init__(fig, guiwidgets.ScrollBar, *args, **kwargs)

        if (self.widget.orientation == 'vertical'):
            self.sizeFigure = (self.inchToFigure((self.dinchWidthDefault, 0))[0], 1)
        else:
            self.sizeFigure = (1, self.inchToFigure((0, self.dinchWidthDefault))[1])





# -----------------------------------------------------------------------------
# Class ComboBox
#
# This class implements a drop-down list box to allow selection of one of
# multiple values.
#
class ComboBox(UIElement):
    class EventChangedArgs:
        def __init__(self, index, label):
            self.index = index
            self.label = label


    @property
    def dropdown_is_open(self):
        return self._listbox is not None



    @property
    def index_selected(self):
        return self._iitemSelected

    @index_selected.setter
    def index_selected(self, index):
        self.closeDropDown()
        if (index is None) or (index < 0):
            self._iitemSelected = None
            self._buttonwidgetexSelected.widget.label.set_text("")
            self.fig.canvas.draw_idle()
        elif (index < len(self._rgstrItem)):
            self._iitemSelected = index
            self._buttonwidgetexSelected.widget.label.set_text(self._rgstrItem[index])
            self.fig.canvas.draw_idle()
        elif (self._rgstrItem is None) or (len(self._rgstrItem) < 1):
            raise ValueError("index value {0} can only be None when there are no items".format(index))
        else:
            raise ValueError("index value {0} is not None nor less than the number of items, {2}".format(index, len(self._rgstrItem) - 1))


    @property
    def item_selected(self):
        return self._rgstrItem[self._iitemSelected] if self._iitemSelected >= 0 else None


    @property
    def items(self):
        return self._rgstrItem

    @items.setter
    def items(self, items):
        if items is not list:
            items = list(items)

        self.closeDropDown()
        self._rgstrItem = items
        self.index_selected = None



    def __init__(self, figParent, items=None, index_selected=None, sizeInch=None, sizeFigure=None, color='lightgray', hovercolor='lightblue'):
        super().__init__(figParent, sizeInch=sizeInch, sizeFigure=sizeFigure)

        # Initialize data
        self._callbackregistry = matplotlib.cbook.CallbackRegistry() # Functions connected to our generated events
        self._iitemSelected = None # Index into self._rgstrItem of the currently selected item or None if none is selected
        self._listbox = None # UI element for drop-down listbox
        self._rgstrItem = None # List of value strings

        # Create selected item and open-drop-down button
        self._buttonwidgetexSelected = ButtonWidgetEx(figParent, "", color=color, hovercolor=hovercolor)
        self._buttonwidgetexSelected.widget.on_clicked(self.onButton_Clicked)
        self._buttonwidgetexSelected.origin = 'bottomleft' # Set to bottom left so it's easy to position at self._ptFigure

        # Set default size to fit the largest value button
        if (sizeInch is None) and (sizeFigure is None):
            dxInch, dyInch = self._buttonwidgetexSelected.sizeInch
            dyInchDescenders = self._buttonwidgetexSelected.calcButtonSizeInch2("GgJjPpQqYy09")[1] # Calculate minimum button height of text with descenders
            if dyInchDescenders > dyInch:
                dyInch = dyInchDescenders

            if items is not None:
                for str in items:
                    size = self._buttonwidgetexSelected.calcButtonSizeInch2(str)
                    if size[0] > dxInch:
                        dxInch = size[0]
                    if size[1] > dyInch:
                        dyInch = size[1]

            self.sizeInch = (dxInch, dyInch)

        # Setup the items
        self.items = items
        self.index_selected = index_selected

        # Connect to mouse button press events
        self.fig.canvas.mpl_connect('button_press_event', self.onEvent_ButtonPress)



    def closeDropDown(self):
        if self._listbox is not None:
            self._listbox.remove()
            self._listbox = None
            self.fig.canvas.draw()



    def onButton_Clicked(self, event):
        if self._listbox is None:
            self.openDropDown()
        else:
            self.closeDropDown()



    def openDropDown(self):
        if self._listbox is None:
            # Create UI element that defines the drop-line list and provides an bordered background
            self._listbox = ListBox(self.fig, self._rgstrItem, index_selected=self._iitemSelected)
            self._listbox.origin = 'bottomleft'
            self._listbox.on_changed(self.onListBox_Changed)
            self._sizeInchListFull = self._listbox.sizeInch # Height of list box so that scrolling is not needed

            self.layoutChildren() # Position and size the list box
            self._listbox.scrollToSelected() # Make sure the selected item is visible
            self._listbox.commitRect() # Update the list box axes

            self.fig.canvas.draw() # Draw now to display the list box as quickly as possible



    def onEvent_ButtonPress(self, event):
        # If click didn't happen inside one of our axes, close the drop-down panel
        if self.dropdown_is_open \
                and (event.inaxes != self._buttonwidgetexSelected.ax) \
                and ((self._listbox is None) or not self._listbox.ownsAxes(event.inaxes)):
            self.closeDropDown()



    def commitRect(self):
        super().commitRect()

        self.layoutChildren()

        self._buttonwidgetexSelected.commitRect()
        if (self._listbox is not None):
            self._listbox.commitRect()



    def disconnect(self, connectionid):
        self._callbackregistry.disconnect(connectionid)



    def layoutChildren(self):
        self._buttonwidgetexSelected.ptFigure = self._ptFigure
        self._buttonwidgetexSelected.sizeFigure = self._sizeFigure

        if self._listbox is not None:
            y = self._ptFigure[1]
            dyList = self.inchToFigure(self._sizeInchListFull)[1]

            # Is there enough room below us so that the list doesn't extend beyond the edge of the figure?
            dyAvailableBelow = y
            if (dyList <= dyAvailableBelow):
                # There's room for the full list below, position there
                self._listbox.ptFigure = (self._ptFigure[0], self._ptFigure[1] - dyList)
            else:
                # Not enough room below us, is there more room above?
                dy = self._sizeFigure[1]
                dyAvailableAbove = 1.0 - y - dy
                if (dyAvailableAbove > dyAvailableBelow):
                    # Yup, position the list above
                    self._listbox.ptFigure = (self._ptFigure[0], self._ptFigure[1] + dy)
                    if (dyAvailableAbove < dyList):
                        dyList = dyAvailableAbove
                else:
                    # Nope, position the list below
                    self._listbox.ptFigure = (self._ptFigure[0], self._ptFigure[1] - dyAvailableBelow)
                    dyList = dyAvailableBelow

            # Resize the list if needed
            if (abs(self._listbox.sizeFigure[1] - dyList) > sys.float_info.epsilon):
                self._listbox.sizeFigure = (self._listbox.sizeFigure[0], dyList)



    def on_changed(self, fnCallback):
        return self._callbackregistry.connect('changed', fnCallback)



    def onListBox_Changed(self, event):
        self.closeDropDown()
        self._setSelectedIndex(event.index)



    def recalcRect(self):
        super().recalcRect()

        self._buttonwidgetexSelected.recalcRect()
        if (self._listbox is not None):
            self._listbox.recalcRect()



    def _setSelectedIndex(self, index):
        # Handle the selected item if it's different from the current selection
        if (index == None) or (index < 0) or (index > len(self._rgstrItem)):
            self._iitemSelected = None
            strValue = None
            self._buttonwidgetexSelected.widget.label.set_text("")
        else:
            self._iitemSelected = index
            strValue = self._rgstrItem[index]
            self._buttonwidgetexSelected.widget.label.set_text(strValue)

        self.fig.canvas.draw()

        # Trigger the Changed event
        self._callbackregistry.process('changed', ComboBox.EventChangedArgs(index, strValue))






# -----------------------------------------------------------------------------
# Class ListBox
#
# This class implements a list box to allow selection of one of multiple values.
#
class ListBox(UIElementEx):
    class EventChangedArgs:
        def __init__(self, index, label):
            self.index = index
            self.label = label

    class ButtonSpoofEvent:
        def __init__(self):
            self.inaxes = None



    @property
    def index_selected(self):
        if self._rgiitemSelected is None or not self._rgiitemSelected:
            return None
        return self._rgiitemSelected[0]

    @index_selected.setter
    def index_selected(self, index):
        self.setSelectedToNormal()
        if (index is None) or (index < 0):
            self._rgiitemSelected = None
        elif (index < len(self._rgstrItem)):
            self._rgiitemSelected = [index]
            self.setSelectedToSelected()
        elif (self._rgstrItem is None) or (len(self._rgstrItem) < 1):
            raise ValueError("index can only be None when there are no items")
        else:
            raise ValueError("index value {0} is not None nor less than the number of items, {2}".format(index, len(self._rgstrItem) - 1))


    @property
    def indices_selected(self):
        return self._rgiitemSelected

    @index_selected.setter
    def indices_selected(self, indices):
        self.setSelectedToNormal()
        citem = len(self._rgstrItem)
        if (indices is None) or (len(indices) < 1):
            self._rgiitemSelected = None
        elif (self._rgstrItem is None) or (len(self._rgstrItem) < 1):
            raise ValueError("indice can only be None when there are no items")
        else:
            for index in indices:
                if (index < 0) or (index > citem):
                    raise ValueError("index value {0} is None or greater than the last valid index, {2}".format(index, len(self._rgstrItem) - 1))
            self._rgiitemSelected = sorted(indices)
            self.setSelectedToSelected()


    @property
    def item_selected(self):
        index = self.index_selected
        return self._rgstrItem[index] if index is not None else None


    @property
    def items_selected(self):
        return [self._rgstrItem[i] for i in self._rgiitemSelected] if self._rgiitemSelected is not None else None


    @property
    def items(self):
        return self._rgstrItem

    @items.setter
    def items(self, items):
        if items is not list:
            items = list(items)

        self._rgstrItem = items

        self.index_selected = None
        self.updateButtonText()
        self.recalcRect() # Relayout with the changed buttons
        self.commitRect()


    @property
    def enable_toggle(self):
        return self._fEnableToggle

    @enable_toggle.setter
    def can_toggle_selected(self, fEnableToggle):
        self._fEnableToggle = fEnableToggle



    def __init__(self, figParent, items=None, index_selected=None, color='lightgoldenrodyellow', hovercolor='lightblue', selected_color='lightcoral', selected_hovercolor='lightpink'):
        super().__init__(figParent)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Initialize data
        self.color = color
        self.hovercolor = hovercolor
        self.selected_color = selected_color
        self.selected_hovercolor = selected_hovercolor

        # Sythesize the empty panel color from the main color
        rgba = matplotlib.colors.to_rgba(color)
        hsv = matplotlib.colors.rgb_to_hsv(rgba[0:3])
        s = hsv[1] * 0.75
        v = hsv[2] * 1.25
        if v > 1:
            v = 1.0
        self.empty_color = matplotlib.colors.to_rgba(matplotlib.colors.hsv_to_rgb((hsv[0], s, v)), rgba[3])

        self._buttonspoofevent = ListBox.ButtonSpoofEvent() # Event used to trigger buttons to update immediately
        self._callbackregistry = matplotlib.cbook.CallbackRegistry() # Subscribers to our generated events
        self._fEnableToggle = False # If true, clicking on a item toggles whether it's selected
        self._iitemTop = 0 # Index of the item visible at the top of the list
        self._rgbuttonwidgetex = None # Widget for each value in the drop-down
        self._rgiitemSelected = None # List of indices into self._rgstrItem of the currently selected item
        self._rgstrItem = [] # List of value strings
        self._scrollbarwidget = None # UI element for drop-down list panel
        self._sizeInchButton = (0.25, 0.25) # Default minimum button size

        # Add to provide a bordered background
        self.ax.add_artist(matplotlib.patches.Rectangle((0.0, 0.0), 1.0, 1.0, color=self.empty_color, fill=True, zorder=1))

        # Determine the largest value button
        button = ButtonWidgetEx(self.fig, "")
        dxInchButton, dyInchButton = button.sizeInch
        dyInchDescenders = button.calcButtonSizeInch2("GgJjPpQqYy09")[1] # Calculate minimum button height of text with descenders
        if dyInchDescenders > dyInchButton:
            dyInchButton = dyInchDescenders

        if items is None:
            citem = 0
        else:
            citem = len(items)
            for str in items:
                size = button.calcButtonSizeInch(str)
                if size[0] > dxInchButton:
                    dxInchButton = size[0]
                if size[1] > dyInchButton:
                    dyInchButton = size[1]

        button.remove()
        button = None
        self._sizeInchButton = (dxInchButton, dyInchButton)

        # Set default size to fit all of the items using the largest value button
        margin = self.pxToInch((1, 1)) # One pixel-widge margin to prevent buttons from drawing over the border
        self.sizeInch = (dxInchButton, citem * dyInchButton + margin[1])

        # Setup the items
        if items is not list:
            self._rgstrItem = list(items)
        else:
            self._rgstrItem = items
        self.index_selected = index_selected

        # Attach to mouse motion events so we can track which axes contains the mouse
        self.fig.canvas.mpl_connect('motion_notify_event', self.onCanvas_MotionNotify)

        # Do layout now to create the item buttons
        self.recalcRect()
        self.commitRect()




    def commitRect(self):
        super().commitRect()

        self.layoutChildren()

        if (self._scrollbarwidget is not None):
            self._scrollbarwidget.commitRect()
        if self._rgbuttonwidgetex is not None:
            for bwex in self._rgbuttonwidgetex:
                bwex.commitRect()


    def createValueButton(self, str):
        buttonwidget = ButtonWidgetEx(self.fig, str, color=self.color, hovercolor=self.hovercolor, sizeFigure=(0, 0))
        buttonwidget.origin = 'bottom-left'
        for key, spine in buttonwidget.ax.spines.items():
            spine.set_alpha(0.0) # Turn off the button border
        buttonwidget.widget.on_clicked(self.onButtonValue_Clicked)
        return buttonwidget


    def disconnect(self, connectionid):
        self._callbackregistry.disconnect(connectionid)



    def find_item(self, item):
        index = None
        for i in range(len(self._rgstrItem)):
            if item == self._rgstrItem[i]:
                index = i
                break

        return index


    def layoutChildren(self):
        margin = self.fig.transFigure.inverted().transform((1, 1)) # One pixel-widge margin to prevent buttons from drawing over the border

        # Determine how many buttons will fit
        sizeFigureButton = [self._sizeFigure[0] - margin[0], self.inchToFigure(self._sizeInchButton)[1]]
        dyButton = sizeFigureButton[1]
        cbuttonvisibleFloat = ((self.sizeFigure[1] - margin[1]) / dyButton)

        if cbuttonvisibleFloat > 1:
            cbuttonvisible = int(cbuttonvisibleFloat)
        else:
            cbuttonvisible = 1
            dyButton = self.sizeFigure[1] - margin[1]
            sizeFigureButton[1] = dyButton

        with ExitStack() as exitstack:
            # Disable unnecessary button redraw while we rearrange things
            if self._rgbuttonwidgetex is not None:
                for bwex in self._rgbuttonwidgetex:
                    exitstack.enter_context(matplotlib.cbook._setattr_cm(bwex, drawon=False))

            # Show or hide the scrollbar as needed
            if (self._rgstrItem is None) or (cbuttonvisible >= len(self._rgstrItem)):
                if self._scrollbarwidget is not None:
                    self._scrollbarwidget.ax.remove()
                    self._scrollbarwidget = None

                if self._iitemTop > 0:
                    self.scrollToIndex(0)
            else:
                if self._scrollbarwidget is None:
                    # Create the list scrollbar
                    cbutton = len(self._rgstrItem) if self._rgstrItem is not None else 0
                    self._scrollbarwidget = ScrollBar(self.fig, orientation='vertical', value=self._iitemTop, value_range = (0, cbutton), value_page=cbuttonvisible)
                    self._scrollbarwidget.origin = 'bottom-left'
                    self._scrollbarwidget.widget.on_changed(self.onScrollBar_Changed)

                    exitstack.enter_context(matplotlib.cbook._setattr_cm(self._scrollbarwidget, drawon=False))

                # Shrink buttons to make room for the scrollbar
                sizeFigureButton[0] -= self._scrollbarwidget.sizeFigure[0]


            # Resize the button list to fit the available space
            if self._rgstrItem is None:
                self._rgbuttonwidgetex = None
            else:
                citem = len(self._rgstrItem) - self._iitemTop
                citem = min(citem, cbuttonvisible)
                cbutton = len(self._rgbuttonwidgetex) if self._rgbuttonwidgetex is not None else 0
                if citem < cbutton:
                    # Too many buttons, remove the extra
                    cbuttonToDelete = cbutton - citem
                    for i in range(cbutton - 1, citem - 1, -1):
                        self._rgbuttonwidgetex[i].remove()
                    self._rgbuttonwidgetex = self._rgbuttonwidgetex[0:citem]
                elif citem > cbutton:
                    # Not enough buttons, add some more
                    if self._rgbuttonwidgetex is None:
                        self._rgbuttonwidgetex = []
                    for i in range(cbutton, citem):
                        buttonwidget = self.createValueButton("")
                        self._rgbuttonwidgetex.append(buttonwidget)
                    self.updateButtonText()

                    # If one of the newly added buttons is a selected item, set the button color
                    if self._rgiitemSelected is not None:
                        for i in range(self._iitemTop + cbutton, self._iitemTop + citem):
                            if i in self._rgiitemSelected:
                                self.setSelectedToSelected()
                                break

            # Layout the buttons
            ptFigure = self._ptFigure
            ptFigure = [ptFigure[0] + margin[0], ptFigure[1] + self.sizeFigure[1] - margin[1]] # The margin stops the buttons from hiding the left and top of the list outline

            if self._rgbuttonwidgetex is not None:
                cbutton = len(self._rgbuttonwidgetex)

                # Position the buttons
                with ExitStack() as exitstack:
                    for bwex in self._rgbuttonwidgetex:
                        exitstack.enter_context(matplotlib.cbook._setattr_cm(bwex, drawon=False))

                        ptFigure[1] -= dyButton
                        bwex.ptFigure = (ptFigure[0], ptFigure[1])
                        bwex.sizeFigure = sizeFigureButton

            # Position scrollbar
            if self._scrollbarwidget is not None:
                self._scrollbarwidget.sizeFigure = (self._scrollbarwidget.sizeFigure[0], self.sizeFigure[1])
                self._scrollbarwidget.ptFigure = (self._ptFigure[0] + self._sizeFigure[0] - self._scrollbarwidget.sizeFigure[0], self._ptFigure[1])
                self._scrollbarwidget.widget.value_page = cbuttonvisible if cbuttonvisible > 0 else 1

        self.fig.canvas.draw_idle()


    def on_changed(self, fnCallback):
        return self._callbackregistry.connect('changed', fnCallback)



    def onButtonValue_Clicked(self, event):
        iitem = None
        widget = None

        # Find the item that was clicked
        if self._rgbuttonwidgetex is not None:
            for i in range(len(self._rgbuttonwidgetex)):
                buttonwidgetex = self._rgbuttonwidgetex[i]
                if event.inaxes == buttonwidgetex.ax:
                    iitem = i + self._iitemTop
                    widget = buttonwidgetex.widget
                    break

        # Report the selected item
        if widget is not None:
            if (self._rgiitemSelected is not None) and (iitem in self._rgiitemSelected):
                if self._fEnableToggle:
                    self._rgiitemSelected.remove(iitem)
                    self.setButtonNormal(iitem)
            else:
                # Unhighlight the previous selection
                self.setSelectedToNormal()

                self._rgiitemSelected = [iitem]

                # Get text of and highlight the new selection
                if iitem is None:
                    strValue = None
                else:
                    strValue = widget.label.get_text()
                    self.setSelectedToSelected()

                # Trigger the Changed event
                self._callbackregistry.process('changed', ComboBox.EventChangedArgs(iitem, strValue))



    def onCanvas_MotionNotify(self, event):
        self._buttonspoofevent.inaxes = event.inaxes


    def onScrollBar_Changed(self, event):
        iitemTop = int(event + 0.5)
        self.scrollToIndex(iitemTop)



    def ownsAxes(self, ax):
        if (ax == self.ax) or ((self._scrollbarwidget is not None) and (ax == self._scrollbarwidget.ax)):
            return True
        elif self._rgbuttonwidgetex is not None:
            for widget in self._rgbuttonwidgetex:
                if ax == widget.ax:
                    return True

        return False



    def recalcRect(self):
        super().recalcRect()

        if self._scrollbarwidget is not None:
            self._scrollbarwidget.recalcRect()

        if self._rgbuttonwidgetex is not None:
            for bwex in self._rgbuttonwidgetex:
                bwex.recalcRect()



    def remove(self):
        if self._rgbuttonwidgetex is not None:
            for bwex in self._rgbuttonwidgetex:
                bwex.remove()
            self._rgbuttonwidgetex = None

        if self._scrollbarwidget is not None:
            self._scrollbarwidget.remove()
            self._scrollbarwidget = None

        super().remove()



    def scrollToIndex(self, index):
        citem = len(self._rgstrItem)
        if index < 0:
            index = 0
        elif index >= citem:
            index = citem - 1

        if (index < citem) and (index != self._iitemTop):
            with ExitStack() as exitstack:
                if self._rgbuttonwidgetex is not None:
                    for bwex in self._rgbuttonwidgetex:
                        exitstack.enter_context(matplotlib.cbook._setattr_cm(bwex, drawon=False))

                self._iitemTop = index
                if self._scrollbarwidget is not None:
                    exitstack.enter_context(matplotlib.cbook._setattr_cm(self._scrollbarwidget, drawon=False))
                    self._scrollbarwidget.widget.value = index
                self.updateButtonText()

                self.setSelectedToSelected()

                self.recalcRect()
                self.commitRect()

            self.fig.canvas.draw()


    def scrollToIndexNearest(self, index):
        if index is not None:
            if index < self._iitemTop:
                # Item is above the visible items, put item at the top
                self.scrollToIndex(index)
            else:
                # Item is below the visible items, put item at the bottom
                margin = self.pxToInch((1, 1)) # One pixel-widge margin to prevent buttons from drawing over the border
                cbuttonVisible = int((self.sizeInch[1] - margin[1]) / self._sizeInchButton[1])
                if cbuttonVisible < 1:
                    cbuttonVisible = 1
                if (index >= self._iitemTop + cbuttonVisible):
                    self.scrollToIndex(index - cbuttonVisible + 1)



    def scrollToSelected(self):
        self.scrollToIndexNearest(self.index_selected)



    def setButtonColor(self, ibutton, strColor, strColorHover):
        if self._rgbuttonwidgetex is not None:
            ibutton -= self._iitemTop
            if (ibutton >= 0) and (ibutton < len(self._rgbuttonwidgetex)):
                widget = self._rgbuttonwidgetex[ibutton].widget
                widget.color = strColor
                widget.hovercolor = strColorHover

                widget._motion(self._buttonspoofevent) # Force the button to change color now


    def setButtonNormal(self, ibutton):
        self.setButtonColor(ibutton, self.color, self.hovercolor)



    def setButtonSelected(self, ibutton):
        self.setButtonColor(ibutton, self.selected_color, self.selected_hovercolor)



    def setSelectedToNormal(self):
        if self._rgiitemSelected is not None:
            for i in self._rgiitemSelected:
                self.setButtonNormal(i)


    def setSelectedToSelected(self):
        if self._rgiitemSelected is not None:
            for i in self._rgiitemSelected:
                self.setButtonSelected(i)


    def updateButtonText(self):
        if (self._rgstrItem is not None) and (self._rgbuttonwidgetex is not None):
            cstr = len(self._rgstrItem)
            for i in range(len(self._rgbuttonwidgetex)):
                istr = i + self._iitemTop
                if istr >= cstr:
                    break

                str = self._rgstrItem[istr]
                self._rgbuttonwidgetex[i].widget.label.set_text(str)

