# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

class CoordConvert:
    # -----------------------------------------------------------------------------
    # figureToInch
    #
    # Converts dimensions in figure coordinates to inches
    #
    # Arguments:
    #   pairFigureValue  2-tuple with figure coordinate dimensions to convert
    #   fig              pairFigureValue specifies value this figure's coordinates
    #
    # Returns:
    #   (Return)         2-tuple with equivalent dimension in inches
    def figureToInch(pairFigureValue, fig):
        return (fig.dpi_scale_trans.inverted().transform(fig.transFigure.transform(pairFigureValue)))



    # -----------------------------------------------------------------------------
    # figureToPx
    #
    # Converts dimensions in figure coordinates to pixels
    #
    # Arguments:
    #   pairFigureValue  2-tuple with figure coordinate dimensions to convert
    #   fig              pairFigureValue specifies value this figure's coordinates
    #
    # Returns:
    #   (Return)         2-tuple with equivalent dimension in pixels
    def figureToPx(pairFigureValue, fig):
        return (fig.transFigure.transform(pairFigureValue))



    # -----------------------------------------------------------------------------
    # inchToFigure
    #
    # Converts dimensions in inches to figure coordinates
    #
    # Arguments:
    #   pairInchValue  2-tuple with dimensions in inches to convert
    #   fig            We want the equivalent in this figure's coordinates
    #
    # Returns:
    #   (Return)       2-tuple with equivalent dimension in figure coordinates
    def inchToFigure(pairInchValue, fig):
        return (fig.transFigure.inverted().transform(fig.dpi_scale_trans.transform(pairInchValue)))



    # -----------------------------------------------------------------------------
    # inchToPx
    #
    # Converts dimensions in inches to pixels
    #
    # Arguments:
    #   pairInchValue  2-tuple with dimensions in inches to convert
    #   fig            Use this figure's DPI
    #
    # Returns:
    #   (Return)       2-tuple with equivalent dimension in pixels
    def inchToPx(pairInchValue, fig):
        return (fig.dpi_scale_trans.transform(pairInchValue))



    # -----------------------------------------------------------------------------
    # pxToFigure
    #
    # Converts dimensions in pixels to figure coordinates
    #
    # Arguments:
    #   pairPxValue  2-tuple with dimensions in inches to convert
    #   fig          We want the equivalent in this figure's coordinates
    #
    # Returns:
    #   (Return)     2-tuple with equivalent dimension in figure coordinates
    def pxToFigure(pairPxValue, fig):
        return (fig.transFigure.inverted().transform(pairPxValue))



    # -----------------------------------------------------------------------------
    # pxToInch
    #
    # Converts dimensions in pixels to inches
    #
    # Arguments:
    #   pairInchValue  2-tuple with dimensions in pixels to convert
    #   fig            Use this figure's DPI
    #
    # Returns:
    #   (Return)       2-tuple with equivalent dimension in pixels
    def pxToInch(pairPxValue, fig):
        return (fig.dpi_scale_trans.inverted().transform(pairPxValue))
