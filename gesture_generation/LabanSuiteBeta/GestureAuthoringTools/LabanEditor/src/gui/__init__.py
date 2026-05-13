# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------


import copy
import matplotlib.widgets

from . import cursor


# --------------------------------------------------------------------------------------------
# Class Cursor
#
# The built-in Cursor class only handles one axes at a time and doesn't work properly
# when there are overlapping axes like a second axis twinned to the first.
# --------------------------------------------------------------------------------------------
class Cursor(cursor.Cursor):
    pass

