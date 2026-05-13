# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import os,sys

import tkFileDialog
import tkMessageBox

# -----------------------------------------------------------------------------
#
def initialize():
    global appVersion           # application version
    global fVerbose             # verbose console output
    global application          # application object
    global cwd                  # current working directory

    if (os.name == 'nt'):
        os.system('color')  # needed on windows platforms to support terminal colors

    appVersion = 'v1.00.0100'
    fVerbose = False
    application = None

    cwd = os.getcwd()

    # If we're in the "src" directory, use the parent directory as the base path
    # for files to make specifying paths a little easier for the user
    if (cwd[-4:] == '\\src') or (cwd[-4:] == '/src'):
        cwd = cwd[0:-4]
 
# -----------------------------------------------------------------------------
# when possible, create a relative sub_path for path from main_path
#
def getRelativePath(main_path, sub_path):
    if sys.platform == "win32":
        main_path = main_path.lower()
        sub_path = sub_path.lower()
    _main_path = os.path.abspath(main_path).split(os.path.sep)
    _sub_path = os.path.abspath(sub_path).split(os.path.sep)
    eq_until_pos = None
    for i in xrange(min(len(_main_path), len(_sub_path))):
        if _main_path[i] == _sub_path[i]:
            eq_until_pos = i
        else:
            break
    if eq_until_pos is None:
        return sub_path
    newpath = [".." for i in xrange(len(_main_path[eq_until_pos+1:]))]
    newpath.extend(_sub_path[eq_until_pos+1:])
    return os.path.join(*newpath) if newpath else "."

# -----------------------------------------------------------------------------
#
def beautifyPath(strPath):
    splitOutput = os.path.split(os.path.abspath(strPath))
    relativePath = getRelativePath(cwd, splitOutput[0])
    return os.path.join(relativePath, splitOutput[1])

# -----------------------------------------------------------------------------
# check whether file already exists, and present action dialog if necessary
#
def checkFileAlreadyExists(filePath, fileExt=".txt", fileTypes=[('text files', '.txt'), ('all files', '.*')]):
    options = {}

    if (tkGuiCanvas is not None):
        options['parent'] = tkGuiCanvas

    if (not os.path.isfile(filePath)):
        return filePath

    result = tkMessageBox.askyesnocancel("Microsoft", "The file '" + filePath + "' already exists.\r\n\r\n\tChoose 'Yes' to overwrite.\r\n\tChoose 'No' to choose a new file name.\r\n\tChoose 'Cancel' to cancel.\r\n", **options)
    if (result is True):
        return filePath
    elif (result is None):
        return None

    # define options for dialog
    splitInput = os.path.split(os.path.abspath(filePath))
    if  splitInput[1] != '':
        fileName = splitInput[1]
    else:
        fileName = splitInput[0]

    options = {}
    options['defaultextension'] = fileExt
    options['filetypes'] = fileTypes
    options['initialdir'] = os.path.dirname(filePath)
    options['initialfile'] = fileName
    options['title'] = "Labanotation"

    if (tkGuiCanvas is not None):
        options['parent'] = tkGuiCanvas

    response = tkFileDialog.asksaveasfilename(**options)
    if (response is ''):
        return None

    return response
