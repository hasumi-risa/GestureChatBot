# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import sys
import os
import threading
import argparse
import glob
import cv2

import settings
import labanotation

# -----------------------------------------------------------------------------
#
class application:
    #------------------------------------------------------------------------------
    # Class initialization
    #
    def __init__(self):
        self.files = []
        self.labanotation = None

        # print('\033[4m\033[1m' + 'Labanotation ' + settings.appVersion + '\033[0m')
        print('Labanotation to PNG converter ' + settings.appVersion + '\r\n')

        self.labanotation = labanotation.Labanotation(self)

    #------------------------------------------------------------------------------
    #
    def readCmdLine(self):
        parser = argparse.ArgumentParser(description='Labanotation to PNG converter.')
        parser.add_argument('--input', help='Laban input file')
        parser.add_argument('--width', default=768, help='width of output image. Default is 768 pixels.')

        self.cmdArgs = parser.parse_args()

        if (self.cmdArgs.input == None):
            print 'Please specify file(s) to convert.'
            exit()

        # print(self.cmdArgs.input)

        self.files = glob.glob(self.cmdArgs.input)

    # -----------------------------------------------------------------------------
    # determine input and various output file paths
    #
    def getOutputFilename(self, filename):
        # determine absolute input file path
        if os.path.isabs(filename):
            inputFilePath = filename
        else:
            inputFilePath = os.path.join(settings.cwd, self.inputName)

        # remove file extension from inputName, if any
        inputNameSplit = os.path.splitext(inputFilePath)
        if  inputNameSplit[1] != '':
            outputName = inputNameSplit[0] + '.png'
        else:
            outputName = inputFilePath + '.png'

        return outputName

    #------------------------------------------------------------------------------
    #
    def run(self):
        self.readCmdLine()

        for filepath in self.files:
            print("Loading '" + filepath + "'...")
            success = self.labanotation.loadGestureFile(filepath)
            if (success is True):
                img = self.labanotation.convertToImage(self.cmdArgs.width)
                if (img is not None):
                    outputFilepath = self.getOutputFilename(filepath)

                    #filePath = settings.application.outputFilePathImg
                    #filePath = settings.checkFileAlreadyExists(filePath, fileExt=".png", fileTypes=[('png files', '.png'), ('all files', '.*')])

                    try:
                        cv2.imwrite(outputFilepath, img)
                        print("    Labanotation score image was saved to '" + settings.beautifyPath(outputFilepath) + "'")
                    except Exception as e:
                        print("    Exception saving Labanotation score image to '" + settings.beautifyPath(outputFilepath) + "': " + str(e))
                else:
                    print("    Failed to convert to image.")

#------------------------------------------------------------------------------
#
# main code
#
if __name__ == '__main__':
    # initialize global variables
    settings.initialize()

    # create main application object, then run it
    settings.application = application()
    settings.application.run()
