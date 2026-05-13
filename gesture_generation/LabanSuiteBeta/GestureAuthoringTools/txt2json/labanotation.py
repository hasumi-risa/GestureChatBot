# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import sys
import os
import json, operator
import numpy as np

import json
from collections import OrderedDict

# -----------------------------------------------------------------------------
#
class Labanotation:
    # -----------------------------------------------------------------------------
    #
    def __init__(self, application):
        if (application is None):
            print("ERROR in Labanotation.init(): application object expected.")

        self.application = application

    # -----------------------------------------------------------------------------
    #
    def close(self):
        pass

    #------------------------------------------------------------------------------
    #
    def resetFrameValues(self, laban, empty=True):
        laban['start time']  = "" if empty else [ "0" ]
        laban['duration']    = "" if empty else [ "0" ]
        laban['head']        = "" if empty else [ "Forward", "Normal" ]
        laban['right elbow'] = "" if empty else [ "Place", "Low" ]
        laban['right wrist'] = "" if empty else [ "Place", "Low" ]
        laban['left elbow']  = "" if empty else [ "Place", "Low" ]
        laban['left wrist']  = "" if empty else [ "Place", "Low" ]
        laban['rotation']    = "" if empty else [ "ToRight", "0" ]

    #------------------------------------------------------------------------------
    #
    def checkDirLv(self, key, frame, previous_frame):
        pose = frame[key]
        previous_pose = previous_frame[key]

        if (len(pose) != 2):
            print("    Pose for '" + key + "' has " + str(len(pose)) + " entries instead of expected 2 (" + str(pose) + "). Reseting to previous.")
            frame[key] = previous_frame[key][:]
        else:
            dir = pose[0].lower()
            lv = pose[1].lower() 

            if ((lv != "high") and (lv != "normal") and (lv != "low")):
                print("    Unknown level '" + lv + "'")

            if ((dir != "forward") and (dir != "right forward") and (dir != "right") and (dir != "right backward") and (dir != "backward") and (dir != "left backward") and (dir != "left") and (dir != "left forward") and (dir != "place")):
                print("    Unknown direction '" + dir + "'")

            if (dir == "place"):
                if ((lv != "high") and (lv != "low")):
                    print ("    Unknown place for '" + lv + "'");

    #------------------------------------------------------------------------------
    #
    def checkRotation(self, key, frame, previous_frame):
        pass

    #------------------------------------------------------------------------------
    #
    def checkFrameValidity(self, laban, frame, previous_frame, position):
        for key in frame:
            value = frame[key]
            if (value == ""):
                print("    Missing information for key '" + key + "' in frame with 'Start Time'=" + str(frame['Start Time']) + ", position " + position + ". Using previous frame's value of '" + str(previous_frame[key]) + "'")
                frame[key] = previous_frame[key][:]

        self.checkDirLv('head', frame, previous_frame)
        self.checkDirLv('right elbow', frame, previous_frame)
        self.checkDirLv('right wrist', frame, previous_frame)
        self.checkDirLv('left elbow', frame, previous_frame)
        self.checkDirLv('left wrist', frame, previous_frame)
        self.checkRotation('rotation', frame, previous_frame)

    #------------------------------------------------------------------------------
    #
    def appendFrame(self, laban, frame, previous_frame, position):
        #
        # before adding, check for missing entries
        self.checkFrameValidity(laban, frame, previous_frame, position)
        laban[position] = frame

    #------------------------------------------------------------------------------
    #
    def convert(self, input, output):
        splitOutput = os.path.split(os.path.abspath(input))

        # remove file extension from inputName, if any
        filename = splitOutput[1]
        inputNameSplit = os.path.splitext(filename)
        if  inputNameSplit[1] != '':
            filename = inputNameSplit[0]

        readfile = open(input, "r")

        txtData = readfile.readlines()

        laban = OrderedDict()
        frame = OrderedDict()
        previous_frame = OrderedDict()
        self.resetFrameValues(previous_frame, False)

        tags = ['start time',
                'duration',
                'head', 
                'right elbow', 
                'right wrist',
                'left elbow',
                'left wrist',
                'rotation']

        position = "undefined"
        count = 0
        emptyFrame = True
        while len(txtData) > 0: 
            rawtext = txtData.pop(0)

            #
            # comments and empty lines will be ignored and not added to json
            if (rawtext == "\r\n" or rawtext == "\n\r" or rawtext == "\n" or rawtext == "\r" or rawtext.startswith("#")):
                continue

            temp = rawtext.split(':')

            #
            # look for 'start time' to reset frame
            tag = temp[0]
            if (tag.lower() == tags[0].lower()):
                if (count > 0):
                    if (not emptyFrame):
                        self.appendFrame(laban, frame, previous_frame, position)
                    else:
                        print("Empty frame!")

                for key in frame:
                    previous_frame[key] = frame[key]

                #
                # create a new frame
                frame = OrderedDict()
                self.resetFrameValues(frame)
                emptyFrame = True
                position = "Position" + str(count)
                count += 1

                # extract and write 'Start Time' value
                frame[tags[0]] = [i.strip() for i in temp[1:]]
            else:
                fFoundKey = False
                for key in tags:
                    if (key.lower() == tag.lower()):
                        frame[key] = [i.strip() for i in temp[1:]]
                        emptyFrame = False
                        fFoundKey = True
                        break

                if (not fFoundKey):
                    print("    Unrecognized tag '" + tag + "'")

        #
        # write the last frame not yet appended...
        if (not emptyFrame):
            self.appendFrame(laban, frame, previous_frame, position)
        else:
            print("Empty frame!")

        readfile.close()

        #
        # prepare data
        data = {}
        # data['version'] = "0.1"
        data[filename] = laban

        #
        # write data as json
        writeFile = open(output, "w")
        json.dump(data, writeFile, indent=4)
        writeFile.close()
