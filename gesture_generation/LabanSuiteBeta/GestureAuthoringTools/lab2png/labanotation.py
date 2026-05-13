# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import sys
import os
import json, operator
import threading
import time
from time import sleep
import datetime
import cv2
import numpy as np

# -----------------------------------------------------------------------------
#
class Labanotation:
    # -----------------------------------------------------------------------------
    #
    def __init__(self, application):
        if (application is None):
            print("ERROR in Labanotation.init(): application object expected.")

        self.application = application

        self.duration = 0.0
        self.labanotation = []
        self.poses = []
        self.frameTimes = []

    # -----------------------------------------------------------------------------
    #
    def scanFolderForGestures(self):
        try:
            folders = self.labanotationFiles['folders']
            for i in range(len(folders)):
                folder = folders[i]

                if (folder[-1] is not '\\') or (folder[-1] is not '/'):
                    folder = folder + '/'
                    self.labanotationFiles['folders'][i] = folder

                set = os.listdir(folder)
                set.sort()

                self.labanotationFiles['sets'].append(set)
        except Exception as e:
            if (e.message):
                print("Error: '" + e.message)
            elif ((e.errno) and (e.errno == 2)):
                print("Error: '" + e.filename + "' - " + e.strerror)
            else:
                print("Error: " + e.strerror)

    # -----------------------------------------------------------------------------
    #
    def close(self):
        self.labanotation = []
        self.poses = []
        self.frameTimes = []

    # -----------------------------------------------------------------------------
    #
    def getDuration(self):
        return self.duration

    # -----------------------------------------------------------------------------
    #
    def loadGestureFile(self, fileName):
        data = None
        try:
            with open(fileName) as f:
                data = json.load(f)

        except Exception as e:
            if (e.message):
                print("Exception: '" + e.message)
            elif (hasattr(e, 'errno') and (e.errno == 2)):
                print("Exception: '" + e.filename + "' - " + e.strerror)
            elif (hasattr(e, 'strerror')):
                print("Exception: " + e.strerror)
            else:
                print("Exception: ", e)

            return False

        self.parseGestureData(data)

        print("    " + str(len(self.poses)) + " frames, duration: " + str(self.duration) + "ms")

        return True

    # -----------------------------------------------------------------------------
    #
    def parseGestureData(self, data):
        for key in data:
            value = data[key]
            return self.parseGesturePositions(key, value)

    # -----------------------------------------------------------------------------
    #
    def parseGesturePositions(self, key, data):
        #print("The key and value are ({}) = ({})".format(key, data))
        #print("The key ({}) ".format(key))

        self.currentTime = 0.0
        self.currentPosePositionIndex = -1
        self.duration = 0.0
        self.labanotation = []
        self.poses = []

        for subKey in data:
            #print("position: ({}) ".format(subKey))
            self.labanotation.append(data[subKey])

        #
        # convert time and durations lists to numbers. Parse angles
        for i in range(len(self.labanotation)):
            pose = self.labanotation[i]
            # print(pose)
            if ('start time' in pose):
                value = pose['start time'][0]
                t = type(value)
                pose['start time'] = float(value)
            else:
                print("missing 'start time' information...") # for key frame '" + str(pose['name']) + "'...")
                continue

            if ('duration' in pose):
                duration = pose['duration'][0]
                t = type(duration)
                pose['duration'] = float(duration)

            #
            # check for complete set
            self.checkFrameValidity(pose)

        #
        # sort by 'start time'
        sorted_items = sorted(self.labanotation, key=operator.itemgetter('start time'))

        #
        # calculate and overwrite durations. create frameTimes array
        self.frameTimes = []
        for i in range(len(sorted_items)):
            startTime1 = float(sorted_items[i]['start time'])
            if ((i + 1) < len(sorted_items)):
                startTime2 = float(sorted_items[i+1]['start time'])
            else:
                startTime2 = startTime1 + 1000.0

            sorted_items[i]['duration'] = float(startTime2 - startTime1)

            self.frameTimes.append(startTime1)

        if (len(sorted_items) > 0):
            last_item = sorted_items[len(sorted_items) - 1]
            self.duration = last_item['start time'] + last_item['duration']

        for i in range(len(sorted_items)):
            startTime = float(sorted_items[i]['start time'])
            duration = sorted_items[i]['duration']
            pose = {}
            pose["StartTime"] = startTime
            pose["Duration"] = duration
            pose["RightElbow"] = self.convertLabanotation(sorted_items[i]["right elbow"], duration)
            pose["RightWrist"] = self.convertLabanotation(sorted_items[i]["right wrist"], duration)
            pose["LeftElbow"] = self.convertLabanotation(sorted_items[i]["left elbow"], duration)
            pose["LeftWrist"] = self.convertLabanotation(sorted_items[i]["left wrist"], duration)

            self.poses.append(pose)

    # -----------------------------------------------------------------------------
    #
    def checkFrameValidity(self, pose):
        tags = ['start time',
                'duration',
                'head', 
                'right elbow', 
                'right wrist',
                'left elbow',
                'left wrist',
                'rotation']

        for tag in tags:
            fFoundTag = False
            for key in pose:
                if (tag.lower() == key.lower()):
                    fFoundTag = True
                    break

            if (not fFoundTag):
                print("Tag '" + tag + "' is missing") # in frame with 'start time'=" + str(pose['start time']) + "")

    # -----------------------------------------------------------------------------
    #
    def convertLabanotation(self, limb, duration):
        phi = 0.0
        theta = 180.0

        dir = limb[0].lower()
        lv = limb[1].lower()

        if (lv == "high"):
            theta = 45.0
        elif (lv == "normal"):
            theta = 90.0
        elif (lv == "low"):
            theta = 135.0
        else:
            theta = 180.0
            print("Unknown level '" + lv + "'")

        if (dir == "forward"):
            phi = 0.0
        elif (dir == "right forward"):
            phi = -45.0
        elif (dir == "right"):
            phi = -90.0
        elif (dir == "right backward"):
            phi = -135.0
        elif (dir == "backward"):
            phi = 180.0
        elif (dir == "left backward"):
            phi = 135.0
        elif (dir == "left"):
            phi = 90.0
        elif (dir == "left forward"):
            phi = 45.0
        elif (dir == "place"):
            if (lv == "high"):
                theta = 5.0
                phi = 0.0
            elif (lv == "low"):
                theta = 175.0
                phi = 0.0
            else:
                theta = 180.0
                phi = 0.0
                print ("Unknown place for '" + lv + "'");
        else:
            phi = 0;
            print ("Unknown direction for '" + dir + "'");

        theta = self.convertToRadians(theta);
        phi = self.convertToRadians(phi);

        return { 'theta': theta, 'phi': phi, 'duration': duration, 'dir': dir, 'lv': lv }

    # -----------------------------------------------------------------------------
    #
    def convertToRadians(self, degrees):
       return (degrees * 3.141592654 / 180.0)

    # -----------------------------------------------------------------------------
    #
    def convertToDegrees(self, radians):
       return (radians * 180.0 / 3.141592654)

    #------------------------------------------------------------------------------
    #
    def LabanKeyframeToScript(self, idx, time, dur, laban_score):
        strScript = ""

        strScript += '#' + str(idx) + '\n'
        strScript += 'Start Time:'+ str(time) +'\nDuration:' + str(dur) + '\nHead:Forward:Normal\n'
        strScript += 'Right Elbow:' + laban_score[0][0] + ':' + laban_score[0][1] + '\n'
        strScript += 'Right Wrist:' + laban_score[1][0] + ':' + laban_score[1][1] + '\n'
        strScript += 'Left Elbow:'  + laban_score[2][0] + ':' + laban_score[2][1] + '\n'
        strScript += 'Left Wrist:'  + laban_score[3][0] + ':' + laban_score[3][1] + '\n'
        strScript += 'Rotation:ToLeft:0.0\n'

        return strScript

    #------------------------------------------------------------------------------
    #
    def getThetaPhi(self, laban):
        l = []
        l.append(laban['theta'])
        l.append(laban['phi'])
        return l;

    #------------------------------------------------------------------------------
    #
    def labanToScript(self):
        strScript = ""
        cnt = len(self.poses)
        for idx in range(cnt):
            startTime = self.poses[idx]['StartTime']
            duration = self.poses[idx]['Duration']

            re = self.poses[idx]["RightElbow"]
            rw = self.poses[idx]["RightWrist"]
            le = self.poses[idx]["LeftElbow"]
            lw = self.poses[idx]["LeftWrist"]

            if idx == 0:
                time = 1
            else:
                time = int(startTime)

            if idx == (cnt - 1):
                dur = '-1'
            else:
                dur = '1'

            strScript += '#' + str(idx) + '\n'
            strScript += 'Start Time:'+ str(time) +'\nDuration:' + str(dur) + '\nHead:Forward:Normal\n'
            strScript += 'Right Elbow:' + re['dir'] + ':' + re['lv'] + '\n'
            strScript += 'Right Wrist:' + rw['dir'] + ':' + rw['lv'] + '\n'
            strScript += 'Left Elbow:'  + le['dir'] + ':' + le['lv'] + '\n'
            strScript += 'Left Wrist:'  + lw['dir'] + ':' + lw['lv'] + '\n'
            strScript += 'Rotation:ToLeft:0.0\n'

        return strScript

    #------------------------------------------------------------------------------
    #
    def convertToImage(self, width):
        height = 4 * width

        l_elbow = []
        l_wrist = []
        r_wrist = []
        r_elbow = []
        head = []

        cnt = len(self.poses)
        for idx in range(cnt):
            startTime = self.poses[idx]['StartTime']
            duration = self.poses[idx]['Duration']

            re = self.poses[idx]["RightElbow"]
            rw = self.poses[idx]["RightWrist"]
            le = self.poses[idx]["LeftElbow"]
            lw = self.poses[idx]["LeftWrist"]

            if idx == 0:
                time = 1
            else:
                time = int(startTime)

            if idx == (cnt - 1):
                dur = '-1'
            else:
                dur = '1'

            #strScript += '#' + str(idx) + '\n'
            #strScript += 'Start Time:'+ str(time) +'\nDuration:' + str(dur) + '\nHead:Forward:Normal\n'
            #strScript += 'Right Elbow:' + re['dir'] + ':' + re['lv'] + '\n'
            #strScript += 'Right Wrist:' + rw['dir'] + ':' + rw['lv'] + '\n'
            #strScript += 'Left Elbow:'  + le['dir'] + ':' + le['lv'] + '\n'
            #strScript += 'Left Wrist:'  + lw['dir'] + ':' + lw['lv'] + '\n'
            #strScript += 'Rotation:ToLeft:0.0\n'
            time_stamp = int(startTime) / 1000.0

            l_elbow.append([time_stamp, re['dir'], re['lv']])
            l_wrist.append([time_stamp, rw['dir'], rw['lv']])
            r_wrist.append([time_stamp, le['dir'], le['lv']])
            r_elbow.append([time_stamp, lw['dir'], lw['lv']])
            # head.append([time_stamp,tmp_str[1], tmp_str[2]])

        self.bottom = 160
        self.scale = height / (2+self.duration/1000)
        self.width = width
        self.height = height + self.bottom

        self.img = np.ones((self.height,self.width), np.uint8)
        self.img = self.img*255
        self.init_canvas()

        self.draw_limb(1, "left", l_wrist)
        self.draw_limb(2, "left", l_elbow)
        self.draw_limb(9, "right", r_elbow)
        self.draw_limb(10, "right", r_wrist)
        # self.draw_limb(11, "right", head)

        return self.img

    #------------------------------------------------------------------------------
    # draw a vertical dashed line.
    def dashed(self, x1, y1, y2):
        dash = 40
        if y1 > y2:
            a = y1; y1 = y2; y2 = a
        for i in range(0,(np.abs(y2-y1))/dash):
            cv2.line(self.img,(x1,y2-i*dash),(x1,y2-i*dash-dash/2),0,2)
        if y2-(i+1)*dash > y1:
            cv2.line(self.img,(x1,y2-(i+1)*dash),(x1,y1),0,2)
    

    #------------------------------------------------------------------------------
    # canvas initialization
    def init_canvas(self):
        unit = self.width/11
        floor = self.height-self.bottom
        cv2.line(self.img,(unit*3,0),(unit*3,floor),0,2)
        cv2.line(self.img,(unit*5,0),(unit*5,floor),0,2)
        cv2.line(self.img,(unit*7,0),(unit*7,floor),0,2)
        cv2.line(self.img,(unit*3,floor),
                 (unit*7,floor),0,2)
        cv2.line(self.img,(unit*3,floor+4),
                 (unit*7,floor+4),0,2)
        for i in range(1,11):
            self.dashed(unit*i, 0,floor)
        i = 0
        while True:
            x1 = unit*5-3
            x2 = unit*5+3
            y = floor - i*self.scale
            if y < 0:
                break
            cv2.line(self.img, (int(x1), int(y)),(int(x2), int(y)), 0, 2)
            i += 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        subtitle = self.height-50
        title = self.height-20
        cv2.putText(self.img,'lower',(0*unit+5,subtitle), font, 0.5, 1,2)
        cv2.putText(self.img,'upper',(1*unit+5,subtitle), font, 0.5, 1,2)
        cv2.putText(self.img,'upper',(8*unit+5,subtitle), font, 0.5, 1,2)
        cv2.putText(self.img,'lower',(9*unit+5,subtitle), font, 0.5, 1,2)
        cv2.putText(self.img,'head',(10*unit-5,title), font, 0.8   , 1,2)
        cv2.putText(self.img,'arm(L)',(0*unit+10,title), font, 0.8, 1,2)
        cv2.putText(self.img,'arm(R)',(8*unit+10,title), font, 0.8, 1,2)
#        cv2.putText(self.img, self.name,(3*unit+10,self.height-35), font, 0.8, 1,2)
    
    #------------------------------------------------------------------------------
    # draw sign of Labanotation.
    # side: right hand side, left hand side
    #     for determin which forward/backward sign should be used.
    # direction: place, 
    #     forward, backward
    #     right, left
    #     right forward (diagonal), right backward (diagonal)
    #     left forward (diagonal), left backward (diagonal)
    # level: low, normal, high
    # (x1,y1) is the left top corner, (x2,y2) is the right bottom corner.
    # 
    def sign(self, cell, (time1,time2), side="right", dire = "place", lv = "low"):
        unit = self.width/11
        x1 = (cell-1)*unit+7#left top corner
        x2 = cell*unit-5
        y1 = self.height-self.bottom-int(time2*self.scale)+3#right bottom corner
        y2 = self.height-self.bottom-int(time1*self.scale)-3
        #shading: pattern/black/dot
        if lv=="normal":
            cv2.circle(self.img,((x1+x2)/2,(y1+y2)/2), 4, 0,-1)
        elif lv=="high":
            step = 20
            i=0
            while True:
                xl = x1#start point at the left
                yl = y1+i*step           
                xr = x1+i*step#end point at th right
                yr = y1
                if yl > y2:
                    xl = yl-y2+xl
                    yl = y2
                if xr > x2:
                    yr = y1+xr-x2
                    xr = x2
                if (xl>xr)or(yr>yl):
                    break
                cv2.line(self.img, (xl,yl),(xr, yr),0,2)
                i+=1
        elif lv=="low":
            cv2.rectangle(self.img,(x1,y1),(x2,y2),0,-1)
        else:
            print "Unknow Level: " + lv
        #shape: trapezoid, polygon, triangle, rectangle
        if dire=="right":
            pts = np.array([[x1,y1-1],[x2+1,y1-1],[x2+1,(y1+y2)/2]],np.int32)
            cv2.fillPoly(self.img,[pts],255)
            pts = np.array([[x1,y2+1],[x2+1,y2+1],[x2+1,(y1+y2)/2]],np.int32)
            cv2.fillPoly(self.img,[pts],255)
            pts = np.array([[x1,y1],[x1,y2],[x2,(y1+y2)/2]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="left":
            pts = np.array([[x1-1,y1-1],[x2,y1-1],[x1-1,(y1+y2)/2]],np.int32)
            cv2.fillPoly(self.img,[pts],255)
            pts = np.array([[x1-1,y2+1],[x2,y2+1],[x1-1,(y1+y2)/2]],np.int32)
            cv2.fillPoly(self.img,[pts],255)
            pts = np.array([[x1,(y1+y2)/2],[x2,y1],[x2,y2]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="left forward":
            pts = np.array([[x1,y1-1],[x2+1,y1-1],[x2+1,y1+(y2-y1)/3]],np.int32)
            cv2.fillPoly(self.img,[pts],255)
            pts = np.array([[x1,y1],[x2,y1+(y2-y1)/3],[x2,y2],[x1,y2]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="right forward":
            pts = np.array([[x1-1,y1-1],[x2+1,y1-1],[x1-1,y1+(y2-y1)/3]],np.int32)
            cv2.fillPoly(self.img,[pts],255)
            pts = np.array([[x1,y1+(y2-y1)/3],[x2,y1],[x2,y2],[x1,y2]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="left backward":
            pts = np.array([[x1,y2+1],[x2+1,y2+1],[x2+1,y2-(y2-y1)/3]],np.int32)
            cv2.fillPoly(self.img,[pts],255)
            pts = np.array([[x1,y1],[x2,y1],[x2,y2-(y2-y1)/3],[x1,y2]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="right backward":
            pts = np.array([[x1-1,y2+1],[x2+1,y2+1],[x1-1,y2-(y2-y1)/3]],np.int32)
            cv2.fillPoly(self.img,[pts],255)
            pts = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2-(y2-y1)/3]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="forward" and side=="right":
            cv2.rectangle(self.img,(x1+(x2-x1)/2,y1-1),(x2+1,y1+(y2-y1)/3),255,-1)
            pts = np.array([[x1,y1],[x1+(x2-x1)/2,y1],[x1+(x2-x1)/2,y1+(y2-y1)/3],
                            [x2,y1+(y2-y1)/3],[x2,y2],[x1,y2]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="forward" and side=="left":
            cv2.rectangle(self.img,(x1-1,y1-1),(x1+(x2-x1)/2,y1+(y2-y1)/3),255,-1)
            pts = np.array([[x1,y1+(y2-y1)/3],[x1+(x2-x1)/2,y1+(y2-y1)/3],[x1+(x2-x1)/2,y1],
                            [x2,y1],[x2,y2],[x1,y2]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="backward" and side=="right":
            cv2.rectangle(self.img,(x1+(x2-x1)/2,y2-(y2-y1)/3),(x2+1,y2+1),255,-1)
            pts = np.array([[x1,y1],[x2,y1],[x2,y2-(y2-y1)/3],
                            [x1+(x2-x1)/2,y2-(y2-y1)/3],[x1+(x2-x1)/2,y2],[x1,y2]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="backward" and side=="left":
            cv2.rectangle(self.img,(x1-1,y2-(y2-y1)/3),(x1+(x2-x1)/2,y2+1),255,-1)
            pts = np.array([[x1,y1],[x2,y1],[x2,y2],
                            [x1+(x2-x1)/2,y2],[x1+(x2-x1)/2,y2-(y2-y1)/3],
                            [x1,y2-(y2-y1)/3]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="place":#"Place"
            cv2.rectangle(self.img,(x1,y1),(x2,y2),0,2)
        else:
            print "Unknow Direction: " + side + ": " + dire
    
    #------------------------------------------------------------------------------
    # draw one column of labanotation for one limb
    # 
    def draw_limb(self,cell,side,laban):
        cnt = len(self.poses)

        self.sign(cell,(-90.0/self.scale,-5.0/self.scale),side,laban[0][1],laban[0][2])
        i=1
        while i <= (cnt - 1):
            if laban[i-1][1]==laban[i][1] and laban[i-1][2]==laban[i][2]:
                pass
            else:
                #sign(cell,(time1,time2), side="Right", dire = "Place",lv = "Low"):
                self.sign(cell,(laban[i-1][0],laban[i][0]),side,laban[i][1],laban[i][2])
            i+=1


    #------------------------------------------------------------------------------
    #
    def convertToImage_OLD(self, w):
        # print("converting to script...")
        script = self.labanToScript()
        # print("converting to image...")
        return labanVisualization.convertLabanScriptToView(w, (4*w), script);

