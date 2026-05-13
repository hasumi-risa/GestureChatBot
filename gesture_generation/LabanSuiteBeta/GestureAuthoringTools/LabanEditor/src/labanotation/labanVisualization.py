# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import cv2
import numpy as np

class convertLabanScriptToView:
    cnt = 0
    width = 480
    height = 960
    bottom = 160
    scale = 100 # pixels/second
    img = np.array((height, width))
    name = ''
    timeOffset = 0.0
    timeScale = 1.0
    duration = 0

    ccolumn = 13 # Number of columns

    #------------------------------------------------------------------------------
    # class initialization
    # w: width; h: height; text: labanotation script
    #
    def __init__(self, w, h, text):
        if (text == ""):
            self.timeOffset = 0.0
            self.timeScale = 1.0
            self.width = w
            self.height = h
            self.img = np.ones((self.height,self.width), np.uint8)
            self.img = self.img*255
            return

        head=[]
        l_elbow=[];l_wrist=[];r_elbow=[];r_wrist=[];
        l_knee=[];l_ankle=[];r_knee=[];r_ankle=[];
        l_support=[];r_support=[]
        i = 0
        self.duration = 0
        self.cnt = 0
        self.name = 'labanotation view'
        laban_line = text.split("\n")

        while i < len(laban_line): # and laban_line[i]!="":
            laban_line[i] = laban_line[i].lower()
            if laban_line[i]=="\n" or laban_line[i]=="":
                # empty line skip
                i+=1
                continue
            elif laban_line[i][0]=="#":
                # comment, skip
                i+=1
                continue
            tmp_str = laban_line[i].split(":")
            # [time, direction, level]
            if tmp_str[0]=="start time":
                if self.cnt==0:
                    start = int(tmp_str[1])
                self.cnt+=1
                self.duration = int(tmp_str[1])-start
                time_stamp = int(tmp_str[1])/1000.0
            elif tmp_str[0]=="left support":
                l_support.append([time_stamp,tmp_str[1],tmp_str[2]])
            elif tmp_str[0]=="left knee":
                l_knee.append([time_stamp,tmp_str[1],tmp_str[2]])
            elif tmp_str[0]=="left ankle":
                l_ankle.append([time_stamp,tmp_str[1],tmp_str[2]])
            elif tmp_str[0]=="left wrist":
                l_wrist.append([time_stamp,tmp_str[1],tmp_str[2]])
            elif tmp_str[0]=="left elbow":
                l_elbow.append([time_stamp,tmp_str[1],tmp_str[2]])
            elif tmp_str[0]=="right support":
                r_support.append([time_stamp,tmp_str[1],tmp_str[2]])
            elif tmp_str[0]=="right knee":
                r_knee.append([time_stamp,tmp_str[1],tmp_str[2]])
            elif tmp_str[0]=="right ankle":
                r_ankle.append([time_stamp,tmp_str[1],tmp_str[2]])
            elif tmp_str[0]=="right wrist":
                r_wrist.append([time_stamp,tmp_str[1],tmp_str[2]])
            elif tmp_str[0]=="right elbow":
                r_elbow.append([time_stamp,tmp_str[1],tmp_str[2]])
            elif tmp_str[0]=="head":
                head.append([time_stamp,tmp_str[1],tmp_str[2]])
            i+=1

        self.scale = h//(2+self.duration//1000)
        self.width = w
        self.height = h+self.bottom
        self.timeOffset = h
        self.timeScale = (self.scale * (self.duration / 1000.0)) # (h * 1000.0 / float(self.duration))

        self.img = np.ones((self.height,self.width), np.uint8)
        self.img = self.img*255
        self.init_canvas()
        
        self.draw_limb(0, "left", l_wrist)
        self.draw_limb(1, "left", l_elbow)
        self.draw_limb(3, "left", l_ankle)
        self.draw_limb(4, "left", l_knee)
        self.draw_support(5, l_support, r_support)
        self.draw_limb(7, "right", r_knee)
        self.draw_limb(8, "right", r_ankle)
        self.draw_limb(10, "right", r_elbow)
        self.draw_limb(11, "right", r_wrist)
        self.draw_limb(12, "right", head)

    #------------------------------------------------------------------------------
    # draw a vertical dashed line.
    def dashed(self, x1, y1, y2):
        dash = 40
        if y1 > y2:
            a = y1; y1 = y2; y2 = a
        for i in range(0,(np.abs(y2-y1))//dash):
            cv2.line(self.img,(x1,y2-i*dash),(x1,y2-i*dash-dash//2),0,2)
        if y2-(i+1)*dash > y1:
            cv2.line(self.img,(x1,y2-(i+1)*dash),(x1,y1),0,2)
    

    #------------------------------------------------------------------------------
    # canvas initialization
    def init_canvas(self):
        unit = self.width//self.ccolumn # Width of a column in pixels
        floor = self.height-self.bottom
        cv2.line(self.img,(unit*3,0),(unit*3,floor),0,2)
        cv2.line(self.img,(unit*6,0),(unit*6,floor),0,2)
        cv2.line(self.img,(unit*9,0),(unit*9,floor),0,2)
        cv2.line(self.img,(unit*3,floor),   (unit*9,floor),0,2)
        cv2.line(self.img,(unit*3,floor+4), (unit*9,floor+4),0,2)
        for i in range(1,self.ccolumn):
            self.dashed(unit*i, 0,floor)
        i = 0
        while True:
            x1 = unit*6-3
            x2 = unit*6+3
            y = floor - i*self.scale
            if y < 0:
                break
            cv2.line(self.img,(x1,y),(x2,y),0,2)
            i += 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        subtitle = self.height-50
        title = self.height-20
        scaleMajor = 0.7
        scaleMinor = 0.4
        cv2.putText(self.img,'arm(L)',  (0*unit+10,title),    font, scaleMajor, 1, 2)
        cv2.putText(self.img,'lower',   (0*unit+5,subtitle),  font, scaleMinor, 1, 2)
        cv2.putText(self.img,'upper',   (1*unit+5,subtitle),  font, scaleMinor, 1, 2)
        cv2.putText(self.img,'leg(L)',  (3*unit+10,title),    font, scaleMajor, 1, 2)
        cv2.putText(self.img,'upper',   (3*unit+5,subtitle),  font, scaleMinor, 1, 2)
        cv2.putText(self.img,'lower',   (4*unit+5,subtitle),  font, scaleMinor, 1, 2)
        cv2.putText(self.img,'support', (5*unit+10,title),    font, scaleMajor, 1, 2)
        cv2.putText(self.img,'leg(R)',  (7*unit+10,title),    font, scaleMajor, 1, 2)
        cv2.putText(self.img,'upper',   (7*unit+5,subtitle),  font, scaleMinor, 1, 2)
        cv2.putText(self.img,'lower',   (8*unit+5,subtitle),  font, scaleMinor, 1, 2)
        cv2.putText(self.img,'arm(R)',  (10*unit+10,title),   font, scaleMajor, 1, 2)
        cv2.putText(self.img,'upper',   (10*unit+5,subtitle), font, scaleMinor, 1, 2)
        cv2.putText(self.img,'lower',   (11*unit+5,subtitle), font, scaleMinor, 1, 2)
        cv2.putText(self.img,'head',    (12*unit-5,title),    font, scaleMajor, 1, 2)
#        cv2.putText(self.img, self.name,(3*unit+10,self.height-35), font, scaleMajor, 1,2)
    
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
    def sign(self, cell, time1, time2, side="right", dire = "place", lv = "low"):
        unit = self.width//self.ccolumn
        x1 = int(cell*unit+7)#left top corner
        x2 = int((cell+1)*unit-5)
        y1 = self.height-self.bottom-int(time2*self.scale)+3#right bottom corner
        y2 = self.height-self.bottom-int(time1*self.scale)-3
        #shading: pattern/black/dot
        if lv=="normal":
            cv2.circle(self.img,((x1+x2)//2,(y1+y2)//2), 4, 0,-1)
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
        elif (lv=="hold") or (lv=='none'):
            pass #No shading needed
        # else:
            # print("Unknown Level: " + lv)
        #shape: trapezoid, polygon, triangle, rectangle
        if lv == 'none':
            pass #No symbol needed
        elif lv=="hold":
            cv2.circle(self.img,((x1+x2)//2,y2 - 15), 10, 0, 2)
        elif dire=="right":
            pts = np.array([[x1,y1-1],[x2+1,y1-1],[x2+1,(y1+y2)//2]],np.int32)
            cv2.fillPoly(self.img,[pts],255)
            pts = np.array([[x1,y2+1],[x2+1,y2+1],[x2+1,(y1+y2)//2]],np.int32)
            cv2.fillPoly(self.img,[pts],255)
            pts = np.array([[x1,y1],[x1,y2],[x2,(y1+y2)//2]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="left":
            pts = np.array([[x1-1,y1-1],[x2,y1-1],[x1-1,(y1+y2)//2]],np.int32)
            cv2.fillPoly(self.img,[pts],255)
            pts = np.array([[x1-1,y2+1],[x2,y2+1],[x1-1,(y1+y2)//2]],np.int32)
            cv2.fillPoly(self.img,[pts],255)
            pts = np.array([[x1,(y1+y2)//2],[x2,y1],[x2,y2]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="left forward":
            pts = np.array([[x1,y1-1],[x2+1,y1-1],[x2+1,y1+(y2-y1)//3]],np.int32)
            cv2.fillPoly(self.img,[pts],255)
            pts = np.array([[x1,y1],[x2,y1+(y2-y1)//3],[x2,y2],[x1,y2]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="right forward":
            pts = np.array([[x1-1,y1-1],[x2+1,y1-1],[x1-1,y1+(y2-y1)//3]],np.int32)
            cv2.fillPoly(self.img,[pts],255)
            pts = np.array([[x1,y1+(y2-y1)//3],[x2,y1],[x2,y2],[x1,y2]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="left backward":
            pts = np.array([[x1,y2+1],[x2+1,y2+1],[x2+1,y2-(y2-y1)//3]],np.int32)
            cv2.fillPoly(self.img,[pts],255)
            pts = np.array([[x1,y1],[x2,y1],[x2,y2-(y2-y1)//3],[x1,y2]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="right backward":
            pts = np.array([[x1-1,y2+1],[x2+1,y2+1],[x1-1,y2-(y2-y1)//3]],np.int32)
            cv2.fillPoly(self.img,[pts],255)
            pts = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2-(y2-y1)//3]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="forward" and side=="right":
            cv2.rectangle(self.img,(x1+(x2-x1)//2,y1-1),(x2+1,y1+(y2-y1)//3),255,-1)
            pts = np.array([[x1,y1],[x1+(x2-x1)//2,y1],[x1+(x2-x1)//2,y1+(y2-y1)//3],
                            [x2,y1+(y2-y1)//3],[x2,y2],[x1,y2]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="forward" and side=="left":
            cv2.rectangle(self.img,(x1-1,y1-1),(x1+(x2-x1)//2,y1+(y2-y1)//3),255,-1)
            pts = np.array([[x1,y1+(y2-y1)//3],[x1+(x2-x1)//2,y1+(y2-y1)//3],[x1+(x2-x1)//2,y1],
                            [x2,y1],[x2,y2],[x1,y2]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="backward" and side=="right":
            cv2.rectangle(self.img,(x1+(x2-x1)//2,y2-(y2-y1)//3),(x2+1,y2+1),255,-1)
            pts = np.array([[x1,y1],[x2,y1],[x2,y2-(y2-y1)//3],
                            [x1+(x2-x1)//2,y2-(y2-y1)//3],[x1+(x2-x1)//2,y2],[x1,y2]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="backward" and side=="left":
            cv2.rectangle(self.img,(x1-1,y2-(y2-y1)//3),(x1+(x2-x1)//2,y2+1),255,-1)
            pts = np.array([[x1,y1],[x2,y1],[x2,y2],
                            [x1+(x2-x1)//2,y2],[x1+(x2-x1)//2,y2-(y2-y1)//3],
                            [x1,y2-(y2-y1)//3]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="place":#"Place"
            cv2.rectangle(self.img,(x1,y1),(x2,y2),0,2)
        # else:
            # print("Unknown Direction: " + side + ": " + dire)
    
    #------------------------------------------------------------------------------
    # draw one column of labanotation for one limb
    # 
    def draw_limb(self,cell,side,laban):
        if (len(laban) > 0):
            self.sign(cell,-90.0/self.scale,-5.0/self.scale,side,laban[0][1],laban[0][2])
            i=1
            while i <= self.cnt-1:
                if laban[i-1][1]==laban[i][1] and laban[i-1][2]==laban[i][2]:
                    pass
                else:
                    #sign(cell,time1,time2, side="Right", dire = "Place",lv = "Low"):
                    self.sign(cell,laban[i-1][0],laban[i][0],side,laban[i][1],laban[i][2])
                i+=1

    #------------------------------------------------------------------------------
    # draw one column of labanotation for one limb
    # 
    def draw_limb_support(self,cell,side,laban):
        if (len(laban) > 0):
            # Draw starting position
            self.sign(cell,-90.0/self.scale,-5.0/self.scale,side,laban[0][1],laban[0][2])

            labanFrameStart = [laban[0][0], laban[0][1], 'hold']  # Skip the first entry which is the starting position assume we start in a hold
            for i in range(1, self.cnt):
                labanFrame = laban[i]
                if labanFrame[1] == labanFrameStart[1] and labanFrame[2] == labanFrameStart[2]:
                    pass
                else:
                    if (labanFrameStart[2] != 'hold'):
                        #sign(cell,time1,time2, side="Right", dire = "Place",lv = "Low"):
                        self.sign(cell,labanFrameStart[0],labanFrame[0],side,labanFrameStart[1],labanFrameStart[2])
                    labanFrameStart = labanFrame


    def draw_support(self, cell, laban_left, laban_right):
        if not laban_left or not laban_right:
            return

        # Column positions
        cellLeft = cell
        cellRight = cell + 1
        cellBoth = cell + 0.5

        # Draw direction and level signs
        self.draw_limb_support(cellLeft, "left", laban_left)
        self.draw_limb_support(cellRight, "right", laban_right)

        # Draw hold signs
        frameLeft = laban_left[0]
        frameRight = laban_right[0]
        dirLeftPrev = frameLeft[1]
        dirRightPrev = frameRight[1]
        levelLeftPrev = frameLeft[2]
        levelRightPrev = frameRight[2]

        for i in range(1, self.cnt):
            frameLeft = laban_left[i]
            frameRight = laban_right[i]
            dirLeft = frameLeft[1]
            dirRight = frameRight[1]
            levelLeft = frameLeft[2]
            levelRight = frameRight[2]

            # Skip frames that are the same as the last
            if (dirLeft == dirLeftPrev) and (levelLeft == levelLeftPrev) and (dirRight == dirRightPrev) and (levelRight == levelRightPrev):
                continue

            # Add hold signs as needed
            fHoldLeft = (levelLeft == 'hold')
            fHoldRight = (levelRight == 'hold')

            if fHoldLeft and fHoldRight:
                self.sign(cell=cellBoth, time1=frameLeft[0], time2=frameLeft[0], dire="place", lv='hold')
            elif fHoldLeft:
                self.sign(cell=cellLeft, time1=frameLeft[0], time2=frameLeft[0], dire="place", lv='hold')
            elif fHoldRight:
                self.sign(cell=cellRight, time1=frameRight[0], time2=frameRight[0], dire="place", lv='hold')

            # Advance to next frame
            dirLeftPrev = dirLeft
            dirRightPrev = dirRight
            levelLeftPrev = levelLeft
            levelRightPrev = levelRight

