# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

from __future__ import absolute_import

from builtins import range
from collections import namedtuple
from enum import IntEnum

import json
import math
import os

import numpy as np

from . import labanProcessor as lp
from .algorithmbase import AlgorithmBase
from .graphLowerBody import *
from .tool import accessory
from .tool import cluster
from .tool import wavfilter



class FootDir(IntEnum):
    Place = -1 # Foot is under the body
    Forward = 0
    LeftForward = 1
    Left = 2
    LeftBackward = 3
    Backward = 4
    RightBackward = 5
    Right = 6
    RightForward = 7
    MAX = 8


class FootLevel(IntEnum):
    Hold = -2 # Same as previous non-"hold" level
    NoLevel = -1 # No level (for support column, no weight this side)
    Low = 0 # Foot extended (e.g., tip-toes)
    Normal = 1 # Normal (e.g., normal standing)
    High = 2 # Draw in (e.g., crouching)





class Algorithm(AlgorithmBase):
    # Stored data for display or analysis
    class Data:
        class JointAnalysisData:
            def __init__(self):
                self.position = None
                self.velocity = None # Velocity or speed in Y
                self.rgiMovementChanges = None
                self.rgfIsMoving = None
                self.rgvec = None # Velocity vector at each timestamp
                self.rgxyz = None # Raw (x, y, z) of joint

        class MinimaDetection:
            class JointData:
                class ScaledSpace:
                    def __init__(self):
                        self.inflection = None
                        self.filtered = None
                        self.peak = None
                        self.rgvFilteredNarrow = None
                        self.rgvFilteredWide = None


                def __init__(self):
                    self.scaled_space = None

            def __init__(self):
                self.left = self.JointData()
                self.right = self.JointData()

        class KeyFrame:
            def __init__(self):
                self.rgsec = None
                self.rgpairsecFIsMoving = None


        def __init__(self):
            self.keyframe = self.KeyFrame()
            self.rgsecTimestamp = []
            self.left = self.JointAnalysisData()
            self.right = self.JointAnalysisData()
            self.minimadetection = self.MinimaDetection()
            self.rgvecVelocityBody = None # Body velocity vector


    # Mapping from foot direction to Laban keyword string
    mpFootDirLabanString = {
        FootDir.Place:         'Place',
        FootDir.Forward:       'Forward',
        FootDir.LeftForward:   'LeftForward',
        FootDir.Left:          'Left',
        FootDir.LeftBackward:  'LeftBackward',
        FootDir.Backward:      'Backward',
        FootDir.RightBackward: 'RightBackward',
        FootDir.Right:         'Right',
        FootDir.RightForward:  'RightForward'
        }

    # Mapping from foot level to Laban keyword string
    mpFootLevelLabanString = {
        FootLevel.Hold:    'Hold',
        FootLevel.NoLevel: 'None',
        FootLevel.Low:     'Low',
        FootLevel.Normal:  'Normal',
        FootLevel.High:    'High',
        }



    #------------------------------------------------------------------------------
    # Initialize this class instance
    # 
    def __init__(self):
        super().__init__()
        self.graph = None # PyPlot graph to display our data

        # Processing options
        self.dictOptions = {
            'algorithm':            'velocity', # Algorithm used to detect keyframes (velocity, parallel)
            'algMotionChange':      'localminima', # Algorithm used to detect motion changes (threshold, localminima)
            'dataset':              'feet', # Input data set to process (feet, hands)
            'fCalc3DSpeed':         True, # If true, calculate speed using (x, y, z) instead of velocity using only y
            'fFilterPosition':      True, # If true, filter the position using the app-wide Gaussian filter settings
            'fFilterMotionChange':  True, # If true, filter motion change indices
            'fFilterVelocity':      False, # If true, filter the velocity using the app-wide Gaussian filter settings

            'fMakeYRelative' :      True, # If true, y is calculated relative to the spine base joint
            'speedThresholdMovementChange': 0.0001, # Speed threshold for indentifying movement change when algMotionChange == threshold
            'minimadetection':      { # Speed minima detection settings
                'wide': GaussParams(9, 3), # Coarse Gaussian filter
                'narrow': GaussParams(5, 2), # Fine Gaussian filter
                },
            'support':              { # Support column settings
                'algorithm':        'correlateddirection', # Algorithm for support detection (correlateddirection, kneeangle, lowesty, lowestposition)
                'fContinuousWalking': False,
                }
            }

        self.data = self.Data() # Stored data
        self.labandata = [] # Laban script for export


    #------------------------------------------------------------------------------
    # Analyze the Kinect skeleton joint series and return the indicies of potential
    # key frames.
    # 
    def analyzePotentialKeyFrames(self, jointD, rgsec):
        # Create Gaussian filter kernel, if we need it
        if self.dictOptions['fFilterPosition'] or self.dictOptions['fFilterVelocity']:
            gausskernel = wavfilter.gaussFilter(self.gauss_window_size, self.gauss_sigma)

        # Get the foot's vertical position at each timestamp
        cjointframe = len(jointD)
        rgmsec = [sec * 1000.0 for sec in rgsec]
        rgdmLeft = []
        rgdmRight = []
        rgxyzLeft = []
        rgxyzRight = []
        fMakeYRelative = self.dictOptions['fMakeYRelative']

        strDataSet = self.dictOptions['dataset']
        # print("alglowerbody: using dataset \"" + strDataSet + "\"")
        if strDataSet == 'debug':
            secFirst = rgsec[0]
            dsec = rgsec[-1] - secFirst
            for sec in rgsec:
                secT = sec - secFirst
                radians = secT * 2.0 * math.pi
                rgxyzLeft.append((math.sin(radians) * 0.20, math.cos(radians) * 0.05, secT / dsec))
                radians = radians / 1.5
                rgxyzRight.append((math.sin(radians) * 0.20, -math.cos(radians) * 0.05, secT / dsec))
        elif strDataSet == 'debug2':
            secFirst = rgsec[0]
            dsec = rgsec[-1] - secFirst
            for sec in rgsec:
                secT = (sec - secFirst) / 2.0
                radians = secT * 2.0 * math.pi
                radians2 = secT * 7.0 / 5.0 * math.pi
                amplitudeModulation = abs(math.sin(radians2)) * 0.5 + 0.5
                rgxyzLeft.append((math.sin(radians) * 0.20 * amplitudeModulation, math.cos(radians) * 0.05 * amplitudeModulation, secT / dsec))
                radians = radians / 1.5
                rgxyzRight.append((math.sin(radians) * 0.20 * amplitudeModulation, -math.cos(radians) * 0.05 * amplitudeModulation, secT / dsec))
        else:
            if strDataSet == 'feet':
                leftJointID = 'ankleL'
                rightJointID = 'ankleR'
            elif strDataSet == 'hands':
                leftJointID = 'wristL'
                rightJointID = 'wristR'

            # Build list of joint positions at each timestamp
            for i in range(cjointframe):
                jointDFrame = jointD[i]

                # Get the position of the joints
                jointPos = jointDFrame[leftJointID]
                rgxyzLeft.append((jointPos['x'][0], jointPos['y'][0], jointPos['z'][0]))
                jointPos = jointDFrame[rightJointID]
                rgxyzRight.append((jointPos['x'][0], jointPos['y'][0], jointPos['z'][0]))

        # Build list of joint positions at each timestamp
        if self.dictOptions['fCalc3DSpeed']:
            # If requested, make position relative to base of the spine
            if fMakeYRelative:
                for i in range(cjointframe):
                    jointDFrame = jointD[i]
                    (xLeft, yLeft, zLeft) = rgxyzLeft[i]
                    (xRight, yRight, zRight) = rgxyzRight[i]

                    jointPos = jointDFrame['hipL']
                    xSpine, ySpine, zSpine = (jointPos['x'][0], jointPos['y'][0], jointPos['z'][0])

                    xLeft = xLeft - xSpine
                    yLeft = yLeft - ySpine
                    zLeft = zLeft - zSpine

                    jointPos = jointDFrame['hipR']
                    xSpine, ySpine, zSpine = (jointPos['x'][0], jointPos['y'][0], jointPos['z'][0])
                    xRight = xRight - xSpine
                    yRight = xRight - ySpine
                    zRight = zRight - ySpine

                    rgxyzLeft[i] = (xLeft, yLeft, zLeft)
                    rgxyzRight[i] = (xRight, yRight, zRight)

            # Apply filter on positions, if enabled
            if self.dictOptions['fFilterPosition']:
                rgxyzLeft = wavfilter.calcFilter(np.asarray(rgxyzLeft), gausskernel)
                rgxyzRight = wavfilter.calcFilter(np.asarray(rgxyzRight), gausskernel)

            # Calculate the distances between sequential vectors for speed calculation
            rgdmLeft = [0]
            rgdmRight = [0]
            vecLeftLast = rgxyzLeft[0]
            vecRightLast = rgxyzRight[0]
            for i in range(1, cjointframe):
                vec = rgxyzLeft[i]
                rgdmLeft.append(self.lenVec((vec[0] - vecLeftLast[0], vec[1] - vecLeftLast[1], vec[2] - vecLeftLast[2])))
                vecLeftLast = vec

                vec = rgxyzRight[i]
                rgdmRight.append(self.lenVec((vec[0] - vecRightLast[0], vec[1] - vecRightLast[1], vec[2] - vecRightLast[2])))
                vecRightLast = vec

            self.data.left.position = rgdmLeft
            self.data.right.position = rgdmRight

        else:
            # Build list of joint positions at each timestamp
            rgyLeft = []
            rgyRight = []
            if not fMakeYRelative:
                rgyLeft = [vec[1] for vec in rgxyzLeft]
                rgyRight = [vec[1] for vec in rgxyzRight]
            else:
                for i in range(cjointframe):
                    jointDFrame = jointD[i]
                    yLeft = rgxyzLeft[i][1]
                    yRight= rgxyzRight[i][1]

                    # If requested, adjust vectors so that they are relative to the spine base
                    # so that movement is relative to the body to reduce positional noise
                    # caused by orientation.
                    ySpine = jointDFrame['spineB']['y']
                    yLeft = yLeft - ySpine
                    yRight = yRight - ySpine

                    rgyLeft.append(yLeft)
                    rgyRight.append(yRight)

            # Apply filter on positions, if enabled
            if self.dictOptions['fFilterPosition']:
                rgyLeft = wavfilter.calcFilter(np.asarray(rgyLeft), gausskernel)
                rgyRight = wavfilter.calcFilter(np.asarray(rgyRight), gausskernel)

            self.data.left.position = rgyLeft
            self.data.right.position = rgyRight

            # Calculate the distances between sequential positions for velocity calculation
            rgdmLeft = [0]
            rgdmRight = [0]
            yLeftLast = rgyLeft[0]
            yRightLast = rgyRight[0]
            for i in range(1, cjointframe):
                y = rgyLeft[i]
                rgdmLeft.append(y - yLeftLast)
                yLeftLast = y

                y = rgyRight[i]
                rgdmRight.append(y - yRightLast)
                yRightLast = y

        self.data.left.rgxyz = rgxyzLeft
        self.data.right.rgxyz = rgxyzRight

        # Get the velocities (or speed) of each side
        rgvelocityLeft = self.vel1D(rgmsec, rgdmLeft)
        rgvelocityRight = self.vel1D(rgmsec, rgdmRight)

        #print("Raw rgvelocityLeft = ", rgvelocityLeft)
        #print("Raw rgvelocityRight = ", rgvelocityRight)

        # Filter the velocities, if enabled
        if self.dictOptions['fFilterVelocity']:
            rgvelocityLeft = wavfilter.calcFilter(rgvelocityLeft, gausskernel)
            rgvelocityRight = wavfilter.calcFilter(rgvelocityRight, gausskernel)
            #print("Filtered rgvelocityLeft = ", rgvelocityLeft)
            #print("Filtered rgvelocityRight = ", rgvelocityRight)

        self.data.left.velocity = rgvelocityLeft
        self.data.right.velocity = rgvelocityRight

        # Get the indexes where there's a change in movement (moving to stopped, and vice-versa)
        if self.dictOptions['algMotionChange'] == 'localminima':
            if self.dictOptions['fCalc3DSpeed']:
                rgposLeft = rgxyzLeft
                rgposRight = rgxyzRight
            else:
                rgposLeft = rgyLeft
                rgposRight = rgyRight
        else:
            rgposLeft = None
            rgposRight = None

        rgiMovementChangesRight = self.identifyMovementChanges(rgvelocityRight, rgdmRight, self.data.minimadetection.right)
        rgiMovementChangesLeft = self.identifyMovementChanges(rgvelocityLeft, rgdmLeft, self.data.minimadetection.left)

        # print("rgiMovementChangesLeft = ", rgiMovementChangesLeft)
        # print("rgiMovementChangesRight = ", rgiMovementChangesRight)

        # Filter out indices for which there is insufficient movement
        if not self.dictOptions['fFilterMotionChange']:
            rgiMovementChangesLeft = self.filterMovementIndices(rgiMovementChangesLeft,  rgdmLeft,  rgmsec)
            rgiMovementChangesRight = self.filterMovementIndices(rgiMovementChangesRight, rgdmRight, rgmsec)

        self.data.left.rgiMovementChanges = rgiMovementChangesLeft
        self.data.right.rgiMovementChanges = rgiMovementChangesRight

        return self.analyzePotentialKeyFramesSpeedCrossing(rgvelocityLeft, rgvelocityRight)


    def analyzePotentialKeyFramesSpeedCrossing(self, rgvelocityLeft, rgvelocityRight):
        speedThreshold = 0.005 # Threshold speed, meters per second
        speedThreshold2 = 0.001 # Threshold speed, meters per second

        cvelocity = len(rgvelocityLeft)
        if (cvelocity < 1):
            return []

        rgikeyframe = []
        rgikeyframeLeft = []
        rgikeyframeRight = []
        fLeftIsMoving = abs(rgvelocityLeft[0]) >= speedThreshold
        fRightIsMoving = abs(rgvelocityRight[0]) >= speedThreshold
        speedLeftPrev = abs(rgvelocityLeft[0])
        speedRightPrev = abs(rgvelocityRight[0])

        for ikeyframe in range(1, cvelocity):
            fIsKeyFrame = False

            # Locate first side that crosses the threshold
            speedLeft = abs(rgvelocityLeft[ikeyframe])
            speedRight = abs(rgvelocityRight[ikeyframe])
            fLeftIsMovingT = False
            fRightIsMovingT = False
            if (speedLeft >= speedRight):
                fLeftIsMovingT = (speedLeft >= speedThreshold2)
            else:
                fRightIsMovingT = (speedRight >= speedThreshold2)

            if (not fLeftIsMoving and fLeftIsMovingT):
                fIsKeyFrame = True
                rgikeyframeLeft.append(ikeyframe);
            if (not fRightIsMoving and fRightIsMovingT):
                fIsKeyFrame = True
                rgikeyframeRight.append(ikeyframe);

            if fIsKeyFrame:
                rgikeyframe.append(ikeyframe)

            fLeftIsMoving = fLeftIsMovingT
            fRightIsMoving = fRightIsMovingT

            speedLeftPrev = speedLeft
            speedRightPrev = speedRight

        self.data.left.rgiMovementChanges = rgikeyframeLeft
        self.data.right.rgiMovementChanges = rgikeyframeRight
        # print("rgvelocityLeft = ", rgvelocityLeft)
        # print("rgikeyframeLeft = ", rgikeyframeLeft)
        # print("rgvelocityRight = ", rgvelocityRight)
        # print("rgikeyframeRight = ", rgikeyframeRight)

        return rgikeyframe



    #------------------------------------------------------------------------------
    # Returns a transform to convert a lower body joint position from Kinect depth
    # camera coordinates to hip-origin coordinates.
    # 
    def calculateHipOriginTransform(self, bodyEntry):
        hipL = self.toVector(bodyEntry['hipL'])
        hipR = self.toVector(bodyEntry['hipR'])
        spM = self.toVector(bodyEntry['spineM'])

        # convert kinect space to spherical coordinate
        # 1. normal vector of plane defined by hipR, hipL and spineM
        hip = np.zeros((3,3))
        v1 = hipL - hipR
        v2 = spM - hipR
        hip[0] = np.cross(v1,v2)#x axis
        hip[1] = v1#y axis
        hip[2] = np.cross(hip[0],hip[1])#z axis
        nv = np.zeros((3,3))
        nv[0] = lp.norm1d(hip[0])
        nv[1] = lp.norm1d(hip[1])
        nv[2] = lp.norm1d(hip[2])

        # 2. generate the rotation matrix for
        # converting point from Kinect space to Euclid space, then spherical
        return np.linalg.inv(np.transpose(nv))


    #------------------------------------------------------------------------------
    # Returns the foot direction and level given the spherical coordinates.
    #
    def convertAnglesToDirLevelSupport(self, degreesPhi, degreesTheta, degreesKnee):
        dirlevel = [FootDir.Place, FootLevel.Normal]

        # Determine direction based on azimuthal angle, degreesPhi (-180, 180] where 0 = forward, 90 = left, 180 = backward, -90 = right
        if (degreesPhi <= 22.5 and degreesPhi >= 0) or (degreesPhi < 0 and degreesPhi > -22.5):
            dirlevel[0] = FootDir.Forward
        elif (degreesPhi <= 67.5 and degreesPhi > 22.5):
            dirlevel[0] = FootDir.LeftForward
        elif (degreesPhi <= 112.5 and degreesPhi > 67.5):
            dirlevel[0] = FootDir.Left
        elif (degreesPhi <= 157.5 and degreesPhi > 112.5):
            dirlevel[0] = FootDir.LeftBackward
        elif (degreesPhi <= -157.5 and degreesPhi > -180) or (degreesPhi <= 180 and degreesPhi > 157.5):
            dirlevel[0] = FootDir.Backward
        elif (degreesPhi <= -112.5 and degreesPhi > -157.5):
            dirlevel[0] = FootDir.RightBackward
        elif (degreesPhi <= -67.5 and degreesPhi > -112.5):
            dirlevel[0] = FootDir.Right
        else:
            dirlevel[0] = FootDir.RightForward

        # Handle leg straight up or down based on polar angle, degreesTheta[0, 180], 0 = straight up, 180 = straight down
        #if (degreesTheta < 22.5) or (degreesTheta > 157.5):
        if (degreesTheta < 45) or (degreesTheta > 135.0):
            dirlevel[0] = FootDir.Place

        # Determine level based on knee angle, degreesKnee[0, 180], 0 = fully bent leg, 180 = straight leg
        #if degreesKnee < 150.0:
        if degreesKnee < 125.0:
            dirlevel[1] = FootLevel.High
        else:
            dirlevel[1] = FootLevel.Normal

        return dirlevel



    #------------------------------------------------------------------------------
    # Returns the Laban support column direction and level strings given the foot
    # direction and level.
    # 
    def convertDirLevelFrameToLabanSupport(self, dirlevelframe, framePrev = None):
        dirlevelRight = dirlevelframe[0]
        dirRight = dirlevelRight[0]
        levelRight = dirlevelRight[1]

        dirlevelLeft = dirlevelframe[1]
        dirLeft = dirlevelLeft[0]
        levelLeft = dirlevelLeft[1]

        # If a level is the same as in the previous frame and the other side isn't 'None', set level to 'Hold'
        if framePrev is not None:
            if (levelLeft != FootLevel.NoLevel) and (levelRight == framePrev[0][1]):
                if (levelRight != FootLevel.NoLevel):
                    levelRight = FootLevel.Hold
            if (levelRight != FootLevel.NoLevel) and (levelLeft == framePrev[1][1]):
                if (levelLeft != FootLevel.NoLevel):
                    levelLeft = FootLevel.Hold

        return [
            [self.mpFootDirLabanString[dirRight], self.mpFootLevelLabanString[levelRight]],
            [self.mpFootDirLabanString[dirLeft], self.mpFootLevelLabanString[levelLeft]],
            ]



    #------------------------------------------------------------------------------
    # Returns the Laban "direction:level" string given a Kinect depth camera
    # coordinate position and re-orient transform.
    # 
    def convertPosToAngles(self, vecHip, vecKnee, vecAnkle, transform):
        # print("  hip = ", vecHip)
        # print("  knee = ", vecKnee)
        # print("  ankle = ", vecAnkle)

        degrees = lp.to_sphere(np.dot(transform, vecAnkle - vecHip))

        vecKneeToHip = vecHip - vecKnee
        vecKneeToAnkle = vecAnkle - vecKnee
        magKneeToHip = np.sqrt(vecKneeToHip.dot(vecKneeToHip))
        magKneeToAnkle = np.sqrt(vecKneeToAnkle.dot(vecKneeToAnkle))
        degreesKnee = math.degrees(math.acos(np.dot(vecKneeToHip, vecKneeToAnkle) / (magKneeToHip * magKneeToAnkle))) # Angle of the knee

        return degrees[0], degrees[1], degreesKnee



    #------------------------------------------------------------------------------
    # Returns the Laban "direction:level" string given a Kinect depth camera
    # coordinate position and re-orient transform.
    # 
    def convertPosToLabanDirLevelSupport(self, vecHip, vecKnee, vecAnkle, transform):
        # print("  hip = ", vecHip)
        # print("  knee = ", vecKnee)
        # print("  ankle = ", vecAnkle)

        degreesPhi, degreesTheta, degreesKnee = self.convertPosToAngles(vecHip, vecKnee, vecAnkle, transform)

        # print("  phi = ", degreesPhi, ", theta = ", degreesTheta, ", knee = ", degreesKnee)
        dirlevel = self.convertAnglesToDirLevelSupport(degreesPhi, degreesTheta, degreesKnee)
        # print("  dir = ", dirlevel[0], ", level = ", dirlevel[1])

        return dirlevel



    #------------------------------------------------------------------------------
    # Returns the Laban support column direction and level strings given the foot
    # direction and level.
    # 
    def convertDirLevelToLabanSupport(self, footdir, footlevel):
        return [self.mpFootDirLabanString[footdir], self.mpFootLevelLabanString[footlevel]]



    #------------------------------------------------------------------------------
    # Calculates the Laban script for the lower body and returns the merger of that
    # and the specified existing script.
    # 
    def convertToLabanotation(self, jointD, timeSExisting, labanscriptExisting, forceReset):
        if (forceReset):
            self.reset()
        else:
            self.data = self.Data()

        # Set graph display X-axis to joint frame time range in seconds
        cjointframe = len(jointD)
        rgsec = [(jointD[i]['timeS'][0] / 1000.0) for i in range(cjointframe)]
        self.data.rgsecTimestamp = rgsec

        # Get the keyframes for the lower body
        rgiPotentialKeyFrames = self.analyzePotentialKeyFrames(jointD, rgsec)

        # Generate the Laban script for just the lower body
        timeSLowerBody, labanscriptLowerBody = self.generateLabanScript(rgiPotentialKeyFrames, jointD)

        # Save keyframe markers for graph display
        rgsecKeyFrame = []
        rgpairsecFIsMoving = []
        secMovingStart = -1
        if (self.graph is not None):
            for i in range(1, len(timeSLowerBody)): # Skip first entry which is always inserted to establish the initial support values
                sec = timeSLowerBody[i] / 1000.0
                rgsecKeyFrame.append(sec)

                fIsMoving = (labanscriptLowerBody[i][0][1] != 'Hold') or (labanscriptLowerBody[i][1][1] != 'Hold')
                if fIsMoving:
                    if secMovingStart < 0:
                        secMovingStart = sec
                elif secMovingStart >= 0:
                    rgpairsecFIsMoving.append((secMovingStart, sec))
                    secMovingStart = -1

        self.data.keyframe.rgsec = rgsecKeyFrame
        self.data.keyframe.rgpairsecFIsMoving = rgpairsecFIsMoving

        # Insert hold after initial support frame so we won't generate an initial movement when merging with the existing script
        if (len(timeSLowerBody) > 1) and (timeSLowerBody[1] > timeSExisting[1]):
            timeSLowerBody.insert(1, timeSLowerBody[0] + 1)
            labanscriptLowerBody.insert(1, [[labanscriptLowerBody[0][0][0], 'Hold'], [labanscriptLowerBody[0][1][0], 'Hold']])

        # Save script times and frames for export
        self.timeSLowerBody = timeSLowerBody
        self.labanscriptLowerBody = labanscriptLowerBody

        # Merge with the Laban script for the upper body
        if (self.dictOptions['dataset'] == 'feet'):
            timeSRet, labanscriptRet = self.mergeLabanScript(timeSLowerBody, labanscriptLowerBody, timeSExisting, labanscriptExisting)

            # Convert duplicated frames to holds, if necessary
            entryRightPrev = labanscriptRet[0][4]
            entryLeftPrev = labanscriptRet[0][5]
            for i in range(1, len(labanscriptRet)):
                scriptframe = labanscriptRet[i]
                entryRight = scriptframe[4]
                entryLeft = scriptframe[5]

                if entryRight == entryRightPrev:
                    if (entryRight[1] != 'None') and (entryLeft[1] != 'None'):
                        scriptframe[4] = [entryRight[0], 'Hold']
                else:
                    entryRightPrev = entryRight
                if entryLeft == entryLeftPrev:
                    if (entryLeft[1] != 'None') and (entryRight[1] != 'None'):
                        scriptframe[5] = [entryLeft[0], 'Hold']
                else:
                    entryLeftPrev = entryLeft

        else:
            timeSRet = timeSExisting
            labanscriptRet = labanscriptExisting

        # print("Merged script:")
        # for iframe in range(len(labanscriptRet)):
        #     print("Frame", iframe, "(" + str(timeSRet[iframe]) + ")", labanscriptRet[iframe])

        # Update the graph with our data
        if (self.graph is not None):
            self.graph.setData(self.data)

        return timeSRet, labanscriptRet



    #------------------------------------------------------------------------------
    # Analyze the movement change points and filter out spurious entries that have
    # insufficient movement
    # 
    def filterMovementIndices(self, rgiMovementChanges, rgfIsMoving, rgdm, rgmsec):
        rgiNew = []
        rgfNew = []
        cii = len(rgiMovementChanges)
        ciiLast = cii - 1
        msecNext = rgmsec[0] if (cii > 0) else 0
        for ii in range(cii):
            i = rgiMovementChanges[ii]
            dm = rgdm[i]
            msec = msecNext
            if (ii >= ciiLast):
               fAdd = True
            else:
                msecNext = rgmsec[rgiMovementChanges[ii + 1]]
                fAdd = ((msecNext - msec) >= 100) or (abs(dm) > 0.05)

            if fAdd:
                rgiNew.append(i)
                rgfNew.append(rgfIsMoving[ii])

        return rgiNew, rgfNew



    #------------------------------------------------------------------------------
    # Given a Kinect skeleton joint frame, return the directions and levels for the
    # support columns
    # 
    def generateDirLevelFrame(self, bodyFrame, matToHip, iframe):
        vecHipLeft = self.toVector(bodyFrame['hipL'])
        vecHipRight = self.toVector(bodyFrame['hipR'])
        vecKneeLeft = self.toVector(bodyFrame['kneeL'])
        vecKneeRight = self.toVector(bodyFrame['kneeR'])
        vecAnkleLeft = self.toVector(bodyFrame['ankleL'])
        vecAnkleRight = self.toVector(bodyFrame['ankleR'])

        # print("ikeyframe ", ikeyframe, ", iframe = ", rgiKeyFrames[ikeyframe], ", time = ", bodyFrame['timeS'][0] - jointD[0]['timeS'][0])

        # Calculate direction and level for each foot
        dirlevelframe = [
            self.convertPosToLabanDirLevelSupport(vecHipRight, vecKneeRight, vecAnkleRight, matToHip),
            self.convertPosToLabanDirLevelSupport(vecHipLeft, vecKneeLeft, vecAnkleLeft, matToHip)
            ]

        # Determine which side of the body has the weight of the body
        algSupport = self.dictOptions['support']['algorithm']
        if algSupport == 'kneeangle':
            # The foot supporting the body has the least bent knee
            degreesPhi, degreesTheta, degreesKneeR = self.convertPosToAngles(vecHipRight, vecKneeRight, vecAnkleRight, matToHip)
            degreesPhi, degreesTheta, degreesKneeL = self.convertPosToAngles(vecHipLeft, vecKneeLeft, vecAnkleLeft, matToHip)
            if (abs(degreesKneeR - degreesKneeL) < 5):
                # Feet are even
                dirlevelframe[0][1] = FootLevel.Normal
                dirlevelframe[1][1] = FootLevel.Normal
            elif (degreesKneeR < degreesKneeL):
                # Right foot is raised
                dirlevelframe[0][1] = FootLevel.High
                dirlevelframe[1][1] = FootLevel.Normal
            else:
                # Left foot is raised
                dirlevelframe[0][1] = FootLevel.Normal
                dirlevelframe[1][1] = FootLevel.High

        elif algSupport == 'lowesty':
            #Determine which foot is raised, if any
            yAnkleR = vecAnkleRight[1]
            yAnkleL = vecAnkleLeft[1]
            if (abs(yAnkleR - yAnkleL) < 0.03):
                # Feet are even
                dirlevelframe[0][1] = FootLevel.Normal
                dirlevelframe[1][1] = FootLevel.Normal
            elif (yAnkleR > yAnkleL):
                # Right foot is raised
                dirlevelframe[0][1] = FootLevel.High
                dirlevelframe[1][1] = FootLevel.Normal
            else:
                # Left foot is raised
                dirlevelframe[0][1] = FootLevel.Normal
                dirlevelframe[1][1] = FootLevel.High

        elif algSupport == 'lowestposition':
            # The support weight is on the foot with the lowest position
            if (dirlevelframe[0][1] < dirlevelframe[1][1]):
                # Weight is on the right foot so there's no weight on the left foot
                dirlevelframe[1] = [FootDir.Place, FootLevel.NoLevel]
            elif (dirlevelframe[0][1] > dirlevelframe[1][1]):
                # Weight is on the left foot, do similar to right foot above
                dirlevelframe[0] = [FootDir.Place, FootLevel.NoLevel]
            # else # Weight is on both feet

        elif algSupport == 'correlateddirection':
            # The foot that is moving in the same direction as the body is without weight

            vecVelBody = np.linalg.norm(self.data.rgvecVelocityBody[iframe])
            vecVelRight = np.linalg.norm(self.data.right.rgvec[iframe])
            vecVelLeft = np.linalg.norm(self.data.left.rgvec[iframe])
            correlationRight = np.dot(vecVelBody, vecVelRight)
            correlationLeft = np.dot(vecVelBody, vecVelLeft)

            if (correlationRight < 0.2) and (correlationLeft < 0.2):
                pass # Weight is on both feet
            elif correlationLeft > correlationRight:
                dirlevelframe[1] = [FootDir.Place, FootLevel.NoLevel] # No weight on left
            else:
                dirlevelframe[0] = [FootDir.Place, FootLevel.NoLevel] # No weight on right

        return dirlevelframe



    #------------------------------------------------------------------------------
    # Analyze the Kinect skeleton joint data and return the Laban script for the
    # lower body
    # 
    def generateLabanScript(self, rgiKeyFrames, jointD):
        # Calculate the velocity vector of the body and of each foot
        gausskernel = wavfilter.gaussFilter(self.gauss_window_size, self.gauss_sigma)
        self.data.rgvecVelocityBody = []
        self.data.rgvecVelocityBody.append(np.zeros(3))
        self.data.right.rgvec = [np.zeros(3)]
        self.data.left.rgvec = [np.zeros(3)]


        rgposBody = wavfilter.calcFilter(np.asarray([self.toVector(jointD[i]['spineM']) for i in range(0, len(jointD))]), gausskernel)
        rgposRight = wavfilter.calcFilter(np.asarray([self.toVector(jointD[i]['ankleR']) for i in range(0, len(jointD))]), gausskernel)
        rgposLeft = wavfilter.calcFilter(np.asarray([self.toVector(jointD[i]['ankleL']) for i in range(0, len(jointD))]), gausskernel)

        vecLastS = rgposBody[0]
        vecLastR = rgposRight[0]
        vecLastL = rgposLeft[0]
        rgsec = self.data.rgsecTimestamp
        secLast = rgsec[0]
        for i in range(1, len(jointD)):
            sec = rgsec[i]
            vecS = rgposBody[i]
            vecR = rgposRight[i]
            vecL = rgposLeft[i]

            self.data.rgvecVelocityBody.append((vecS - vecLastS) / (sec - secLast))
            self.data.right.rgvec.append((vecR - vecLastR) / (sec - secLast))
            self.data.left.rgvec.append((vecL - vecLastL) / (sec - secLast))

            vecLastS = vecS
            vecLastR = vecR
            vecLastL = vecL
            secLast = sec

        # Calculate transform to convert from Kinect depth camera coordinates
        # to hip-relative coordinates
        matToHip = self.calculateHipOriginTransform(jointD[0])

        # Create Labanotation script from keyframes
        # Add first body frame as the initial position
        bodyframe = jointD[0]
        dirlevelframePrev = self.generateDirLevelFrame(bodyframe, matToHip, 0)
        rgscriptframe = [self.convertDirLevelFrameToLabanSupport(dirlevelframePrev)]
        rgtimeS = [bodyframe['timeS'][0]]

        if len(rgiKeyFrames) > 1:
            bodyframe = jointD[rgiKeyFrames[0]]
            dirlevelframe = self.generateDirLevelFrame(bodyframe, matToHip, rgiKeyFrames[0])
            timeS = bodyframe['timeS'][0]
            # print("ikeyframe {0}: {1} ms, [R, L]: {2}".format(0, timeS, dirlevelframe))

            for ikeyframe in range(1, len(rgiKeyFrames)):
                # Get the next frame
                bodyFrameNext = jointD[rgiKeyFrames[ikeyframe]]
                dirlevelframeNext = self.generateDirLevelFrame(bodyFrameNext, matToHip, rgiKeyFrames[ikeyframe])
                timeSNext = bodyFrameNext['timeS'][0]
                scriptframe = None

                # print("ikeyframe {0}: {1} ms, [R, L]: {2}".format(ikeyframe, timeSNext, dirlevelframeNext))

                # Add frame to script if the frame is different from previous
                if dirlevelframe != dirlevelframePrev:
                    rgscriptframe.append(self.convertDirLevelFrameToLabanSupport(dirlevelframe))
                    rgtimeS.append(timeS)
                    dirlevelframePrev = dirlevelframe

                # Advance to next frame
                dirlevelframe = dirlevelframeNext
                timeS = timeSNext

            # Append the last keyframe
            if dirlevelframe != dirlevelframePrev:
                rgscriptframe.append(self.convertDirLevelFrameToLabanSupport(dirlevelframe))
                rgtimeS.append(timeS)

        # print("Script for lower body before walking adjustment:")
        # for iframe in range(len(rgscriptframe)):
        #     print("Frame", iframe, "(" + str(rgtimeS[iframe]) + ")", rgscriptframe[iframe])


        if self.dictOptions['support']['fContinuousWalking']:
            # In order to generate a continuous script for walking, remove any frames holding both sides
            i = len(rgscriptframe) - 1
            while i > 0:
                scriptframe = rgscriptframe[i]
                if (scriptframe[0][1] != 'Hold') or (scriptframe[1][1] != 'Hold'):
                    break
                i -= 1

            fPrevWasHold = False
            i -= 1
            while i > 0:
                scriptframe = rgscriptframe[i]
                if (scriptframe[0][1] == 'Hold') and (scriptframe[1][1] == 'Hold'):
                    if fPrevWasHold:
                        rgscriptframe.pop(i)
                        rgtimeS.pop(i)
                    else:
                       fPrevWasHold = True
                elif (i > 1) and (i < 5) and (scriptframe[0][1] == 'Normal') and (scriptframe[1][1] == 'Normal'):
                    rgscriptframe.pop(i)
                    rgtimeS.pop(i)
                else:
                    fPrevWasHold = False;
                i -= 1

        # print("Script for lower body:")
        # for iframe in range(len(rgscriptframe)):
        #     print("Frame", iframe, "(" + str(rgtimeS[iframe]) + ")", rgscriptframe[iframe])


        return rgtimeS, rgscriptframe



    #------------------------------------------------------------------------------
    # Returns the indices into the velocity array where motion changes.
    # 
    def identifyMovementChanges(self, rgvelocity, rgpos, minimadetectiondata):
        if self.dictOptions['algMotionChange'] == 'localminima':
            return self.identifyMovementChangesScaleSpace(rgvelocity, rgpos, minimadetectiondata)
        else:
            return self.identifyMovementChangesThreshold(rgvelocity)



    #------------------------------------------------------------------------------
    # Returns the indices into the velocity array where the velocity have local
    # minima.
    # 
    def identifyMovementChangesScaleSpace(self, rgvelocity, rgpos, jointdata):
        if jointdata.scaled_space is None:
            jointdata.scaled_space = self.Data.MinimaDetection.JointData.ScaledSpace()
        scaled_space = jointdata.scaled_space


        gaussParamWide = self.dictOptions['minimadetection']['wide']
        gauss_large = wavfilter.gaussFilter(gaussParamWide.window_size, gaussParamWide.sigma)
        rgspeed = [abs(v) for v in rgvelocity]
        rgspeedFilteredWide = wavfilter.calcFilter(np.asarray(rgspeed), gauss_large)
        scaled_space.rgvFilteredWide = rgspeedFilteredWide

        gaussParamNarrow = self.dictOptions['minimadetection']['narrow']
        gauss_small = wavfilter.gaussFilter(gaussParamNarrow.window_size, gaussParamNarrow.sigma)
        #rgspeedFilteredNarrow = wavfilter.calcFilter(rgvelocity, gauss_small)
        rgspeedFilteredNarrow = wavfilter.calcFilter(np.asarray(rgspeed), gauss_small)
        scaled_space.rgvFilteredNarrow = rgspeedFilteredNarrow

        infl = accessory.inflection(rgspeedFilteredWide)
        scaled_space.inflection = infl

        corner = cluster.b_peak_dect(rgspeedFilteredWide, [])
        scaled_space.peak = corner

        real_corner = []
        for j in corner:
            right = 0
            left = 0
            for k in range(len(infl)):
                if infl[k] > j:
                    left = infl[k-1]
                    right = infl[k]
                    break
            min_val = rgspeedFilteredNarrow[left]
            min_ptr = left
            for k in range(left,right+1):
                if rgspeedFilteredNarrow[k] < min_val:
                    min_val = rgspeedFilteredNarrow[k]
                    min_ptr = k
            real_corner.append(min_ptr)

        scaled_space.filtered = real_corner

        return real_corner



    #------------------------------------------------------------------------------
    # Returns the indices into the velocity array where the velocity first drops
    # below the speed threshold or rises above speed threshold
    # 
    def identifyMovementChangesThreshold(self, rgvelocity):
        speedThreshold = self.dictOptions['speedThresholdMovementChange']
        rgi = []
        fIsMoving = (abs(rgvelocity[0]) > speedThreshold)
        cvelocity = len(rgvelocity)
        for ivelocity in range(1, cvelocity):
            speed = abs(rgvelocity[ivelocity])
            fIsMovingT = (speed > speedThreshold)
            if (fIsMovingT != fIsMoving):
                rgi.append(ivelocity)
                fIsMoving = fIsMovingT

        return rgi



    #------------------------------------------------------------------------------
    # Returns the length of a vector
    # 
    def lenVec(self, vec):
        sumSquared = 0
        for v in vec:
            sumSquared = sumSquared + v * v
        return math.sqrt(sumSquared)



    #------------------------------------------------------------------------------
    # Returns the merger of two keyframe index lists
    # 
    def mergeKeyframeLists(self, rgikeyframe1, rgikeyframe2):
        rgikeyframeMerged = []
        i1 = 0
        i2 = 0
        ci1 = len(rgikeyframe1)
        ci2 = len(rgikeyframe2)
        ikeyframeLast = 0
        while ((i1 < ci1) and (i2 < ci2)):
            ikeyframe1 = rgikeyframe1[i1]
            ikeyframe2 = rgikeyframe2[i2]
            if (ikeyframe1 < ikeyframe2):
                # 1 index is lower
                ikeyframe = ikeyframe1
                i1 += 1
            elif (ikeyframe1 > ikeyframe2):
                # 2 index is lower
                ikeyframe = ikeyframe2
                i2 += 1
            else:
                # Both indices are the same, add only one
                ikeyframe = ikeyframe1
                i1 += 1
                i2 += 1

            if ikeyframe != ikeyframeLast:
                rgikeyframeMerged.append(ikeyframe)
                ikeyframeLast = ikeyframe

        # Append any 1 overs
        while (i1 < ci1):
            rgikeyframeMerged.append(rgikeyframe1[i1])
            i1 += 1
        while (i2 < ci2):
            rgikeyframeMerged.append(rgikeyframe2[i2])
            i2 += 1

        # Return the indices as the keyframes
        return rgikeyframeMerged



    #------------------------------------------------------------------------------
    # Returns the merger of the given lower body Laban frame fields with an
    # existing Laban frame
    # 
    def mergeLabanData(self, labandataLowerBody, labandataExisting):
        labandataMerged = OrderedDict()

        cExisting = len(labandataExisting)
        cLB = len(labandataLowerBody)

        if cLB == 0:
            # No lower body, just return upper body
            labandataMerged = labandataExisting
        elif cExisting == 0:
            # Copy lower body script to merged using default position for upper body
            frameUpperDefault['head'] = ['Forward', 'Normal']
            frameUpperDefault['right elbow'] = ['Place', 'Low']
            frameUpperDefault['right wrist'] = ['Place', 'Low']
            frameUpperDefault['left elbow'] = ['Place', 'Low']
            frameUpperDefault['left wrist'] = ['Place', 'Low']

            for iLB in range(0, cLB):
                key = 'Position' + str(iLB)
                frameLB = labandataLowerBody[key].copy()
                for keyExisting in list(frameUpperDefault):
                    frameLB[keyExisting] = frameUpperDefault[keyExisting]
                labandataMerged[key] = frameLB
        else:
            # Merge the two scripts
            listKeyExisting = list(labandataExisting)
            listKeyLB = list(labandataLowerBody)
            frameExistingPrev = labandataExisting[listKeyExisting[0]]
            frameLBPrev = labandataLowerBody[listKeyLB[0]]

            iExisting = 0
            iLB = 0
            iMerged = 0

            while (iLB < cLB) and (iExisting < cExisting):
                frameExisting = labandataExisting[listKeyExisting[iExisting]]
                frameLB = labandataLowerBody[listKeyLB[iLB]]
                timeStartExisting = frameExisting['start time']
                timeStartLB = frameLB['start time']

                if timeStartLB == timeStartExisting:
                    # Frame have the same times--merge the two
                    frameMerged = self.mergeLabanDataFrames(timeStartLB, frameLB, frameExisting)
                    frameExistingPrev = frameExisting
                    frameLBPrev = frameLB
                    iExisting += 1
                    iLB += 1
                elif timeStartLB < timeStartExisting:
                    # Lower body frame is before existing--merge lower body frame with previous existing frame
                    frameMerged = self.mergeLabanDataFrames(timeStartLB, frameLB, frameExistingPrev)
                    frameLBPrev = frameLB
                    iLB += 1
                else:
                    # Existing frame is before lower body--merge existing frame with previous lower body frame
                    frameMerged = self.mergeLabanDataFrames(timeStartExisting, frameLBPrev, frameExisting)
                    frameExistingPrev = frameExisting
                    iExisting += 1

                labandataMerged['Position' + str(iMerged)] = frameMerged
                iMerged += 1

            # Add unmerged frames to merged script
            for key in listKeyExisting[iExisting:]:
                frameExisting = labandataExisting[key]
                labandataMerged['Position' + str(iMerged)] = self.mergeLabanDataFrames(frameExisting['start time'], frameLBPrev, frameExisting)
                iMerged += 1

            for key in listKeyLB[iLB:]:
                frameLB = labandataLB[key]
                labandataMerged['Position' + str(iMerged)] = self.mergeLabanDataFrames(frameLB['start time'], frameLB, frameExistingPrev)
                iMerged += 1

        return labandataMerged


    #------------------------------------------------------------------------------
    # Returns the merger of the given lower body Laban data frame fields with an
    # existing Laban data frame
    #
    def mergeLabanDataFrames(self, timeStarting, frameLowerBody, frameExisting):
        frameMerged = OrderedDict()
        frameMerged['start time'] = timeStarting
        if frameExisting['duration'] == '-1':
            frameMerged['duration'] = frameLowerBody['duration']
        else:
            frameMerged['duration'] = frameExisting['duration']

        for key in list(frameLowerBody):
            if (key != 'start time') and (key != 'duration') and (frameLowerBody[key][1] != 'None'): # Omit limbs with a level of 'None'
                frameMerged[key] = frameLowerBody[key]

        for key in list(frameExisting):
            if (key != 'start time') and (key != 'duration'):
                frameMerged[key] = frameExisting[key]

        return frameMerged



    #------------------------------------------------------------------------------
    # Returns the merger of the given lower body Laban frame fields with an
    # existing Laban frame
    # 
    def mergeLabanFrame(self, frameLowerBody, frameExisting):
        frameMerged = frameExisting.copy()

        # Extend existing frame with the support column fields
        while (len(frameMerged) < 6):
            frameMerged.append(["Place","Normal"])

        # Set the support column fields
        frameMerged[4] = frameLowerBody[0]
        frameMerged[5] = frameLowerBody[1]

        return frameMerged



    #------------------------------------------------------------------------------
    # Returns the merger of a Laban script for the lower body with an existing
    # Laban script.
    def mergeLabanScript(self, timeSNew, labanscriptNew, timeSExisting, labanscriptExisting):
        cframeExisting = len(labanscriptExisting)
        cframeNew = len(labanscriptNew)

        if (cframeNew < 1):
            # No lower body script, return the existing script
            labanscriptMerged = labanscriptExisting
            timeSMerged = timeSExisting
        elif (cframeExisting < 1):
            # No existing script, use placeholders for upper body
            labanscriptMerged = []
            for iframeNew in range(len(labanscriptNew)):
                labanscriptMerged[iframeNew] = [["place", "low"], ["place", "low"], ["place", "low"], ["place", "low"], labanscriptNew[iframeNew][0], labanscriptNew[iframeNew][1]]
            timeSMerged = timeSNew
        else:
            labanscriptMerged = []
            timeSMerged = []

            iframeExisting = 0
            frameExisting = labanscriptExisting[0]
            starttimeExisting = timeSExisting[0]

            iframeNew = 0
            frameNew = labanscriptNew[0]
            frameNewLast = frameNew
            starttimeNew = timeSNew[0]

            frameLast = frameExisting

            while (iframeNew < cframeNew) and (iframeExisting < cframeExisting):
                fAdvanceExisting = False
                fAdvanceNew = False

                if (starttimeExisting < starttimeNew):
                    # Existing frame is before the new frame copy to merged script and go to next existing frame
                    frameLast = self.mergeLabanFrame(frameNewLast, frameExisting)
                    starttime = starttimeExisting
                    fAdvanceExisting = True
                elif starttimeExisting == starttimeNew:
                    # Existing frame is at the same time as the new frame; merge the new frame with the
                    # existing frame
                    frameLast = self.mergeLabanFrame(frameNew, frameExisting)
                    starttime = starttimeExisting
                    fAdvanceExisting = True
                    fAdvanceNew = True;
                else:
                    # Existing frame is after the new frame, merge the new frame with the last one
                    frameLast = self.mergeLabanFrame(frameNew, frameLast)
                    starttime = starttimeNew
                    fAdvanceNew = True;

                # Add the frame to the merged script
                labanscriptMerged.append(frameLast)
                timeSMerged.append(starttime)

                # Advance to next existing frame
                if fAdvanceExisting:
                    iframeExisting += 1
                    if iframeExisting < cframeExisting:
                        frameExisting = labanscriptExisting[iframeExisting]
                        starttimeExisting = timeSExisting[iframeExisting]

                # Advance to next new frame
                if fAdvanceNew:
                    frameNewLast = frameNew
                    iframeNew += 1
                    if iframeNew < cframeNew:
                        frameNew = labanscriptNew[iframeNew]
                        starttimeNew = timeSNew[iframeNew]

            # Merge any left over existing frames (timestamps after the last new frame)
            while iframeExisting < cframeExisting:
                frameLast = self.mergeLabanFrame(frameNewLast, labanscriptExisting[iframeExisting])
                labanscriptMerged.append(frameLast)
                timeSMerged.append(timeSExisting[iframeExisting])
                iframeExisting += 1

            # Merge any left over new frames (timestamps after the last existing frame)
            while iframeNew < cframeNew:
                frameLast = self.mergeLabanFrame(labanscriptNew[iframeNew], frameLast)
                labanscriptMerged.append(frameLast)
                timeSMerged.append(timeSNew[iframeNew])
                iframeNew += 1

        return timeSMerged, labanscriptMerged


    #------------------------------------------------------------------------------
    # Returns the reflected foot direction and level
    #
    def reflectFootDirLevel(self, footdirlevel):
        return [self.reflectFootDir(footdirlevel[0]), self.reflectFootLevel(footdirlevel[1])]



    #------------------------------------------------------------------------------
    # Returns the foot direction reflected through the center (forward becomes
    # backward, left-backward becomes right-forward, etc.)
    #
    def reflectFootDir(self, footdir):
        ipos = int(footdir) + 4
        if (ipos > FootDir.MAX):
            ipos = ipos - int(FootDir.MAX)
        return FootDir(ipos)



    #------------------------------------------------------------------------------
    # Returns the foot level reflected through "normal" ("low" becomes "high",
    # "high" becomes "low", and anything else does not change.)
    #
    def reflectFootLevel(self, footlevel):
        if (footlevel == FootLevel.Low):
            footlevel = FootLevel.High
        elif (footlevel == FootLevel.High):
            footlevel = FootLevel.Low
        return (footlevel)


    #------------------------------------------------------------------------------
    # Reset this instance and clear any previous calculations
    #
    def reset(self):
        self.data = self.Data()

        if (self.graph is not None):
            self.graph.reset()



    #------------------------------------------------------------------------------
    #
    def saveToJSON(self, labandataOther):
        filePath = settings.checkFileAlreadyExists(settings.application.outputFilePathJson, fileExt=".json", fileTypes=[('json files', '.json'), ('all files', '.*')])
        if (filePath is None):
            return

        # Generate Laban data for JSON export
        labandata = OrderedDict()
        iPosition = 0
        for i in range(0, len(self.timeSLowerBody)):
            frameLeft = self.labanscriptLowerBody[i][0]
            frameRight = self.labanscriptLowerBody[i][1]
            if (frameLeft[1] != 'None') or (frameRight[1] != 'None'):
                frame = OrderedDict()
                frame['start time'] = [str(self.timeSLowerBody[i])]
                frame['duration'] = ['1']
                frame['left support'] = frameLeft
                frame['right support'] = frameRight
                labandata['Position' + str(iPosition)] = frame
                iPosition += 1

        #print("Lower body Laban script for export:");
        #for i in range(0, len(labandata)):
        #    print(labandata['Position' + str(i)]);

        # save JSON script
        file_name = os.path.splitext(os.path.basename(filePath))[0]

        labanjson = OrderedDict()
        labanjson[file_name] = self.mergeLabanData(labandata, labandataOther)

        try:
            with open(filePath,'w') as file:
                json.dump(labanjson, file, indent=2)
                settings.application.logMessage("Labanotation json script was saved to '" + settings.beautifyPath(filePath) + "'")
        except Exception as e:
            strError = e
            settings.application.logMessage("Exception saving Labanotation json script to '" + settings.beautifyPath(filePath) + "': " + str(e))

    #------------------------------------------------------------------------------
    #
    def saveToTXT(self, timeSOther, labanscriptOther):
        filePath = settings.checkFileAlreadyExists(settings.application.outputFilePathTxt, fileExt=".txt", fileTypes=[('text files', '.txt'), ('all files', '.*')])
        if (filePath is None):
            return

        # save text script
        timeSMerged, labanMerged = self.mergeLabanScript(self.timeSLowerBody, self.labanscriptLowerBody, timeSOther, labanscriptOther)
        script = settings.application.labanotation.labanToScript(timeSMerged, labanMerged)

        try:
            with open(filePath,'w') as file:
                file.write(script)
                file.close()
                settings.application.logMessage("Labanotation text script was saved to '" + settings.beautifyPath(filePath) + "'")
        except Exception as e:
            strError = e
            settings.application.logMessage("Exception saving Labanotation text script to '" + settings.beautifyPath(filePath) + "': " + str(e))



    #------------------------------------------------------------------------------
    # Set the graph associated with this algorithm
    #
    def setGraph(self, graph):
        self.graph = graph
        self.graph.dictOptions = self.dictOptions



    #------------------------------------------------------------------------------
    # Returns a Kinect skeleton joint position as a three element vector
    #
    def toVector(self, jointEntry):
        vec = np.zeros(3)
        vec[0] = jointEntry['x']
        vec[1] = jointEntry['y']
        vec[2] = jointEntry['z']
        return vec



    #------------------------------------------------------------------------------
    # Returns the velocities given the positions and times
    #
    def vel1D(self, time, dpos):
        cdpos = len(dpos)
        v = np.zeros(cdpos)
        for i in range(cdpos):
            v[i] = dpos[i] / (time[i] - time[i-1])

        return v



    #------------------------------------------------------------------------------
    # Returns the speed given the 3D position and times
    #
    def speed3D(self, time, pos):
        cpos = len(pos)
        s = np.zeros(cpos)
        for i in range(1, cpos):
            dx = pos[i][0] - pos[i-1][0]
            dy = pos[i][1] - pos[i-1][1]
            dz = pos[i][2] - pos[i-1][2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            s[i] = dist /(time[i]-time[i-1])
        s[0] = s[1]

        return s


