import os
import sys
import math
import numpy as np
import pandas as pd
from pyquaternion import Quaternion
from scipy.ndimage import gaussian_filter1d

from motion_processing import load_kinect_csv, calcAngular
from src.cmu2kinect import CMUPose2KinectData
from src.plotPose import Plot

    
def plotUpperBody2D(pose3d, save_path, fps=25):
    upper_idx = [2, 20, 4, 5, 6, 8, 9, 10]
    
    pose3d = pose3d[:, upper_idx]
    pose2d = np.delete(pose3d, 2, 2).reshape([-1, len(upper_idx)*2])

    # rotate y axis
    # for frame in range(len(pose2d)):
    #     for joint in range(len(pose2d[frame])):
    #         pose2d[frame][joint][1] *= -1

    poses = []
    for i in range(len(pose2d)):
        pose = []
        for j in range(len(pose2d[i])):
            pose.append(pose2d[i][j])
        poses.append(pose)

    time = np.arange(len(pose2d))
    time = time.reshape(-1, 1)
    poses = np.concatenate([np.array(poses), time], axis=1)

    p = Plot((-0.75, 0.75), (-0.5, 1))
    anim = p.animate(poses, 1000/fps)
    p.save(anim, save_path, fps=fps)

def tangent_angle(u: np.ndarray, v: np.ndarray):
    i = np.inner(u, v)
    n = np.linalg.norm(u) * np.linalg.norm(v)
    c = i / n
    return np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))

def rotate_skel(skel_3d, degree):
    """Rotates the skeleton pose
    skel_3d - 3D pose, xyz per limb endpoint.
    degree - degrees in radian it needs to be rotated.
    """
    quaternion = Quaternion(axis=[0, 1, 0], angle=math.radians(degree))
    rotated_skel = np.copy(skel_3d)
    n_joints = skel_3d.shape[0]
    for i in range(n_joints):
        rotated_skel[i] = quaternion.rotate(skel_3d[i])
    return rotated_skel

def findConnectFrame(pose3d, frame, after_frame, first_interval=10):
    r = pose3d[frame][10]
    l = pose3d[frame][6]
    min_norm = 1000
    min_frame = 0
    for i in range(after_frame+first_interval, len(pose3d)):
        tmp = 0
        tmp += np.linalg.norm(pose3d[i][10] - r)
        tmp += np.linalg.norm(pose3d[i][6] - l)
        if min_norm > tmp:
            min_norm = tmp
            min_frame = i
            print(i)
    return min_frame

if __name__ == '__main__':
    base_beat_file = './data/Beat_1L6l-FiV4xo_8/1L6l-FiV4xo_8.csv'
    save_csv_path = './data/Beat_1L6l-FiV4xo_8_sample/sample_beat_1L6l-FiV4xo_8.csv'
    motion_gaussian = 1
    motion_range = [48, 113]
    fps = 25

    # Extract motion feature
    pose3d = load_kinect_csv(base_beat_file)
    # Extract Beat motion

    # plotUpperBody2D(pose3d, "./tmp.mp4", fps=fps)

    # Rotate to face front
    rotated_pose = np.zeros([pose3d.shape[0], pose3d.shape[1], pose3d.shape[2]])
    for i in range(len(pose3d)):
        angle = tangent_angle(pose3d[i][8] - pose3d[i][4], np.array([1,0,0]))
        rotated_pose[i] = rotate_skel(pose3d[i], angle)


    # connect_frame = findConnectFrame(rotated_pose, motion_range[0])
    pose3d_cut = rotated_pose[motion_range[0]: motion_range[1]]

    concat = np.concatenate([pose3d_cut, pose3d_cut, pose3d_cut, pose3d_cut])
    plotUpperBody2D(concat, "./.tmp/tmp_rotated.mp4", fps=fps)
    
    # Save csv
    CMUPose2KinectData(pose3d_cut, save_csv=save_csv_path, fps=fps, isConvert=False)
    
    print("Output to {}".format(save_csv_path))    

    