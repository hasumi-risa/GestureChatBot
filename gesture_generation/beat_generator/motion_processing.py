import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d

def load_kinect_csv(csv_file):
    kinect_3d_pose = pd.read_csv(csv_file, header=None)
    pose3d = []
    for i in range(len(kinect_3d_pose)):
        frame_pose = kinect_3d_pose.iloc[i]
        pose = []
        for j in range(len(frame_pose)):
            if j == 0:
                tmp = []
            elif j % 4 == 0:
                pose.append(tmp)
                tmp = []
            else:
                tmp.append(frame_pose[j])
        pose3d.append(pose)
    pose3d = np.array(pose3d)
    return pose3d

# calculate cosine angle among joint1, joint2 and joint3
def calcAngular(joint1, joint2, joint3, kernel_size=1):
    def cos_sim(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # calculate angular
    angles = []
    for i in range(len(joint1)):
        bone1 = joint2[i] - joint3[i]
        bone2 = joint1[i] - joint2[i] 
        angles.append(cos_sim(bone1, bone2))

    fileterd_angles = gaussian_filter1d(angles, kernel_size)
    return np.array(fileterd_angles)

def plotTransition(lists, labels, title=None, save_path=None):
    plt.rcParams["figure.figsize"] = (8, 4)
    for i in range(len(lists)):
        plt.plot(lists[i], label=labels[i])
    plt.legend()
    if title:
        plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

if __name__ == '__main__':
    motion_data = './data/1L6l-FiV4xo_8/1L6l-FiV4xo_8.csv'

    pose3d = load_kinect_csv(motion_data)

    lelbow_angles = calcAngular(pose3d[:,6,:], pose3d[:,5,:], pose3d[:,4,:])
    relbow_angles = calcAngular(pose3d[:,10,:], pose3d[:,9,:], pose3d[:,8,:])
    plotTransition([relbow_angles, lelbow_angles], ['right', 'left'], title='Elbow Angle')

    print()