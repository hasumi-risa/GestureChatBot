import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .tool import wavfilter as wf
from .tool import accessory as ac


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


def energy_function_ijcv(v_l, a_l, v_r, a_r):
    if  len(v_l.shape) == 2 and len(a_l.shape) == 2\
    and len(v_r.shape) == 2 and len(a_r.shape) == 2\
    and v_l.shape[1] ==3 and a_l.shape[1] ==3\
    and v_r.shape[1]==3 and a_r.shape[1]==3\
    and v_l.shape[0]==a_l.shape[0]\
    and a_l.shape[0]==v_r.shape[0]\
    and v_r.shape[0]==a_r.shape[0]:
        row = v_l.shape[0]
        e = np.zeros((4,row)) # right/left, velocity/acceleration

        for i in range(row):
            e[0,i] = np.sqrt((a_l[i][0]**2+a_l[i][1]**2+a_l[i][2]**2)/3.0) # a_l
            e[1,i] = np.sqrt((v_l[i][0]**2+v_l[i][1]**2+v_l[i][2]**2)/3.0) # v_l
            e[2,i] = np.sqrt((a_r[i][0]**2+a_r[i][1]**2+a_r[i][2]**2)/3.0) # a_r
            e[3,i] = np.sqrt((v_r[i][0]**2+v_r[i][1]**2+v_r[i][2]**2)/3.0) # a_r

        # do the normalization then add them toghether
        for i in range(4):
            e[i] = ac.norm(e[i])
            
        total_e = np.zeros((row))
        for i in range(row):
            total_e[i] = e[0,i] + e[2,i] - e[1,i] - e[3,i]

        return total_e
    else:
        print("Energy function input data error!")
        return


def totalEnergy(pose3d):
    # filtered by a Gaussian filter with window-size of 101 and sigma of 10
    # window-size of 61 also works
    gauss_window_size = 31
    gauss_large_sigma = 5
    gauss_small_sigma = 1

    handR = pose3d[:, 10]
    handL = pose3d[:, 6]
    unfilteredTimeS = np.arange(0, len(pose3d)*40, 40)
    unfilteredTimeS[0] = 1

    gauss = wf.gaussFilter(gauss_window_size, gauss_large_sigma)
    handRF = wf.calcFilter(handR, gauss)
    handLF = wf.calcFilter(handL, gauss)

    handRv = ac.vel(unfilteredTimeS, handRF)
    handLv = ac.vel(unfilteredTimeS, handLF)
    handRa = ac.acc(unfilteredTimeS, handRv)
    handLa = ac.acc(unfilteredTimeS, handLv)

    # calculate energy
    energy = energy_function_ijcv(v_l=handLv, a_l=handLa, v_r=handRv, a_r=handRa)

    return energy


if __name__ == "__main__":
    
    csv_file = "D:/TED_videos/segmented_by_gesture/0ZfSOArXbGQ/0ZfSOArXbGQ_10_4/0ZfSOArXbGQ_10_4.csv"
    json_file = "D:/TED_videos/segmented_by_gesture/0ZfSOArXbGQ/0ZfSOArXbGQ_10_4/0ZfSOArXbGQ_10_4.json"
    fps = 25

    pose3d = load_kinect_csv(csv_file)
    with open(json_file, 'r') as f:
        laban = json.load(f)
    laban = laban[list(laban.keys())[0]]
    keyframes = [int(int(laban[key]['start time'][0]) * fps / 1000) for key in list(laban.keys())]
    
    energy = totalEnergy(pose3d)

    # LabanSuiteが出すKeyframeと一致しているかどうかの確認
    # (極大値にマーカーが付いていれば合っている)
    plt.figure()
    plt.plot(np.arange(len(energy)), energy, label='energy', marker="D", markevery=keyframes)
    plt.legend()
    plt.show()



