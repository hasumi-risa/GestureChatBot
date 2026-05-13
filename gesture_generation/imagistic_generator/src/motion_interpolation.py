import numpy as np
from scipy import interpolate as itpl
import matplotlib.pyplot as plt

# motions.shape = ({motion_num}, {frame_num}, {joint_num}, {xyz})
def linearInterpolation(motions, interpolate_frame=20, mean_pose=None):
    def interpolate(m1, m2, interpolate_frame):
        first_pose = m1[len(m1)-1]
        last_pose = m2[0]

        pose = []
        for i in range(len(first_pose)):
            joint = []
            for j in range(len(first_pose[i])):
                diff = (last_pose[i][j] - first_pose[i][j]) / (interpolate_frame + 1)
                frame = []
                pos = first_pose[i][j]
                for k in range(interpolate_frame):
                    pos += diff
                    frame.append(pos)
                joint.append(frame)
            pose.append(joint)
        pose = np.array(pose)
        pose = pose.transpose(2, 0, 1)
        
        return pose
    
    interpolated_motions = []
    for i in range(len(motions)):
        if i == 0:
            for m in motions[i]:
                interpolated_motions.append(m)
            continue

        interpo_motion = interpolate(motions[i-1], motions[i], interpolate_frame)

        for m in interpo_motion:
            interpolated_motions.append(m)

        for m in motions[i]:
            interpolated_motions.append(m)

    interpolated_motions = np.array(interpolated_motions)
    return interpolated_motions

def splineInterpolation(motions, interpolate_frame=20, mean_pose=None):
    def interpolate(m1, m2, interpolate_frame):
        ip_func = itpl.Akima1DInterpolator

        m1 = m1.transpose(1,0,2)
        m2 = m2.transpose(1,0,2)
        m1_frames = np.arange(m1.shape[1])
        m2_frames = np.arange(m2.shape[1]) + m1.shape[1] + interpolate_frame 
        interpo_frames = np.arange(m1.shape[1], m1.shape[1] + interpolate_frame)
        interpo_motion = []
        for joint in range(len(m1)):
            # pre_points = m1[joint][::int(interpolate_frame/2)]
            # pre_frames = m1_frames[::int(interpolate_frame/2)]
            # post_points = m2[joint][::int(interpolate_frame/2)]
            # post_frames = m2_frames[::int(interpolate_frame/2)]
            pre_points = m1[joint]
            pre_frames = m1_frames
            post_points = m2[joint]
            post_frames = m2_frames
            points = np.concatenate([pre_points, post_points])
            frames = np.concatenate([pre_frames, post_frames])
            
            f_x = ip_func(frames, points[:,0]) 
            f_y = ip_func(frames, points[:,1]) 
            f_z = ip_func(frames, points[:,2]) 

            interpo_points_x = f_x(interpo_frames)
            interpo_points_y = f_y(interpo_frames)
            interpo_points_z = f_z(interpo_frames)
            interpo_motion.append([interpo_points_x, interpo_points_y, interpo_points_z])

            # if joint == 6 or joint == 10:
            #     plt.figure()
            #     plt.scatter(frames, points[:,0])
            #     plt.plot(m1_frames, m1[joint][:,0], c='b', label="motion1")
            #     plt.plot(interpo_frames, interpo_points_x, c='r', label="interpolation")
            #     plt.plot(m2_frames, m2[joint][:,0], c='g', label="motion2")
            #     plt.legend()
            #     plt.savefig("./.tmp/x_{}.png".format(joint))

            #     plt.figure()
            #     plt.scatter(frames, points[:,1])
            #     plt.plot(m1_frames, m1[joint][:,1], c='b', label="motion1")
            #     plt.plot(interpo_frames, interpo_points_y, c='r', label="interpolation")
            #     plt.plot(m2_frames, m2[joint][:,1], c='g', label="motion2")
            #     plt.legend()
            #     plt.savefig("./.tmp/y_{}.png".format(joint))

            #     plt.figure()
            #     plt.scatter(frames, points[:,2])
            #     plt.plot(m1_frames, m1[joint][:,2], c='b', label="motion1")
            #     plt.plot(interpo_frames, interpo_points_z, c='r', label="interpolation")
            #     plt.plot(m2_frames, m2[joint][:,2], c='g', label="motion2")
            #     plt.legend()
            #     plt.savefig("./.tmp/z_{}.png".format(joint))
            
        interpo_motion = np.array(interpo_motion).transpose(2,0,1)
        return interpo_motion

        # pose = []
        # for i in range(len(first_pose)):
        #     joint = []
        #     for j in range(len(first_pose[i])):
        #         diff = (last_pose[i][j] - first_pose[i][j]) / (interpolate_frame + 1)
        #         frame = []
        #         pos = first_pose[i][j]
        #         for k in range(interpolate_frame):
        #             pos += diff
        #             frame.append(pos)
        #         joint.append(frame)
        #     pose.append(joint)
        # pose = np.array(pose)
        # pose = pose.transpose(2, 0, 1)
        
        # return pose
    
    interpolated_motions = []
    for i in range(len(motions)):
        if i == 0:
            for m in motions[i]:
                interpolated_motions.append(m)
            continue

        interpo_motion = interpolate(motions[i-1], motions[i], interpolate_frame)

        for m in interpo_motion:
            interpolated_motions.append(m)

        for m in motions[i]:
            interpolated_motions.append(m)

    interpolated_motions = np.array(interpolated_motions)
    return interpolated_motions

def motionAdjustment(motion, adjust_frame, types=None):
    if len(motion) == adjust_frame:
        adjusted_motion = motion
    
    # Interpolation
    elif len(motion) < adjust_frame:
        adjusted_motion = np.zeros([adjust_frame, 25, 3])
        adjusted_motion[0] = motion[0]
        if types:
            adjusted_types = [""] * adjust_frame
            adjusted_types[0] = types[0]
            adjusted_types[len(adjusted_types)-1] = types[len(types)-1]
        adjusted_motion[len(adjusted_motion)-1] = motion[len(motion)-1]
        interval = (adjust_frame - 2) / (len(motion) - 2)
        interval_sum, count = 0, 0
        e_index, tmp = [], []
        isEmpty = False
        if len(motion) > 2:
            for i in range(1, len(adjusted_motion)-1):
                if i > interval_sum:
                    adjusted_motion[i] = motion[count]
                    if types:
                        adjusted_types[i] = types[count]
                    interval_sum += interval
                    count += 1
                else:
                    e_index.append(i)

        empty_index, tmp = [], []
        for i in range(len(e_index)):
            if i == len(e_index) - 1:
                tmp.append(e_index[i])
                empty_index.append(tmp)
            elif e_index[i]+1 == e_index[i+1]:
                tmp.append(e_index[i])
            else:
                tmp.append(e_index[i])
                empty_index.append(tmp)
                tmp = []

        for e in empty_index:
            unit = (adjusted_motion[e[len(e)-1]+1] - adjusted_motion[e[0]-1]) / (len(e) + 1)
            cnt = 1
            for i in range(e[0], e[len(e)-1] + 1):
                adjusted_motion[i] = adjusted_motion[e[0]-1] + unit*cnt
                if types:
                    adjusted_types[i] = adjusted_types[e[0]-1]
                cnt += 1

    # Sampling
    else:
        adjusted_motion = []
        adjusted_types = []
        decrease_frame = len(motion) - adjust_frame
        if decrease_frame == 1:
            decrease_interval = len(motion) / 2 + 1
        else:
            decrease_interval = len(motion) / decrease_frame
        for i in range(len(motion)):
            if i != 0 and i % decrease_interval == 0:
                continue
            adjusted_motion.append(motion[i])
            if types:
                adjusted_types.append(types[i])
            if len(adjusted_motion) == adjust_frame:
                break
    
    if types:
        return np.array(adjusted_motion), adjusted_types
    else:
        return np.array(adjusted_motion)
        
def motionAndWordsAdjustment(motion, adjust_frame, input_words, nn_words, clusters):
    if len(motion) == adjust_frame:
        return motion, input_words, nn_words, clusters
    
    # Interpolation
    elif len(motion) < adjust_frame:
        adjusted_motion = np.zeros([adjust_frame, 25, 3])
        adjusted_input_words = [""] * adjust_frame
        adjusted_nn_words = [""] * adjust_frame
        adjusted_clusters = [0] * adjust_frame

        adjusted_motion[0] = motion[0]
        adjusted_motion[len(adjusted_motion)-1] = motion[len(motion)-1]

        adjusted_input_words[0] = input_words[0]
        adjusted_nn_words[0] = nn_words[0]
        adjusted_clusters[0] = clusters[0]
        adjusted_input_words[len(adjusted_motion)-1] = input_words[len(motion)-1]
        adjusted_nn_words[len(adjusted_motion)-1] = nn_words[len(motion)-1]
        adjusted_clusters[len(adjusted_motion)-1] = clusters[len(motion)-1]

        interval = (adjust_frame - 2) / (len(motion) - 2)
        interval_sum, count = 0, 0
        e_index, tmp = [], []
        isEmpty = False
        for i in range(1, len(adjusted_motion)-1):
            if i > interval_sum:
                adjusted_motion[i] = motion[count]
                adjusted_input_words[i] = input_words[count]
                adjusted_nn_words[i] = nn_words[count]
                adjusted_clusters[i] = clusters[count]
                interval_sum += interval
                count += 1
            else:
                e_index.append(i)

        empty_index, tmp = [], []
        for i in range(len(e_index)):
            if i == len(e_index) - 1:
                tmp.append(e_index[i])
                empty_index.append(tmp)
            elif e_index[i]+1 == e_index[i+1]:
                tmp.append(e_index[i])
            else:
                tmp.append(e_index[i])
                empty_index.append(tmp)
                tmp = []

        for e in empty_index:
            unit = (adjusted_motion[e[len(e)-1]+1] - adjusted_motion[e[0]-1]) / (len(e) + 1)
            cnt = 1
            for i in range(e[0], e[len(e)-1] + 1):
                adjusted_motion[i] = adjusted_motion[e[0]-1] + unit*cnt
                adjusted_input_words[i] = adjusted_input_words[e[0]-1]
                adjusted_nn_words[i] = adjusted_nn_words[e[0]-1]
                adjusted_clusters[i] = adjusted_clusters[e[0]-1]
                cnt += 1

    # Sampling
    else:
        adjusted_motion = []
        adjusted_input_words = []
        adjusted_nn_words = []
        adjusted_clusters = []
        decrease_frame = len(motion) - adjust_frame
        if decrease_frame == 1:
            decrease_interval = len(motion) / 2 + 1
        else:
            decrease_interval = len(motion) / decrease_frame
        for i in range(len(motion)):
            if i != 0 and i % decrease_interval == 0:
                continue
            adjusted_motion.append(motion[i])
            adjusted_input_words.append(input_words[i])
            adjusted_nn_words.append(nn_words[i])
            adjusted_clusters.append(clusters[i])
            if len(adjusted_motion) == adjust_frame:
                break
    
    return np.array(adjusted_motion), adjusted_input_words, adjusted_nn_words, adjusted_clusters


