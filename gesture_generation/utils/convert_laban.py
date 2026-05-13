import json

def loadKeyframeData(json_file):
    with open(json_file, 'r') as f:
        laban_score = json.load(f)

    gesture_name = list(laban_score.keys())[0]
    laban_score = laban_score[gesture_name]
    keyframe_list = list(laban_score.keys())
    
    return laban_score, keyframe_list

def convert_laban_format(laban_file, save_path=None):

    laban_data, kf_list = loadKeyframeData(laban_file)

    aft_joints = ["head", "relbow", "rwrist", "lelbow", "lwrist"]
    bef_joints = ["head", "right elbow", "right wrist", "left elbow", "left wrist"]

    converted_data = []
    for i in range(len(kf_list)):
        if i == len(kf_list) - 1:
            break

        kf_data = laban_data[kf_list[i]]
        next_kf_data = laban_data[kf_list[i+1]]

        start_time = float(kf_data["start time"][0])/1000
        next_start_time = float(next_kf_data["start time"][0])/1000
        duration = next_start_time - start_time

        kf_format = {"time": start_time}

        for j in range(len(bef_joints)):
            dic = {
                "dur": duration,
                "dir": str.lower(kf_data[bef_joints[j]][0]),
                "lvl": str.lower(kf_data[bef_joints[j]][1])
            }
            kf_format[aft_joints[j]] = dic

        converted_data.append(kf_format)


    # return both hands to rest position (for MSRABot)
    duration = 0.3
    time = converted_data[-1]['time'] + duration
    last_position =   {
            "time": time,
            "head": {
                "dur": duration,
                "dir": "forward",
                "lvl": "normal"
                },
            "relbow": {
                "dur": duration,
                "dir": "place",
                "lvl": "low"
                },
            "rwrist": {
                "dur": duration,
                "dir": "left forward",
                "lvl": "normal"
                },
            "lelbow": {
                "dur": duration,
                "dir": "place",
                "lvl": "low"
                },
            "lwrist": {
                "dur": duration,
                "dir": "right forward",
                "lvl": "normal"
                }
        }
    converted_data.append(last_position)

    if save_path:
        with open(save_path, mode='wt', encoding='utf-8') as file:
            json.dump(converted_data, file, ensure_ascii=False, indent=2)

    return converted_data

if __name__ == "__main__":
    laban_file = "C:/Users/b19.teshima/Documents/Gesture/Labanotation/LabanotationSuite/GestureAuthoringTools/LabanEditor/data_output/they_keep__4word_interpo_10frame.json"
    save_path = "../../Labanotation/kinectSuite/they_keep__4word_interpo_10frame_converted.json"

    convert_laban_format(laban_file, save_path)