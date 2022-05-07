import cv2
import mediapipe as mp
import numpy as np
import time, os
import glob
import sys
import json



actions = []
seq_length = 20 #lstm 넣을 데이터 크기
#secs_for_action =0  #액션 녹화 시간
end_time=[]
start_time=[]
duration=[]
# MediaPipe hands model
hands = mp.solutions.holistic
mp_hands=hands.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

video_dir="C:/Users/user/Desktop/동규/캡스톤 프로젝트 정리/004.수어_영상_sample/원시데이터/*.mp4"
language_dir="C:/Users/user/Desktop/동규/캡스톤 프로젝트 정리/004.수어_영상_sample/라벨링데이터/morpheme/"
index=len(video_dir)
count=0
language_list=os.listdir(language_dir)
language_list_py=[file for file in language_list if file.endswith('.json')]
dict_list = []

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def landmark_to_array(mp_landmark_list):
    """Return a np array of size (nb_keypoints x 3)"""
    keypoints = []
    for landmark in mp_landmark_list.landmark:
        keypoints.append([landmark.x, landmark.y, landmark.z])
    return np.nan_to_num(keypoints)



for i in language_list_py:
    with open((language_dir+i),"r",encoding="UTF8") as f:
        contents = f.read()
        json_data=json.loads(contents)
        actions.append(json_data["data"][0]["attributes"][0]["name"])
        duration.append(json_data["metaData"]["duration"])
        start_time.append(json_data["data"][0]["start"])
        end_time.append(json_data["data"][0]["end"])
#print("시작 시간"+str(start_time))
#print("끝나는 시간"+str(end_time))
dictionary= {string : i for i,string in enumerate(actions)}
start_list=[round(x,3) for x in start_time]
end_list=[round(x,3) for x in end_time]


i=0
for key,value in dictionary.items():
    dictionary[key]=i
    i+=1


if not video_dir:
    print("동영상이 없네요..")
    sys.exit()

idx=0
joint = np.zeros((21, 4))
os.makedirs('dataset', exist_ok=True)

for video in glob.glob(video_dir):
    print("video:"+video)
    cap = cv2.VideoCapture(video)
    created_time = int(time.time())
    start_time = time.time()
    while cap.isOpened():

        print(idx)
        data = []

        ret, img = cap.read()
        if ret==False:
            break


        #img = cv2.flip(img, 1) #웹캠이라서 한거임 우리는 동영상이니까 괜츈

        #cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), #동영상이라서 상관 없음
        #            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)



        while round(time.time() - start_time, 3) < end_time[idx]:
            ret, img = cap.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = mp_hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            left_hand = np.zeros(63).tolist()
            right_hand = np.zeros(63).tolist()

            if result.right_hand_landmarks and result.left_hand_landmarks:
                right_hand =landmark_to_array(result.right_hand_landmarks).reshape(63)
                left_hand = landmark_to_array(result.left_hand_landmarks).reshape(63)

                d=np.concatenate([right_hand,left_hand])




                d = np.append(d, dictionary[actions[idx]])


                data.append(d)

                cv2.imshow('img', img)
                if cv2.waitKey(1) == ord('q'):
                    break
                """
                img = mp_drawing.draw_landmarks(
                    img,
                    landmark_list=result.left_hand_landmarks,
                    connections=hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(232, 254, 255), thickness=1, circle_radius=4
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 249, 161), thickness=2, circle_radius=2
                    ),
                )
                print(img)
                """

            #count+=1

            #if index==count:
                #break
        idx += 1
        data = np.array(data)
        print(actions[idx], data.shape)
        np.save(os.path.join('dataset', f'raw_{actions[idx]}_{created_time}_{idx}'), data)

        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(actions[idx], full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{actions[idx]}_{created_time}_{idx}'), full_seq_data)
        
        cap.release()
        cv2.destroyAllWindows()
        break









