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
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
video_dir="004.수어_영상_sample/원시데이터/*.mp4"
language_dir="004.수어_영상_sample/라벨링데이터/morpheme/"
index=len(video_dir)
count=0
language_list=os.listdir(language_dir)
language_list_py=[file for file in language_list if file.endswith('.json')]
dict_list = []

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles




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
print(str(actions))
new_list=[]
for v in actions:
    if v not in new_list:
        new_list.append(v)
print(new_list)
dictionary= {string : i for i,string in enumerate(new_list)}
start_list=[round(x,3) for x in start_time]
end_list=[round(x,3) for x in end_time]
#print(actions)
print(dictionary)
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
        print(dictionary[actions[idx]])
        ret, img = cap.read()
        if ret==False:
            break


        #img = cv2.flip(img, 1) #웹캠이라서 한거임 우리는 동영상이니까 괜츈

        #cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), #동영상이라서 상관 없음
        #            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)



        while round(time.time() - start_time, 3) < end_list[idx]:
            ret, img = cap.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                joint = np.zeros((21, 4))  # joint 초기화
                for res in result.multi_hand_landmarks:

                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    # Compute angles between joints
                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
                         :3]  # Parent joint
                    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                         :3]  # Child joint
                    v = v2 - v1  # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],
                                                :]))  # [15,]

                    angle = np.degrees(angle)  # Convert radian to degree

                    angle_label = np.array([angle], dtype=np.float32)
                    #print(dictionary[actions[idx]])
                    angle_label = np.append(angle_label, dictionary[actions[idx]])

                    d = np.concatenate([joint.flatten(), angle_label])

                    data.append(d)
                    mp_drawing.draw_landmarks(img, res,
                                     mp_hands.HAND_CONNECTIONS,
                                     mp_drawing_styles.get_default_hand_landmarks_style(),
                                     mp_drawing_styles.get_default_hand_connections_style())

            cv2.imshow('img', img)

            if cv2.waitKey(1) == ord('q'):
                break
            #count+=1

            #if index==count:
                #break

        data = np.array(data)
        print(data.shape)
        #print(actions[idx], data.shape)
        #np.save(os.path.join('dataset', f'raw_{actions[idx]}_{created_time}_{idx}'), data)

        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])


        full_seq_data = np.array(full_seq_data)
        print(full_seq_data.shape)
        print(full_seq_data[:, 0, -1])

        print(actions[idx], full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{actions[idx]}_{created_time}_{idx}'), full_seq_data)
        idx += 1
        cap.release()
        cv2.destroyAllWindows()
        break




