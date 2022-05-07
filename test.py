import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import time,os,json
import glob
from PIL import ImageFont,ImageDraw,Image
language_dir="C:/Users/user/Desktop/동규/캡스톤 프로젝트 정리/004.수어_영상_sample/라벨링데이터/morpheme/"
actions = []
duration=[]
language_list=os.listdir(language_dir)
language_list_py=[file for file in language_list if file.endswith('.json')]
for i in language_list_py:
    with open((language_dir+i),"r",encoding="UTF8") as f:
        contents = f.read()
        json_data=json.loads(contents)
        actions.append(json_data["data"][0]["attributes"][0]["name"])
        duration.append(json_data["metaData"]["duration"])
seq_length = 20

model = load_model('models/model.h5')

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

video_dir="C:/Users/user/Desktop/동규/캡스톤 프로젝트 정리/004.수어_영상_sample/테스트용/*.mp4"


# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# out = cv2.VideoWriter('input.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))
# out2 = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))
answer_list=[]
for v in actions:
    if v not in answer_list:
        answer_list.append(v)
print(answer_list)
dictionary= {string : i for i,string in enumerate(answer_list)}

dictionary_list={'말해주다': "talk", '말하다': "speak", '운전':"drive" , '시간': "time", '여기': "here", '왼쪽': "left", '약속': "promise",
                   '위험': "danger", '당신': "you", '저기': "there", '오른쪽': "right", '급하다': "hurry", '그남자': "he", '항상': "always", '나': "I", '가다': "go"}
idx=0
seq = []
action_seq = []

def landmark_to_array(mp_landmark_list):
    """Return a np array of size (nb_keypoints x 3)"""
    keypoints = []
    for landmark in mp_landmark_list.landmark:
        keypoints.append([landmark.x, landmark.y, landmark.z])
    return np.nan_to_num(keypoints)

for video in glob.glob(video_dir):
    print("video:"+video)
    cap = cv2.VideoCapture(video)
    created_time = int(time.time())
    start_time = time.time()
    while cap.isOpened():
        ret, img = cap.read()
        print(idx)
        print(dictionary[actions[idx]])
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if ret==False:

            break


        #img = cv2.flip(img, 1) #웹캠이라서 한거임 우리는 동영상이니까 괜츈

        #cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), #동영상이라서 상관 없음
        #            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)



        while ret==True:

            ret, img = cap.read()
            if ret==False:

                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            left_hand = np.zeros(63).tolist()
            right_hand = np.zeros(63).tolist()

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    # Compute angles between joints
                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent joint
                    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                         :3]  # Child joint
                    v = v2 - v1  # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                    angle = np.degrees(angle)  # Convert radian to degree

                    d = np.concatenate([joint.flatten(), angle])

                    seq.append(d)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)


                    if len(seq) < seq_length:
                        continue

                    input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

                    y_pred = model.predict(input_data).squeeze()

                    i_pred = int(np.argmax(y_pred))
                    conf = y_pred[i_pred]

                    if conf < 0.8:
                        continue

                    action = answer_list[i_pred]
                    print(action)
                    action_seq.append(action)

                    if len(action_seq) < 3:
                        continue

                    this_action = '?'

                    if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                        this_action = action

                    if this_action in dictionary_list.keys():
                        this_action = dictionary_list[this_action]
                    cv2.putText(img, f'{this_action}',
                                org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        #count+=1

        #if index==count:
            #break
        idx += 1
        cap.release()
        cv2.destroyAllWindows()
        break