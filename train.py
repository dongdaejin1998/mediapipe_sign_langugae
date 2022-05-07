import keras.utils.np_utils
import numpy as np
import os
import json




from sklearn.metrics import multilabel_confusion_matrix
from keras.models import load_model

from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import LSTM, Dense

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

language_dir="C:/Users/user/Desktop/동규/캡스톤 프로젝트 정리/004.수어_영상_sample/라벨링데이터/morpheme/"
actions = []
language_list=os.listdir(language_dir)
language_list_py=[file for file in language_list if file.endswith('.json')]
for i in language_list_py:
    with open((language_dir+i),"r",encoding="UTF8") as f:
        contents = f.read()
        json_data=json.loads(contents)
        actions.append(json_data["data"][0]["attributes"][0]["name"])


data_dir="./dataset/seq/"
NP_list=os.listdir(data_dir)
NP_list_npy=[file for file in NP_list if file.endswith(".npy")]
L=[]
for i in range(len(actions)-1):
    A=np.load(data_dir+NP_list_npy[i])
    print(A.shape)
    data = np.concatenate([
        A
    ], axis=0)
print(data.shape)


x_data = data[:, :, :-1]
labels = data[:, 0, -1]

print(x_data.shape)
print(labels.shape)


actions=set(actions)
actions=list(actions)
y_data = to_categorical(labels, num_classes=len(actions))
print(y_data.shape)



x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=2021)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)



model = Sequential([
    LSTM(64, activation='relu', input_shape=x_train.shape[1:3]),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()



history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=200,
    callbacks=[
        ModelCheckpoint('models/model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')
    ]
)



model = load_model('models/model.h5')

y_pred = model.predict(x_val)

multilabel_confusion_matrix(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))

