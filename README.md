# mediapipe sign language

mediapipe를 활용한 한국어 수어 인식 프로젝트

## process

1.데이터셋 생성(train.py)

2.모델 생성 및 학습(train.ipynb)

3.테스트(test.py)


### 1.데이터셋 생성

test.py 파일로 Ai hub에서 수어인식 데이터를 가져와서 데이터의 영상데이터 하나마다 좌표값63개, 각도값 15개, visibility 21개
정답 라벨1개로 구성된 100개의 데이터를 가진 배열을 쌓아서 np파일로 dataset 파일에 저장된다.

### 2.모델 생성 및 학습

train.ipynb 파일로 저장된 데이터셋을 확인하고 확인된 데이터셋을 LSTM 모델로 학습시켜서 평가까지 한다. 그리고 학습된 모델은 model 파일에 저장되어있다.

### 3.테스트

test.py 파일로 동영상들 6개를 테스트 해보았다. 해당하는 수어의 영어적 표현이 손에 뜬다. 그리고 무슨 동작인지 눈에 보일수 있도록 print문을 통해서 보이게 된다.

![미디어파이프](https://user-images.githubusercontent.com/35069745/167256738-5cce6580-d68f-4c7b-8895-19d4fba87863.gif)

### 그 외

test_webcam.py 파일은 현재 웹캠과 연결하여 실제 사람의 손 수어를 테스트해보고자 했지만 아직 개발 진행중에있다.

## Contributing

https://www.youtube.com/watch?v=udeQhZHx-00

https://github.com/kairess/Rock-Paper-Scissors-Machine
