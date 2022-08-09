# 7장의 카드(나의 카드 2장, 커뮤니티 카드 5장)를 보고 텍사스 홀덤 포커 게임 승리 확률을 예측하는 AI 모델



import tensorflow as tf
import numpy as np
import os
import pickle


# 데이터 불러오기
scriptpath = os.path.dirname(__file__)
file = os.path.join(scriptpath, 'PokerTestDataSet.pickle')

with open(file, "rb") as fr:
    dataset = pickle.load(fr)

(x_train, t_train), (x_test, t_test) = dataset.loadTestData()

# 텐서플로우 딥러닝 모델 생성
model = tf.keras.models.Sequential([ # 딥러닝 신경망 모델 생성
    tf.keras.layers.Dense(64, activation='tanh'), # 64개의 노드를 가진 은닉층 1 (활성화 함수는 tanh)
    tf.keras.layers.Dense(128, activation='tanh'), # 128개의 노드를 가진 은닉층 2
    tf.keras.layers.Dense(128, activation='tanh'), # 128개의 노드를 가진 은닉층 2
    tf.keras.layers.Dense(1, activation='sigmoid'), # 출력층(0~1 사이의 확률 결과를 원하므로 활성화 함수로 시그모이드 함수 사용)
])

# 모델 컴파일 (매개변수 최적화 기법(Adam), 손실 함수(이진 교차 엔트로피 함수), 모델 평가 방법(정확도) 결정)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 

# 모델 딥러닝 학습 (x 데이터는 입력값, y 데이터는 정답 레이블, 에폭)
model.fit(np.array(x_train), np.array(t_train), epochs=1000)

# 테스트 파일 임의로 2개 불러와서 예측 결과와 실제 정답 비교
predict = model.predict([x_test[10], x_test[321]])
answer = t_test[10], t_test[321]
print(predict)
print(answer)
