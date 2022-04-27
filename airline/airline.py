# 파이썬 패키지 가져오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import InputLayer, Dense
from keras.layers import LSTM

# 하이퍼 파라미터
MY_PAST = 12
MY_SPLIT = 0.8
MY_UNIT = 300
MY_SHAPE = (MY_PAST, 1)

MY_EPOCH = 300
MY_BATCH = 64
np.set_printoptions(precision=3)

# 데이터 파일 읽기
# 결과는 pandas 의 데이터 프레임 형식
raw = pd.read_csv('airline.csv',
                  header=None,
                  usecols=[1])

# 시계열 데이터 시각화
# plt.plot(raw)
# plt.show()

# 데이터 원본 출력
print('원본 데이터 샘플 13개')
print(raw.head(13))
print('\n원본 데이터 통계')
print(raw.describe())

# MinMaxScaler 데이터 정규화
scaler = MinMaxScaler()
s_data = scaler.fit_transform(raw)

print('\nMinMax 정규화 형식:', type(s_data))

# 정규화 데이터 출력
df = pd.DataFrame(s_data)

print('\n정규화 데이터 샘플 13개')
print(df.head(13))
print('\n정규화 데이터 통계')
print(df.describe())

# 13개 묶음으로 데이터 분할
# 결과는 python 리스트
bundle = []
for i in range(len(s_data) - MY_PAST):
    bundle.append(s_data[i: i+MY_PAST+1])

# 데이터 분할 결과 확인
print('\n 총 13개 묶음의 수:', len(bundle))
print(bundle[0])
print(bundle[1])

# numpy 로 전환
print('분할 데이터의 타입:', type(bundle))
bundle = np.array(bundle)
print('분할 데이터의 모양:', bundle.shape)

# 데이터를 입력과 출력으로 분할
X_data = bundle[:, 0:MY_PAST]
Y_data = bundle[:, -1]

# 데이터 학습용과 평가용으로 분할
split = int(len(bundle) * MY_SPLIT)
X_train = X_data[: split]
X_test = X_data[split:]

Y_train = Y_data[: split]
Y_test = Y_data[split:]

# 최종 데이터 모양
print('\n학습용 입력 데이터 모양:', X_train.shape)
print('학습용 출력 데이터 모양:', Y_train.shape)

print('평가용 입력 데이터 모양:', X_test.shape)
print('평가용 출력 데이터 모양:', Y_test.shape)

# 인공신경망 구현

# RNN 구현
# 케라스 RNN은 2차원 입력만 허용
model = Sequential()
model.add(InputLayer(input_shape=MY_SHAPE))
model.add(LSTM(MY_UNIT))

model.add(Dense(1,
                activation='sigmoid'))

# 결과 출력
print('\nRNN 데이터 요약')
model.summary()