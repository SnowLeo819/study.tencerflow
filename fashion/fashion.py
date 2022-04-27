# 파이썬 패키지 가져오기
import numpy as np
import matplotlib.pyplot as plt
from time import time

from keras.datasets import fashion_mnist
from keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score, confusion_matrix

from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense, InputLayer
from keras.layers import Conv2D, MaxPool2D

# 하이퍼 파라미터
MY_EPOCH = 3    # 반복횟수
MY_BATCH = 300  # 1회 불러오기 수

## 데이터 준비
# 데이터 파일 읽기
# 결과는 numpy의 n-차원 행렬 형식
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

# 4 분할 데이터 모양 출력
print('\n학습용 입력 데이터 모양:', X_train.shape)
print('학습용 출력 데이터 모양:', Y_train.shape)
print('평가용 입력 데이터 모양:', X_test.shape)
print('평가용 출력 데이터 모양:', Y_test.shape)

# 샘플데이터 출력
print(X_train[0])
# plt.imshow(X_train[0], cmap='gray')
# plt.show()
print('샘플 데이터 라벨:', Y_train[0])

# 입력데이터 스케일링 : [0,1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# 채널정보 추가
# 케라스 cnn 에서 4차원 정보 필요
train = X_train.shape[0]
X_train = X_train.reshape(train, 28, 28, 1)
test = X_test.shape[0]
X_test = X_test.reshape(test, 28, 28, 1)

# 출력 데이터(=라벨 정보) 원핫 인코딩
print('원핫 인코딩 전:', Y_train[0])
Y_train = to_categorical(Y_train, 10)

print('원핫 인코딩 후:', Y_train[0])
Y_test = to_categorical(Y_test, 10)

print('학습용 출력 데이터 모양:', Y_train.shape)
print('평가용 출력 데이터 모양:', Y_test.shape)


## 인공신경망 구현
# CNN  구현(순차적 방법)
model = Sequential()

# 입력층
model.add(InputLayer(input_shape=(28, 28, 1)))

# 첫번째 합성곱 블럭
model.add(Conv2D(filters=32,
                 kernel_size=2,
                 padding='same',
                 activation='relu'))

model.add(MaxPool2D(pool_size=2))

model.summary()
