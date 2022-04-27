# 파이썬 패키지 가져오기
import matplotlib.pyplot as plt
import numpy as np
from time import time
import os
import glob

from keras.datasets import mnist
from keras.layers import Dense, Flatten, Reshape
from keras.layers import LeakyReLU
from keras.models import Sequential

# 하이퍼 파라미터
MT_GEN = 128
MY_DIS = 128
MY_NOISE = 100

MY_SHAPE = (28, 28, 1)
MY_EPOCH = 5000
MY_BATCH = 300

# 출력이미지 폴더 생성
MY_FOLDER = 'output/'
os.makedirs(MY_FOLDER,
            exist_ok=True)

for f in glob.glob(MY_FOLDER + '*' ): os.remove(f)


# 데이터 준비 --------------

# 결과는  numpy의 n-차원 행렬 형식
def read_data():
    # 학습용 입력값만 사용(GAN은 비지도 학습)
    (X_train, _), (_, _) = mnist.load_data()

    print('데이터 모양:', X_train.shape)
    # plt.imshow(X_train[0], cmap='gray')
    # plt.show()

    # [-1, 1] 데이터 스케일링
    X_train = X_train / 127.5 - 1.0

    # 채널정보 추가
    X_train = np.expand_dims(X_train, axis=3)
    print('데이터 모양:', X_train.shape)

    return X_train

# 인공 신경망 구현 --------------

# 생성자 설계
def build_generator():
    model = Sequential()

    # 입력층 + 은닉층 1
    model.add(Dense(MT_GEN,
                    input_dim=MY_NOISE))
    model.add(LeakyReLU(alpha=0.01))

    print('\n생성자 요약')
    model.summary()

build_generator()


