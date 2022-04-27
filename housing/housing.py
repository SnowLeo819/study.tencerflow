# 파이썬 패키지 불러오기
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 하이퍼 파라미터
MY_EPOCH = 500
MY_BATCH = 64

## 데이터 준비
# 데이터 파일 읽기
# 결과는 pandas 의 데이터 프레임 형식
heading = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM'
           'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO'
           'LSTAT', 'MEDV']

raw = pd.read_csv('housing.csv')

# 데이터 원본 출력
print('원본데이터 샘플 10개')
print(raw.head(10))

print('원본데이터 통계')
print(raw.describe())

# Z-점수 정규화
# 결과는 numpy 의 n-차원 행렬 형식
scaler = StandardScaler()
z_data = scaler.fit_transform(raw)

# numpy에서 pandas로 전환
# header 정보 복구 필요
z_data = pd.DataFrame(z_data,
                      columns=heading)

# 정규화 된 데이터 출력
print('정규화 된 데이터 샘플 10개')
print(z_data.head(10))
print('정규화 된 데이터 통계')
print(z_data.describe())