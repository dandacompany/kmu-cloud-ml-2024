import argparse
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from glob import glob

# 입력 인수 파싱
parser = argparse.ArgumentParser()
parser.add_argument('--n_components', type=int, default=4)
parser.add_argument('--train-test-split-ratio', type=float, default=0.3)
args, _ = parser.parse_known_args()

# SageMaker 데이터 경로
input_data_path = '/opt/ml/processing/input'
train_data_path = '/opt/ml/processing/train'
validation_data_path = '/opt/ml/processing/validation'
test_data_path = '/opt/ml/processing/test'
asset_path = '/opt/ml/processing/asset'

# 데이터 로드
print("데이터 로드 중")

input_files = glob(os.path.join(input_data_path, '*.csv'))
df = pd.concat([pd.read_csv(file, low_memory=False) for file in input_files])

# 특성과 타겟 분리
print("특성과 타겟 준비 중")
X = df.drop('income', axis=1)
y = df['income']

# 테스트 데이터 분리
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 타겟 변수가 이미 숫자형인지 확인
if y.dtype == 'object':
    # 문자열인 경우에만 매핑 적용
    y = y.map({
        '<=50K': 0,
        '<=50K.': 0,
        '>50K': 1,
        '>50K.': 1
    })
else:
    # 이미 숫자형인 경우 그대로 사용
    print("타겟 변수가 이미 숫자형입니다.")

# 결측치 처리
X_tmp = X_tmp.replace('?', np.nan)

# 범주형 변수와 수치형 변수 구분
categorical_features = X_tmp.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_features = X_tmp.select_dtypes(include=['int64', 'float64']).columns.tolist()

# 범주형 변수에 'Unknown' 카테고리 추가 및 결측치 처리
for feature in categorical_features:
    X_tmp[feature] = X_tmp[feature].astype('category')
    X_tmp[feature] = X_tmp[feature].cat.add_categories('Unknown')
    X_tmp[feature] = X_tmp[feature].fillna('Unknown')

# 수치형 특성의 결측치는 중앙값으로 대체
for feature in numeric_features:
    X_tmp[feature] = X_tmp[feature].fillna(X_tmp[feature].median())

# 훈련 / 검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.2, random_state=2024)

# 범주형 컬럼 레이블 인코딩
label_encoders = {}
for feature in categorical_features:
    le = LabelEncoder()
    X_train[feature] = le.fit_transform(X_train[feature])
    X_val[feature] = le.transform(X_val[feature])
    label_encoders[feature] = le

# 표준화
print("특성 표준화 중")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# PCA 적용
pca = PCA(n_components=args.n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)

# PCA 결과를 DataFrame으로 변환
train_data = pd.concat([pd.Series(y_train, name='income'), pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(X_train_pca.shape[1])], index=X_train.index)], axis=1)
val_data = pd.concat([pd.Series(y_val, name='income'), pd.DataFrame(X_val_pca, columns=[f'PC{i+1}' for i in range(X_val_pca.shape[1])], index=X_val.index)], axis=1)
test_data = pd.concat([y_test, X_test], axis=1)

# 전처리 데이터 저장
print("결과 저장 중")
train_file_path = os.path.join(train_data_path, "train.csv")
train_data.to_csv(train_file_path, index=False)
val_file_path = os.path.join(validation_data_path, "validation.csv")
val_data.to_csv(val_file_path, index=False)
test_file_path = os.path.join(test_data_path, "test.csv")
test_data.to_csv(test_file_path, index=False)

# 에셋 저장
with open(f'{asset_path}/adult_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
with open(f'{asset_path}/adult_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open(f'{asset_path}/adult_pca.pkl', 'wb') as f:
    pickle.dump(pca, f)
    
print("PCA 전처리 완료")
