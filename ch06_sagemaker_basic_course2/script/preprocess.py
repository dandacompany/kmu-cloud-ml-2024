import argparse
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

def preprocess_data(input_data_path, output_train_path, output_val_path, asset_path, n_components, test_size):
    # 데이터 읽기
    original_data = pd.read_csv(input_data_path)

    # 특성과 타겟 분리
    X = original_data.iloc[:, 1:]
    y = original_data.iloc[:, 0]
    
    # 결측치 처리
    X = X.replace('?', np.nan)
    
    # 타겟 변수 인코딩
    if y.dtype == 'object':
        y = y.map({
            '<=50K': 0,
            '<=50K.': 0,
            '>50K': 1,
            '>50K.': 1
        })
    else:
        print("타겟 변수가 이미 숫자형입니다.")

    # 범주형 변수와 수치형 변수 구분
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # 범주형 변수에 'Unknown' 카테고리 추가 및 결측치 처리
    for feature in categorical_features: 
        X[feature] = X[feature].astype('category')
        X[feature] = X[feature].cat.add_categories('Unknown')
        X[feature] = X[feature].fillna('Unknown')

    # 수치형 특성의 결측치는 중앙값으로 대체
    for feature in numeric_features:
        X[feature] = X[feature].fillna(X[feature].median())
    
    # 훈련 - 검증 데이터셋 분할
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 2024)

    # 데이터 스케일링
    scaler = StandardScaler()
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    
    # 레이블 인코딩
    encoders = {}
    for col in categorical_features:
        encoder = LabelEncoder()
        X_train[col] = encoder.fit_transform(X_train[col])
        encoders[col] = encoder
        X_val[col] = encoder.transform(X_val[col])
    

    # PCA 수행
    pca = PCA(n_components=n_components)
    X_train_pca = pd.DataFrame(pca.fit_transform(X_train), index=X_train.index, columns=[f'PC{i}' for i in range(1, pca.n_components_+1)])
    X_val_pca = pd.DataFrame(pca.transform(X_val), index=X_val.index, columns=[f'PC{i}' for i in range(1, pca.n_components_+1)])

    
    print(f"훈련데이터 차원축소 : {X_train.shape} -> {X_train_pca.shape}")
    print(f"검증데이터 차원축소 : {X_val.shape} -> {X_val_pca.shape}")
    
    # 레이블 데이터 추가
    train_data = pd.concat([y_train, X_train_pca], axis=1)
    val_data = pd.concat([y_val, X_val_pca], axis=1)

    # 전처리된 데이터 저장
    train_data.to_csv(output_train_path, index=False)
    val_data.to_csv(output_val_path, index=False)

    # 인코더와 스케일러 저장
    with open(os.path.join(asset_path, 'encoder.pkl'), 'wb') as f:
        pickle.dump(encoders, f)
    with open(os.path.join(asset_path, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(asset_path, 'pca.pkl'), 'wb') as f:
        pickle.dump(pca, f)
    
    print("전처리 완료 및 데이터 저장 완료")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-components', type=float, default=0.9)
    parser.add_argument('--test-size', type=float, default=0.2)
    
    args, _ = parser.parse_known_args()
    
    input_data_path = '/opt/ml/processing/input/original_data.csv'
    output_train_path = '/opt/ml/processing/output/train/train_data.csv'
    output_val_path = '/opt/ml/processing/output/validation/val_data.csv'
    asset_path = '/opt/ml/processing/output/asset'
    
    os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_val_path), exist_ok=True)
    os.makedirs(asset_path, exist_ok=True)
    
    preprocess_data(input_data_path, output_train_path, output_val_path, asset_path, args.n_components, args.test_size)
