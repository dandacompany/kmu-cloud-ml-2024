import argparse
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(input_data_path, output_train_path, output_val_path, output_test_path, asset_path, test_size):
    # 데이터 읽기
    original_data = pd.read_csv(input_data_path)

    # 특성과 타겟 분리
    X = original_data.iloc[:, 1:]
    y = original_data.iloc[:, 0]
    
    # 전처리
    X = X.fillna('nan')
    
    # 모든 열을 문자열로 변환
    X = X.astype(str)
    
    # train, val, test 셋으로 나누기
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=2024)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.3, random_state=2024)
    
    # 범주형 데이터 인코딩
    feature_encoders = {
        name: LabelEncoder().fit(X[name]) for name in X.columns
    }
    
    feature_encoders_dict = {
        col: {
            orig: encoded 
            for orig, encoded in zip(feature_encoders[col].classes_, feature_encoders[col].transform(feature_encoders[col].classes_))
        }
        for col in feature_encoders.keys()
    }

    X_train_encoded = X_train.copy()
    X_val_encoded = X_val.copy()

    for name, encoder in feature_encoders.items():
        X_train_encoded[name] = encoder.transform(X_train[name])
        X_val_encoded[name] = encoder.transform(X_val[name])
        
    # y 값들을 인코딩
    y_train = (y_train != 'e').astype(int)
    y_val = (y_val != 'e').astype(int)

    # 타겟 레이블을 첫번째 열로 붙임
    train_data = pd.concat([y_train, X_train_encoded], axis=1)
    val_data = pd.concat([y_val, X_val_encoded], axis=1)
    test_data = pd.concat([y_test, X_test], axis=1)

    # 전처리된 데이터 저장
    train_data.to_csv(output_train_path, index=False, header=None)
    val_data.to_csv(output_val_path, index=False, header=None)
    test_data.to_csv(output_test_path, index=False, header=None)
    
    # 인코더와 스케일러 저장
    with open(os.path.join(asset_path, 'feature_encoders_dict.pkl'), 'wb') as f:
        pickle.dump(feature_encoders_dict, f)
    
    print("전처리 완료 및 데이터 저장 완료")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-size', type=float, default=0.2)
    
    args, _ = parser.parse_known_args()
    
    input_data_path = '/opt/ml/processing/input/original_data.csv'
    output_train_path = '/opt/ml/processing/output/train/train_data.csv'
    output_val_path = '/opt/ml/processing/output/validation/val_data.csv'
    output_test_path = '/opt/ml/processing/output/test/test_data.csv'
    asset_path = '/opt/ml/processing/output/asset'
    
    os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_val_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_test_path), exist_ok=True)
    os.makedirs(asset_path, exist_ok=True)
    
    preprocess_data(input_data_path, output_train_path, output_val_path, output_test_path, asset_path, args.test_size)
