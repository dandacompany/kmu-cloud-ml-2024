import argparse
import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def model_fn(model_dir):
    """모델 로드 함수"""
    print("모델 로딩 중...")
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    print("모델 로딩 완료")
    return {'model': model, 'scaler': scaler}

def predict_fn(input_data, model):
    """예측 함수"""
    scaler = model['scaler']
    clf = model['model']
    scaled_input = scaler.transform(input_data)
    predictions = clf.predict(scaled_input)
    return predictions

if __name__ == '__main__':
    print("프로그램 시작")
    
    parser = argparse.ArgumentParser()
    
    # 하이퍼파라미터 등 파라미터 추가
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--min-samples-split', type=int, default=2)
    
    # SageMaker 특정 인자
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    args, _ = parser.parse_known_args()
    print(f"파라미터 설정: n_estimators={args.n_estimators}, min_samples_split={args.min_samples_split}")
    
    # Iris 데이터셋 로드
    print("Iris 데이터셋 로드 중...")
    iris = load_iris()
    X, y = iris.data, iris.target
    print("데이터셋 로드 완료")
    
    # 데이터 전처리
    print("데이터 전처리 시작")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("데이터 전처리 완료")
    
    # 모델 훈련
    print("모델 훈련 시작")
    model = RandomForestClassifier(n_estimators=args.n_estimators, min_samples_split=args.min_samples_split, random_state=2024)
    model.fit(X_train_scaled, y_train)
    print("모델 훈련 완료")
    
    # 모델 평가
    print("모델 평가 중...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'테스트 정확도: {accuracy}')
    
    if args.model_dir:
        print("모델 및 스케일러 저장 중...")
        # 모델 저장
        joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))
    
        # 스케일러 저장 (추론 시 사용)
        joblib.dump(scaler, os.path.join(args.model_dir, 'scaler.joblib'))
        print("모델 및 스케일러 저장 완료")
    
    print("프로그램 종료")
