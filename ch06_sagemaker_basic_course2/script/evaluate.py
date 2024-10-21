import json
import pathlib
import pickle as pkl
import tarfile
import os
import boto3

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def preprocess_test_data(test_data, assets):
    """입력 데이터를 전처리합니다."""
    scaler, encoders, pca = assets

    y = test_data['income']
    X = test_data.drop(columns=['income'])

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
        
    
    # 범주형 컬럼 레이블 인코딩
    for feature in encoders.keys() :
        le = encoders[feature]
        X[feature] = X[feature].astype(str)
        # 인코더 업데이트
        unique_values = np.unique(X[feature])
        le.classes_ = np.unique(np.concatenate([le.classes_, unique_values]))
        # 변환 처리
        X[feature] = le.transform(X[feature])

    # 스케일링
    X[numeric_features] = scaler.transform(X[numeric_features])

    # PCA 차원축소
    X_pca = pca.transform(X)
    X_pca = pd.DataFrame(X_pca, columns=[f'PC{i}' for i in range(1, pca.n_components_ + 1)])
    
    return X_pca, y

if __name__ == "__main__":
    # 모델 파일 로드
    model_path = '/opt/ml/processing/model/model.tar.gz'
    with tarfile.open(model_path) as tar:
        tar.extractall(path='.')
    
    xgb_model = pkl.load(open('xgboost-model', 'rb'))
    
    # S3에서 asset 파일을 로컬로 복사
    s3 = boto3.client('s3')
    bucket_name = 'dante-sagemaker' # 본인의 버킷명으로 반드시 수정하세요!
    project_name = 'income-prediction'
    
    # 자산 파일 로드
    scaler_key = f'{project_name}/asset/scaler.pkl'
    encoder_key = f'{project_name}/asset/encoder.pkl'
    pca_key = f'{project_name}/asset/pca.pkl'
    
    scaler_obj = s3.get_object(Bucket=bucket_name, Key=scaler_key)
    encoder_obj = s3.get_object(Bucket=bucket_name, Key=encoder_key)
    pca_obj = s3.get_object(Bucket=bucket_name, Key=pca_key)
    
    scaler = pkl.loads(scaler_obj['Body'].read())
    encoders = pkl.loads(encoder_obj['Body'].read())
    pca = pkl.loads(pca_obj['Body'].read())
    
    # 추론 데이터 로드
    test_data = pd.read_csv('/opt/ml/processing/test/test.csv')
    
    # 추론 데이터 전처리
    X, y_true = preprocess_test_data(test_data, (scaler, encoders, pca))
    
    # 추론 데이터 예측
    dmatrix = xgb.DMatrix(X)
    y_pred = xgb_model.predict(dmatrix)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # 평가 데이터 저장
    report_dict = {
        'classification_metrics': {
            'roc_auc': roc_auc_score(y_true, y_pred_binary),
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary),
            'recall': recall_score(y_true, y_pred_binary),
            'f1': f1_score(y_true, y_pred_binary)
        }
    }
    
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
