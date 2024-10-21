import os
import json
import pickle as pkl
import numpy as np
import xgboost as xgb
import pandas as pd
import io
import boto3

def model_fn(model_dir):
    """XGBoost 모델과 필요한 자산을 `model_dir`에서 로드합니다."""
    # 모델 객체 로드
    model_file = 'xgboost-model'
    xgb_model = pkl.load(open(os.path.join(model_dir, model_file), 'rb'))
    
    # S3에서 asset 파일을 로컬로 복사
    s3 = boto3.client('s3')
    bucket_name = 'dante-sagemaker'
    project_name = 'adult-income-classification-v2'
    
    # 자산 파일 로드
    scaler_key = f'{project_name}/asset/adult_scaler.pkl'
    encoder_key = f'{project_name}/asset/adult_encoders.pkl'
    pca_key = f'{project_name}/asset/adult_pca.pkl'
    
    local_scaler_path = os.path.join(model_dir, 'adult_scaler.pkl')
    local_encoder_path = os.path.join(model_dir, 'adult_encoders.pkl')
    local_pca_path = os.path.join(model_dir, 'adult_pca.pkl')
    
    scaler_obj = s3.get_object(Bucket=bucket_name, Key=scaler_key)
    encoder_obj = s3.get_object(Bucket=bucket_name, Key=encoder_key)
    pca_obj = s3.get_object(Bucket=bucket_name, Key=pca_key)
    
    scaler = pkl.loads(scaler_obj['Body'].read())
    encoders = pkl.loads(encoder_obj['Body'].read())
    pca = pkl.loads(pca_obj['Body'].read())
    
    return xgb_model, (scaler, encoders, pca)

def input_fn(request_body, request_content_type):
    """입력 데이터 페이로드를 파싱합니다."""
    if request_content_type != "text/csv":
        raise ValueError(f"지원되지 않는 컨텐츠 타입입니다: {request_content_type}")
    df = pd.read_csv(io.StringIO(request_body), header=None)
    return df.values

def output_fn(prediction, accept):
    """예측 출력을 포맷팅합니다."""
    if accept != "text/csv":
        raise ValueError(f"지원되지 않는 accept 타입입니다: {accept}")
    return ','.join(map(str, prediction))

def predict_fn(input_data, model):
    """로드된 모델로 예측을 수행합니다."""
    xgb_model, (scaler, encoders, pca) = model
    prep_input_data = preprocess_input_data(input_data, (scaler, encoders, pca))
    dmatrix = xgb.DMatrix(prep_input_data)
    return xgb_model.predict(dmatrix)


def preprocess_input_data(input_data, assets):
    """입력 데이터를 전처리합니다."""
    scaler, encoders, pca = assets
    
    total_cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
       'marital-status', 'occupation', 'relationship', 'race', 'sex',
       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
       'hours-per-week']

    
    X = pd.DataFrame(input_data, columns=total_cols)

    # 전처리
    X[X == '?'] = np.nan
    X['workclass'].fillna(X['workclass'].mode()[0], inplace=True)
    X['occupation'].fillna(X['occupation'].mode()[0], inplace=True)
    X['native-country'].fillna(X['native-country'].mode()[0], inplace=True)
    X[numeric_cols] = X[numeric_cols].astype('float64')
    
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
    X_scaled = scaler.transform(X)

    # PCA 차원축소
    X_pca = pca.transform(X_scaled)
    
    return pd.DataFrame(X_pca, columns=[f'PC{i}' for i in range(1, pca.n_components_ + 1)])
   
