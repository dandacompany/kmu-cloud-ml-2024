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
    model_file = 'xgboost-model'
    booster = pkl.load(open(os.path.join(model_dir, model_file), 'rb'))
    
    # S3에서 asset 파일을 로컬로 복사
    s3 = boto3.client('s3')
    bucket_name = 'dante-sagemaker' # 본인의 버킷명으로 반드시 수정하세요!
    project_name = 'income-prediction'
    scaler_key = f'{project_name}/asset/scaler.pkl'
    encoder_key = f'{project_name}/asset/encoder.pkl'
    pca_key = f'{project_name}/asset/pca.pkl'
    
    scaler_obj = s3.get_object(Bucket=bucket_name, Key=scaler_key)
    encoder_obj = s3.get_object(Bucket=bucket_name, Key=encoder_key)
    pca_obj = s3.get_object(Bucket=bucket_name, Key=pca_key)
    
    scaler = pkl.loads(scaler_obj['Body'].read())
    encoders = pkl.loads(encoder_obj['Body'].read())
    pca = pkl.loads(pca_obj['Body'].read())
    
    original_feature_columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
       'marital-status', 'occupation', 'relationship', 'race', 'sex',
       'capitalgain', 'capitalloss', 'hoursperweek', 'native-country']
    
    numeric_columns = ['age', 'fnlwgt', 'education-num', 'capitalgain', 'capitalloss', 'hoursperweek']
    
    return booster, (scaler, encoders, pca, original_feature_columns, numeric_columns)

def input_fn(request_body, request_content_type):
    """입력 데이터 페이로드를 파싱합니다."""
    if request_content_type != "text/csv":
        raise ValueError(f"지원되지 않는 컨텐츠 타입입니다: {request_content_type}")
    df = pd.read_csv(io.StringIO(request_body), header=None)
    return df.values

def predict_fn(input_data, model):
    """로드된 모델로 예측을 수행합니다."""
    booster, (scaler, encoders, pca, original_feature_columns, numeric_columns) = model
    prep_input_data = preprocess_input_data(input_data, (scaler, encoders, pca, original_feature_columns, numeric_columns))
    dmatrix = xgb.DMatrix(prep_input_data)
    return booster.predict(dmatrix)

def output_fn(prediction, accept):
    """예측 출력을 포맷팅합니다."""
    if accept != "text/csv":
        raise ValueError(f"지원되지 않는 accept 타입입니다: {accept}")
    return ','.join(map(str, prediction))

def preprocess_input_data(input_data, assets):
    """입력 데이터를 전처리합니다."""
    scaler, encoders, pca, original_feature_columns, numeric_columns = assets
    X = pd.DataFrame(input_data, columns=original_feature_columns)
    X[X == '?'] = np.nan

    # 범주형 변수에 'Unknown' 카테고리 추가 및 결측치 처리
    for feature in (set(original_feature_columns) - set(numeric_columns)): 
        X[feature] = X[feature].astype('category')
        X[feature] = X[feature].cat.add_categories('Unknown')
        X[feature] = X[feature].fillna('Unknown')

    # 수치형 특성의 결측치는 중앙값으로 대체
    for feature in set(numeric_columns):
        X[feature] = pd.to_numeric(X[feature], errors='coerce')
        X[feature] = X[feature].fillna(X[feature].median())
    
    X[numeric_columns] = X[numeric_columns].astype('float64')
    X[numeric_columns] = scaler.transform(X[numeric_columns])
    
    # 범주형 컬럼 레이블 인코딩
    for feature in encoders.keys() :
        le = encoders[feature]
        X[feature] = X[feature].astype(str)
        # 인코더 업데이트
        unique_values = np.unique(X[feature])
        le.classes_ = np.unique(np.concatenate([le.classes_, unique_values]))
        # 변환 처리
        X[feature] = le.transform(X[feature])
        
    # NaN 및 무한대 값 처리
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    
    # PCA 변환 수행
    X_pca = pd.DataFrame(pca.transform(X), columns=[f'PC{i}' for i in range(1, pca.n_components_ + 1)])
    return X_pca
