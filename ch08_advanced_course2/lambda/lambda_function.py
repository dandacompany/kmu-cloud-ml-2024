import json
import boto3
import pickle
import pandas as pd

def find_endpoint(project_name):
    sagemaker_client = boto3.client('sagemaker')
    
    # 엔드포인트 목록 가져오기
    endpoints = sagemaker_client.list_endpoints()
    
    # 프로젝트 이름을 포함하는 엔드포인트 찾기
    matching_endpoints = [
        endpoint for endpoint in endpoints['Endpoints'] 
        if project_name in endpoint['EndpointName']
    ]
    
    if matching_endpoints:
        # 가장 최근에 생성된 엔드포인트 반환
        return sorted(matching_endpoints, key=lambda x: x['CreationTime'], reverse=True)[0]['EndpointName']
    else:
        return None
    
def lambda_handler(event, context):
    
    # S3와 SageMaker 런타임 클라이언트 생성
    
    # 로컬에서 실행할때
    # boto3_session = boto3.Session(profile_name='awstutor')
    # s3 = boto3_session.client('s3')
    # sagemaker_runtime = boto3_session.client('sagemaker-runtime')
    
    # SageMaker IDE 인스턴스에서 실행할때
    s3 = boto3.client('s3')
    sagemaker_runtime = boto3.client('sagemaker-runtime')
    
    # 버킷 이름과 프로젝트 이름 설정
    bucket_name = 'dante-sagemaker'
    project_name = 'mushroom-classification-api-integration'
    
    # 엔드포인트 이름 찾기
    endpoint_name = find_endpoint(project_name)

    # 인코더 로드
    encoder_key = f'{project_name}/output/asset/feature_encoders_dict.pkl'
    encoder_obj = s3.get_object(Bucket=bucket_name, Key=encoder_key)
    feature_encoders_dict = pickle.loads(encoder_obj['Body'].read())
    
    # 입력 데이터 가져오기
    input_data = event['data']
    
    # 특성 컬럼 정의
    feature_columns = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
       'ring-type', 'spore-print-color', 'population', 'habitat']
    
    # 데이터프레임 생성 및 전처리
    X_test = pd.DataFrame(input_data, columns=feature_columns)
    X_test = X_test.fillna('nan')
    X_test_encoded = X_test.copy()
    
    # 특성 인코딩
    for col in feature_columns:
        X_test_encoded[col] = X_test[col].map(feature_encoders_dict[col])
        
    # 페이로드 생성
    payload =  "\n".join([",".join([str(x) for x in row]) for row in X_test_encoded.values])
    
    # SageMaker 엔드포인트 호출
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='text/csv',
        Body=payload
    )
    
    # 예측 결과 처리
    predictions = response['Body'].read().decode()
    predictions = list(map(lambda x: int(float(x) > 0.5), predictions.strip().split('\n')))
    
    # 결과 반환
    return {
        'statusCode': 200,
        'body': json.dumps(predictions)
    }