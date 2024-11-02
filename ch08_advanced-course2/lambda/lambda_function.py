# 람다 함수 작성
import json
import pickle
import boto3
import numpy as np

import os
SAGEMAKER_ENDPOINT_NAME = os.environ['SAGEMAKER_ENDPOINT_NAME']

with open('feature_encoders_dict.pkl', 'rb') as f:
    feature_encoders_dict = pickle.load(f)

with open('label_encoder_dict.pkl', 'rb') as f:
    label_encoder_dict = pickle.load(f)

    
feature_names = np.genfromtxt('feature_names.csv', delimiter=',', dtype=str)


feature_encoder = lambda features : [feature_encoders_dict[feature_names[idx]][feat if feat else 'nan'] for idx, feat in enumerate(features)]
    

def lambda_handler(event, context) :
    
    client = boto3.client('sagemaker-runtime', region_name='ap-northeast-2')
    
    request = json.loads(json.dumps(event, indent=4))
    features_list = request['features']
    features_list = np.array(features_list).tolist()
    features_list = list(map(feature_encoder, features_list))
    csv_input = "\n".join([",".join(map(str, feature)) for feature in features_list])

    result = client.invoke_endpoint(EndpointName=SAGEMAKER_ENDPOINT_NAME,
                        Body=csv_input.encode('utf-8'),
                        ContentType='text/csv')

    predictions = result['Body'].read().decode('utf-8').strip().split('\n')
    predictions = [float(pred) for pred in predictions]
    predictions
    return {
        'statusCode' : 200,
        'body' : json.dumps({'predictions' : predictions})
    }