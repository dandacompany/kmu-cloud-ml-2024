import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import pickle as pkl
from glob import glob

def main():
    parser = argparse.ArgumentParser()
    # SageMaker 특정 인자 설정 (기본값은 환경 변수에서 가져옴)
    parser.add_argument('--output_data_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    # 하이퍼파라미터 설정
    parser.add_argument('--max-depth', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--reg-alpha', type=float, default=0)
    parser.add_argument('--reg-lambda', type=float, default=1)
    parser.add_argument('--subsample', type=float, default=1)
    parser.add_argument('--colsample-bytree', type=float, default=1)
    parser.add_argument('--num-round', type=int, default=200)
    parser.add_argument('--early-stopping-rounds', type=int, default=10)
    parser.add_argument('--objective', type=str, default='binary:logistic')
    parser.add_argument('--eval-metric', type=str, default='auc')
    args, _ = parser.parse_known_args()

    # 데이터 로드
    
    # CSV 파일 목록 가져오기
    train_files = glob(args.train + "/*.csv")
    train_data = pd.concat([pd.read_csv(file) for file in train_files], ignore_index=True)
    val_files = glob(args.validation + "/*.csv")
    val_data = pd.concat([pd.read_csv(file) for file in val_files], ignore_index=True)

    # 특성과 타겟 분리
    X_train = train_data.iloc[:, 1:]
    y_train = train_data.iloc[:, 0]
    X_val = val_data.iloc[:, 1:]
    y_val = val_data.iloc[:, 0]
    
    d_train = xgb.DMatrix(X_train, label=y_train)
    d_val = xgb.DMatrix(X_val, label=y_val)

    # XGBoost 모델 생성 및 훈련
    watchlist = [(d_train, '훈련'), (d_val, '검증')]
    
    params = {
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'reg_alpha': args.reg_alpha,
        'reg_lambda': args.reg_lambda,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'objective': args.objective,
        'eval_metric': args.eval_metric,
    }
    xgb_model = xgb.train(params, d_train, args.num_round, watchlist, early_stopping_rounds=args.early_stopping_rounds, verbose_eval=10)
       
    # 검증 데이터로 성능 평가
    y_pred = xgb_model.predict(d_val)
    y_pred_binary = (y_pred > 0.5).astype(int)

    accuracy = accuracy_score(y_val, y_pred_binary)
    precision = precision_score(y_val, y_pred_binary)
    recall = recall_score(y_val, y_pred_binary)
    f1 = f1_score(y_val, y_pred_binary)

    print(f'검증 정확도: {accuracy:.4f}')
    print(f'검증 정밀도: {precision:.4f}')
    print(f'검증 재현율: {recall:.4f}')
    print(f'검증 F1 점수: {f1:.4f}')

    # 모델 저장
    model_path = os.path.join(args.model_dir, 'xgboost-model')
    pkl.dump(xgb_model, open(model_path, 'wb'))

if __name__ == '__main__':
    main()
