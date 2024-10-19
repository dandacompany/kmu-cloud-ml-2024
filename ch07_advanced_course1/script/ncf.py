import tensorflow as tf
import argparse
import os
import numpy as np
import json

class DataLoader:
    @staticmethod
    def load_training_data(base_dir):
        """훈련 데이터를 로드하고 분할합니다."""
        # 'train.npy' 파일에서 훈련 데이터를 로드합니다.
        df_train = np.load(os.path.join(base_dir, 'train.npy'))
        # 파일 목록을 출력합니다.
        print("훈련 데이터 디렉토리 내용:")
        for file in os.listdir(base_dir):
            print(f"- {file}")
        # 데이터를 사용자, 아이템, 라벨로 분할합니다.
        print(df_train.shape)
        return np.split(np.transpose(df_train).flatten(), 3)

    @staticmethod
    def batch_generator(user_data, item_data, labels, batch_size, n_batch, shuffle, user_dim, item_dim):
        """훈련 및 테스트를 위한 배치를 생성합니다."""
        counter = 0
        training_index = np.arange(user_data.shape[0])

        if shuffle:
            # 학습 데이터를 무작위로 섞습니다.
            np.random.shuffle(training_index)

        while True:
            # 현재 배치의 인덱스를 선택합니다.
            batch_index = training_index[batch_size * counter:batch_size * (counter + 1)]
            # 사용자와 아이템 데이터를 원-핫 인코딩합니다.
            user_batch = tf.one_hot(user_data[batch_index], depth=user_dim)
            item_batch = tf.one_hot(item_data[batch_index], depth=item_dim)
            y_batch = labels[batch_index]
            counter += 1
            # 배치 데이터를 생성합니다.
            yield [user_batch, item_batch], y_batch

            if counter == n_batch:
                if shuffle:
                    # 모든 배치를 순회한 후 데이터를 다시 섞습니다.
                    np.random.shuffle(training_index)
                counter = 0

class NeuralCollaborativeFiltering:
    def __init__(self, user_dim, item_dim, dropout_rate=0.25):
        """Neural Collaborative Filtering 모델 초기화"""
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.dropout_rate = dropout_rate

    def build_model(self):
        """Neural Collaborative Filtering 모델을 구축합니다."""
        # 사용자와 아이템 입력 레이어를 정의합니다.
        user_input = tf.keras.Input(shape=(self.user_dim,))
        item_input = tf.keras.Input(shape=(self.item_dim,))

        # 사용자와 아이템의 임베딩을 생성합니다.
        user_gmf_emb, user_mlp_emb = self._create_user_embeddings(user_input)
        item_gmf_emb, item_mlp_emb = self._create_item_embeddings(item_input)

        # GMF(General Matrix Factorization)와 MLP(Multi-Layer Perceptron) 출력을 계산합니다.
        gmf_output = self._general_matrix_factorization(user_gmf_emb, item_gmf_emb)
        mlp_output = self._multi_layer_perceptron(user_mlp_emb, item_mlp_emb)

        # 최종 출력을 계산합니다.
        output = self._neural_cf(gmf_output, mlp_output)

        # 모델을 생성하고 반환합니다.
        return tf.keras.Model(inputs=[user_input, item_input], outputs=output)

    def _create_user_embeddings(self, inputs):
        """GMF와 MLP를 위한 사용자 임베딩을 생성합니다."""
        # GMF를 위한 사용자 임베딩
        gmf_emb = tf.keras.layers.Dense(32, activation='relu')(inputs)
        # MLP를 위한 사용자 임베딩
        mlp_emb = tf.keras.layers.Dense(32, activation='relu')(inputs)
        return gmf_emb, mlp_emb

    def _create_item_embeddings(self, inputs):
        """GMF와 MLP를 위한 아이템 임베딩을 생성합니다."""
        # GMF를 위한 아이템 임베딩
        gmf_emb = tf.keras.layers.Dense(32, activation='relu')(inputs)
        # MLP를 위한 아이템 임베딩
        mlp_emb = tf.keras.layers.Dense(32, activation='relu')(inputs)
        return gmf_emb, mlp_emb

    def _general_matrix_factorization(self, user_emb, item_emb):
        """General Matrix Factorization 브랜치를 구현합니다."""
        # 사용자와 아이템 임베딩의 요소별 곱을 계산합니다.
        return tf.keras.layers.Multiply()([user_emb, item_emb])

    def _multi_layer_perceptron(self, user_emb, item_emb):
        """Multi-Layer Perceptron 브랜치를 구현합니다."""
        # 사용자와 아이템 임베딩을 연결합니다.
        concat_layer = tf.keras.layers.Concatenate()([user_emb, item_emb])
        # 드롭아웃 레이어를 적용하여 과적합을 방지합니다.
        dropout = tf.keras.layers.Dropout(self.dropout_rate)(concat_layer)

        # 여러 개의 완전 연결 레이어를 추가합니다.
        dense1 = tf.keras.layers.Dense(64, activation='relu')(dropout)
        dense2 = tf.keras.layers.Dense(32, activation='relu')(dense1)
        dense3 = tf.keras.layers.Dense(16, activation='relu')(dense2)
        dense4 = tf.keras.layers.Dense(8, activation='relu')(dense3)

        return dense4

    def _neural_cf(self, gmf, mlp):
        """GMF와 MLP 출력을 결합합니다."""
        # GMF와 MLP 출력을 연결합니다.
        concat_layer = tf.keras.layers.Concatenate()([gmf, mlp])
        # 최종 출력 레이어(시그모이드 활성화 함수를 사용한 이진 분류)
        return tf.keras.layers.Dense(1, activation='sigmoid')(concat_layer)

def train_model(x_train, y_train, n_user, n_item, num_epoch, batch_size):
    """Neural Collaborative Filtering 모델을 훈련시킵니다."""
    # 전체 배치 수를 계산합니다.
    num_batch = np.ceil(x_train[0].shape[0] / batch_size)

    # 모델 인스턴스를 생성합니다.
    ncf = NeuralCollaborativeFiltering(n_user, n_item)
    model = ncf.build_model()

    # 옵티마이저를 설정합니다.
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # 모델을 컴파일합니다.
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    # 데이터 로더 인스턴스를 생성합니다.
    data_loader = DataLoader()
    # 모델을 훈련시킵니다.
    model.fit(
        data_loader.batch_generator(
            x_train[0], x_train[1], y_train,
            batch_size=batch_size, n_batch=num_batch,
            shuffle=True, user_dim=n_user, item_dim=n_item),
        epochs=num_epoch,
        steps_per_epoch=num_batch,
        verbose=2
    )

    return model

def parse_args():
    """명령줄 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description="Neural Collaborative Filtering 모델 훈련")
    parser.add_argument('--model_dir', type=str, help="모델 출력 디렉토리")
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'), help="SageMaker 모델 출력 디렉토리")
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'), help="훈련 데이터 디렉토리")
    parser.add_argument('--hosts', type=json.loads, default=json.loads(os.environ.get('SM_HOSTS')), help="분산 훈련을 위한 호스트")
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'), help="분산 훈련에서 현재 호스트")
    parser.add_argument('--epochs', type=int, default=3, help="훈련 에포크 수")
    parser.add_argument('--batch_size', type=int, default=256, help="훈련 배치 크기")
    parser.add_argument('--n_user', type=int, required=True, help="사용자 수")
    parser.add_argument('--n_item', type=int, required=True, help="아이템 수")

    return parser.parse_args()

if __name__ == "__main__":
    # 명령줄 인자를 파싱합니다.
    args = parse_args()

    # 훈련 데이터를 로드합니다.
    data_loader = DataLoader()
    user_train, item_train, train_labels = data_loader.load_training_data(args.train)

    # 모델을 훈련시킵니다.
    ncf_model = train_model(
        x_train=[user_train, item_train],
        y_train=train_labels,
        n_user=args.n_user,
        n_item=args.n_item,
        num_epoch=args.epochs,
        batch_size=args.batch_size
    )

    # 모델을 저장합니다.
    if args.current_host == args.hosts[0]:
        ncf_model.save(os.path.join(args.sm_model_dir, '000000001'), 'neural_collaborative_filtering.h5')
