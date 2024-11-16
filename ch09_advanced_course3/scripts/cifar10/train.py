
import tensorflow as tf
import argparse
import os
import numpy as np

def load_data():
    x_train = np.load(os.path.join('/opt/ml/input/data/train', 'x_train.npy'))
    y_train = np.load(os.path.join('/opt/ml/input/data/train', 'y_train.npy'))
    x_test = np.load(os.path.join('/opt/ml/input/data/test', 'x_test.npy'))
    y_test = np.load(os.path.join('/opt/ml/input/data/test', 'y_test.npy'))
    return (x_train, y_train), (x_test, y_test)

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    return model

def model(x_train, y_train, x_test, y_test, batch_size, epochs, learning_rate, model_dir):
    model = create_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
    
    # 모델 저장
    model.save(os.path.join(model_dir, 'model'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--model_dir', type=str, default='/opt/ml/model')
    args = parser.parse_args()
    
    (x_train, y_train), (x_test, y_test) = load_data()
    
    model(x_train, y_train, x_test, y_test, args.batch_size, args.epochs, args.learning_rate, args.model_dir)
