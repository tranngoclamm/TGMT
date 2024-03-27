cnn import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense
from tensorflow.keras import Input
from keras.layers import ReLU
from keras.layers import Dropout
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def train_3dcnn(video_file, label_file):
     # Load dữ liệu từ file npy
    videos = np.load(video_file)
    labels = np.load(label_file)

    # Chuẩn hóa dữ liệu
    for i in range(len(videos)):
        videos[i] = (videos[i] * 255).astype('float16')

    # Trộn lên dữ liệu
    num_samples = videos.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    videos = videos[indices]
    labels = labels[indices]

    # Tính toán số lượng mẫu trong tập huấn luyện
    num_train_samples = int(num_samples * 0.8)

    # Tách dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train = videos[:num_train_samples]
    X_test = videos[num_train_samples:]

    encoder = LabelBinarizer()

    # Fit bộ mã hóa và chuyển đổi nhãn thành dạng one-hot encoding
    one_hot_labels = encoder.fit_transform(labels)

    # Chia nhãn dạng one-hot encoding thành tập huấn luyện và tập kiểm thử
    y_train = one_hot_labels[:num_train_samples]
    y_test = one_hot_labels[num_train_samples:]

    # Định nghĩa các thông số
    depth, height, width, channels = X_train.shape[1:]
    num_classes = np.unique(y_train).shape[0]
    batch_size = 10
    epochs = 15

    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(depth, height, width, channels), padding='same'))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv3D(128, kernel_size=(3, 3, 3), padding='same'))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # Huấn luyện mô hình
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    return model, history