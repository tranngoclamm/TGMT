from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Reshape

model = Sequential()

# Phần 2D: Xử lý không gian
model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same'), input_shape=(depth, height, width, channels)))
model.add(TimeDistributed(ReLU()))
model.add(TimeDistributed(MaxPooling2D((2, 2), padding='same')))
model.add(TimeDistributed(Dropout(0.25)))

model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same')))
model.add(TimeDistributed(ReLU()))
model.add(TimeDistributed(MaxPooling2D((2, 2), padding='same')))
model.add(TimeDistributed(Dropout(0.25)))

model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same')))
model.add(TimeDistributed(ReLU()))
model.add(TimeDistributed(MaxPooling2D((2, 2), padding='same')))
model.add(TimeDistributed(Dropout(0.25)))

# Chuyển từ 2D sang 1D
model.add(TimeDistributed(Flatten()))

# Phần 1D: Xử lý thời gian
model.add(Conv1D(256, 3, padding='same'))
model.add(ReLU())
model.add(MaxPooling1D(2, padding='same'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
