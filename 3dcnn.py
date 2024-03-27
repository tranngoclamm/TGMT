def load_and_prepare_data(video_file, label_file):
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

    return X_train, X_test, y_train, y_test, depth, height, width, channels, num_classes, batch_size, epochs
