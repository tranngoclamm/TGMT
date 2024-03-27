import cv2
import numpy as np
import os

def process_videos(data_dir, height, width):
    # Kích thước mới cho video
    new_size = (width, height)

    # Khởi tạo danh sách các video và nhãn
    videos = []
    labels = []

    # Duyệt qua các thư mục nhãn
    for label_dir in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_dir)

        # Duyệt qua các thư mục video ID
        for video_dir in os.listdir(label_path):
            video_path = os.path.join(label_path, video_dir)

            # Khởi tạo danh sách các frame ảnh trong video ID
            frames = []

            # Duyệt qua các frame ảnh trong video ID
            for frame_file in os.listdir(video_path):
                frame_path = os.path.join(video_path, frame_file)

                # Đọc frame ảnh và thay đổi kích thước
                frame = cv2.imread(frame_path)
                resized_frame = cv2.resize(frame, new_size)

                # Thêm frame ảnh đã thay đổi kích thước vào danh sách
                frames.append(resized_frame)

            # Thêm danh sách các frame ảnh vào danh sách videos
            videos.append(frames)
            labels.append(label_dir)

    # Kiểm tra kích thước của các video
    max_length = max(len(video) for video in videos)
    print("Kích thước tối đa của video:", max_length)

    # Chuyển đổi các video thành cùng kích thước
    for i in range(len(videos)):
        video = videos[i]
        while len(video) < max_length:
            # Tạo một frame ảnh trống có kích thước new_size
            empty_frame = np.zeros((height, width, 3), dtype=np.uint8)
            video.append(empty_frame)

    # Chuyển danh sách videos và labels thành mảng numpy
    videos = np.array(videos)
    labels = np.array(labels)

    # Kiểm tra kích thước và kiểu dữ liệu
    print(videos.shape)  # (số lượng mẫu, kích thước tối đa, chiều cao, chiều rộng, số kênh)
    print(videos.dtype)  # Kiểu dữ liệu của mảng videos
    print(labels.shape)  # (số lượng mẫu,)
    print(labels.dtype)  # Kiểu dữ liệu của mảng labels

    return videos, labels
