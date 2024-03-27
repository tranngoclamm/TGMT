import os
import cv2

def get_video_frames(video_path, label, output_folder, video_skip=2, video_count_limit=3):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(output_folder, label, video_name)
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        return

    frame_count = 0
    video_count = 0  # Biến đếm số lượng video đã được xử lý

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (end of video?). Exiting ...")
            break

        frame_count += 1
        frame_path = os.path.join(output_folder, f"frame{frame_count}.png")
        cv2.imwrite(frame_path, frame)

        if cv2.waitKey(1) == ord('q'):
            break

        if frame_count % (video_skip + 1) == 0:
            video_count += 1

        if video_count == video_count_limit:  # Chỉ xử lý mỗi video_count_limit video
            break

    cap.release()
