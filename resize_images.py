import os
import cv2
import numpy as np

def resize_frames(input_folder):
    data_sight_path = os.path.join("data_resize")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if not os.path.exists(data_sight_path):
        os.makedirs(data_sight_path)
        
    for label_folder in os.listdir(input_folder):
        label_folder_path = os.path.join(input_folder, label_folder)
        if not os.path.isdir(label_folder_path):
            continue
        
        label_data_sight_path = os.path.join(data_sight_path, label_folder)
        if not os.path.exists(label_data_sight_path):
            os.makedirs(label_data_sight_path)
        
        for video_folder in os.listdir(label_folder_path):
            video_folder_path = os.path.join(label_folder_path, video_folder)
            if not os.path.isdir(video_folder_path):
                continue
            
            video_data_sight_path = os.path.join(label_data_sight_path, video_folder)
            if not os.path.exists(video_data_sight_path):
                os.makedirs(video_data_sight_path)
            
            for frame_file in os.listdir(video_folder_path):
                frame_path = os.path.join(video_folder_path, frame_file)

                if frame_file.endswith('.png'):
                    frame = cv2.imread(frame_path)

                    # Detect faces
                    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                    for (x, y, w, h) in faces:
                        # Cut a fixed size around the face
                       # Cut a fixed size around the face
                        start_x = max(0, x - 2*w)
                        start_y = max(0, y - h)
                        end_x = min(frame.shape[1], x + 3*w)
                        end_y = min(frame.shape[0], y + 5*h)

                        # Cut the image
                        cut_image = frame[start_y:end_y, start_x:end_x]

                        # Resize the image to a fixed size
                        resized_image = cv2.resize(cut_image, (405, 486))  # 

                        # Save the cut image
                        cut_image_path = os.path.join(video_data_sight_path, f"cut_{frame_file}")
                        cv2.imwrite(cut_image_path, resized_image)


