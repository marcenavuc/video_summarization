import os

import cv2
from PIL import Image
from moviepy.editor import VideoFileClip


def save_audio(file_path, target_path=None):
    filename = os.path.basename(file_path).split(".")[0]
    if target_path is None:
        result_path = filename + ".mp3"
    else:
        result_path = os.path.join(target_path, filename + ".mp3")
    VideoFileClip(file_path).audio.write_audiofile(result_path)
    return result_path


def get_images(video_path, fps, image_size):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    images = []

    for i in range(0, frame_count, int(cap.get(cv2.CAP_PROP_FPS) / fps)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(image_size)
            images.append(image)

    cap.release()
    return images
