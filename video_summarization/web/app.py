import os

import gradio as gr
import cv2
from PIL import Image
import numpy as np
import yaml

from video_summarization.pipelines.predict import main


def get_images(video_path, fps):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    images = []

    for i in range(0, frame_count, int(cap.get(cv2.CAP_PROP_FPS) / fps)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            image = Image.fromarray(frame)
            # image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            images.append(image)

    cap.release()
    return images


def video_identity(video):
    threshold = 0.8
    fps = 29

    with open("/Users/m.averchenko/PycharmProjects/video_summarization/confs/predict.yaml") as f:
        cfg = yaml.safe_load(f)

    cfg['video_path'] = video
    cfg['inference'] = True
    preds = main(cfg)['preds']
    imp_seconds = preds[preds['prob'] > threshold]['second'].values
    images = get_images(video, fps)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    result_video = cv2.VideoWriter("result_video.mp4", fourcc, fps, (images[0].width, images[0].height))
    for i, image in enumerate(images):
        if i // fps in imp_seconds:
            result_video.write(np.array(image))

    cv2.destroyAllWindows()
    result_video.release()
    return "result_video.mp4"


if __name__ == "__main__":
    video = gr.Interface(video_identity, gr.Video(), "video")
    gr.TabbedInterface([video], ["video"])
    video.launch()
