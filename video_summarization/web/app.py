import os.path

import gradio as gr
import cv2
import numpy as np
import yaml

from video_summarization.pipelines.predict import main
from video_summarization.src.io import get_images


def video_identity(video, fps=29, threshold=0.7):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "../../confs/predict.yaml")) as f:
        cfg = yaml.safe_load(f)

    cfg['video_path'] = video
    cfg['inference'] = True
    preds = main(cfg)['preds']
    imp_seconds = preds[preds['prob'] > threshold]['second'].values
    images = get_images(video, fps, transform=False)

    result_path = f"{os.path.basename(video)}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    result_video = cv2.VideoWriter(result_path, fourcc, fps, (images[0].width, images[0].height))
    for i, image in enumerate(images):
        if i // fps in imp_seconds:
            result_video.write(np.array(image))

    cv2.destroyAllWindows()
    result_video.release()
    return result_path


if __name__ == "__main__":
    video = gr.Interface(video_identity, gr.Video(), "video")
    gr.TabbedInterface([video], ["video"])
    video.launch()
