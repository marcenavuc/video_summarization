import os

import cv2
import openai
import pandas as pd
import numpy as np
import whisper
from PIL import Image
from moviepy.editor import VideoFileClip
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from sklearn.metrics import f1_score

openai.api_key = os.getenv("OPENAI_APIKEY")

vit_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vit_feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vit_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

asr_model = whisper.load_model("tiny.en")

FPS = 1
IMAGE_SIZE = (224, 224)
video_path = "/media/mark/ADATA HV620S/diplom/tvsum/ydata-tvsum50-v1_1/video/Bhxk-O1Y7Ho.mp4"

labels_df = pd.read_parquet('labels.parquet')
labels_df = labels_df.reset_index().drop(columns=['index'])


def save_audio(filepath):
    filename = os.path.basename(filepath).split(".")[0]
    result_path = filename + ".mp3"
    VideoFileClip(filepath).audio.write_audiofile(result_path)
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


def get_image_caption(images, feature_extractor, tokenizer, model, **kwags):
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values, **kwags)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    df = pd.DataFrame({"text": [pred.strip() for pred in preds]})
    df = df.reset_index().rename(columns={"index": "second"})
    return df


def get_text_recognition(asr_model, audio_file):
    asr_result = asr_model.transcribe(audio_file)
    return pd.DataFrame([
        {
            'start_second': segment['start'],
            'end_second': segment['end'],
            'text': segment['text'],
        }
        for segment in asr_result['segments']
    ])


def create_prompt(asr_result, ic_result):
    return """
        Summarize the video by description data
        Decription of each image per second:
        {ic_result}

        Speech recognition results:
        {asr_result}
        Return summarization as table with columns of second and importance. importance should be from 0 to 1
        """.format(ic_result=ic_result.to_string(index=None),
                   asr_result=asr_result.to_string(index=None)
                   )


def transform_predictions(pred_text):
    result = []
    for line in pred_text.split('\n'):
        pnumbers = line.strip().split()
        if len(pnumbers) != 2:
            continue
        try:
            result.append({'second': float(pnumbers[0]), 'prob': float(pnumbers[1])})
        except:
            continue
    return pd.DataFrame(result)


def fill_preds(ic_result, preds):
    sec = np.array(range(ic_result['second'].max() + 1))
    prob = np.zeros(shape=sec.shape)

    for i in range(preds.shape[0] - 1):
        prob[(sec >= preds.iloc[i].second) & (sec <= preds.iloc[i + 1].second)] = preds.iloc[i].prob
    last_i = preds.shape[0] - 1
    prob[sec >= preds.iloc[last_i].second] = preds.iloc[last_i].prob
    return pd.DataFrame({'second': sec, 'prob': prob})


def eval_metric(preds, annotation, t=0.5):
    post_preds = preds.copy()
    post_preds['reps'] = annotation.shape[0] // preds['second'].max()
    post_preds = post_preds.loc[post_preds.index.repeat(post_preds.reps)]
    post_preds = post_preds['prob'].values

    same_len = min(post_preds.shape[0], annotation.shape[0])
    y_true = np.where(annotation[:same_len] >= t, 1, 0)
    y_pred = np.where(post_preds[:same_len] >= t, 1, 0)
    return f1_score(y_true, y_pred)
