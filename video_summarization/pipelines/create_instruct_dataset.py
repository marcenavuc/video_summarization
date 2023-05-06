import json
import logging
import os

import hydra
import openai
import pandas as pd
import whisper
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

from video_summarization.src.asr import get_text_recognition
from video_summarization.src.ic import get_image_caption
from video_summarization.src.io import save_audio, get_images
from video_summarization.src.llm import create_prompt, transform_predictions, fill_preds, merge_results, \
    create_batch_prompt
from video_summarization.src.metric import eval_metric_dummy, eval_metric_f1_canonical


log = logging.getLogger(__name__)


@hydra.main(config_path="../../confs", config_name="predict")
def main(cfg):
    openai.api_key = os.getenv(cfg["open_ai_env_name"])

    vit_model = VisionEncoderDecoderModel.from_pretrained(cfg["image_caption"]["model_name"])
    vit_feature_extractor = ViTImageProcessor.from_pretrained(cfg["image_caption"]["feature_extractor"])
    vit_tokenizer = AutoTokenizer.from_pretrained(cfg["image_caption"]["tokenizer"])

    asr_model = whisper.load_model(cfg["asr"])

    labels_df = pd.read_parquet(os.path.join(cfg["dataset_path"], cfg["annotation_path"]))
    log.info("Load labels")

    video_path = os.path.join(cfg["dataset_path"], cfg["video_path"])
    ic_kwargs = cfg["image_caption"]["kwargs"]

    audio_path = save_audio(video_path, target_path=cfg["work_path"])
    log.info("Converted video to image")
    images = get_images(video_path, cfg["image_caption"]["fps"], cfg["image_caption"]["image_size"])
    ic_result = get_image_caption(images, vit_feature_extractor, vit_tokenizer, vit_model, **ic_kwargs)
    log.info("Got image caption")

    asr_result = get_text_recognition(asr_model, audio_path)

    all_texts = merge_results(ic_result, asr_result, cfg['max_tokens_in_batch'], max_tokens=500)
    prompts = []
    for batch_id in range(all_texts['batch'].nunique()):
        batch_df = all_texts[all_texts['batch'] == batch_id]
        batch_df = batch_df[['second', 'text_ic', 'text_asr']].fillna("")
        batch_df = batch_df.rename({'text_ic': 'image description', 'text_asr': 'speech recognition'})
        log.info("Batch: start {}, end {}".format(batch_df['second'].min(), batch_df['second'].max()))
        batch_prompt = create_batch_prompt(batch_df)

        prompts.append(batch_prompt)


if __name__ == "__main__":
    main()
