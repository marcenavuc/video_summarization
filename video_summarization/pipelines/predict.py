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
    if not os.path.isfile(video_path):
        video_path = cfg["video_path"]
    ic_kwargs = cfg["image_caption"]["kwargs"]

    audio_path = save_audio(video_path, target_path=cfg["work_path"])
    log.info("Converted video to image")
    images = get_images(video_path, cfg["image_caption"]["fps"], cfg["image_caption"]["image_size"])
    ic_result = get_image_caption(images, vit_feature_extractor, vit_tokenizer, vit_model, **ic_kwargs)
    log.info("Got image caption")

    asr_result = get_text_recognition(asr_model, audio_path)

    all_texts = merge_results(ic_result, asr_result, cfg['max_tokens_in_batch'])
    tdf = []
    for batch_id in range(all_texts['batch'].nunique()):
        batch_df = all_texts[all_texts['batch'] == batch_id]
        batch_df = batch_df[['second', 'text_ic', 'text_asr']].fillna("")
        batch_df = batch_df.rename({'text_ic': 'image description', 'text_asr': 'speech recognition'})
        log.info("Batch: start {}, end {}".format(batch_df['second'].min(), batch_df['second'].max()))
        batch_prompt = create_batch_prompt(batch_df)

        response = openai.Completion.create(**dict({"prompt": batch_prompt}, **dict(cfg["llm"])))
        log.info("Response from gpt: {}".format(response['choices'][0]['text']))
        batch_tdf = transform_predictions(response['choices'][0]['text'])
        tdf.append(batch_tdf)
    tdf = pd.concat(tdf)

    preds = fill_preds(ic_result, tdf)
    log.info("Sample of preds {}".format(preds))

    if cfg['inference']:
        return {"preds": preds}
    else:
        filename = os.path.basename(video_path).split(".")[0]
        annotation = labels_df[labels_df['video_id'] == filename].annotation.values[0]

        log.info("Metrics")
        result = {
            "preds": preds,
            "metrics": {
                'eval_metric_dummy': eval_metric_dummy(preds, annotation),
                'eval_metric_dummy_t_0.25': eval_metric_dummy(preds, annotation, t=0.25),
                'eval_metric_dummy_t_0.1': eval_metric_dummy(preds, annotation, t=0.1),
                'eval_metric_f1_canonical': eval_metric_f1_canonical(preds, annotation)
            }
        }

        log.info(json.dumps(result['metrics']))

        return result


if __name__ == "__main__":
    main()
