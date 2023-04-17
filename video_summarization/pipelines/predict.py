import os

import hydra
import openai
import pandas as pd
import whisper
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

from video_summarization.src.asr import get_text_recognition
from video_summarization.src.ic import get_image_caption
from video_summarization.src.io import save_audio, get_images
from video_summarization.src.llm import create_prompt, transform_predictions, fill_preds
from video_summarization.src.metric import eval_metric


@hydra.main(config_path="../../confs", config_name="predict")
def main(cfg):
    openai.api_key = os.getenv(cfg["open_ai_env_name"])

    vit_model = VisionEncoderDecoderModel.from_pretrained(cfg["image_caption"]["model_name"])
    vit_feature_extractor = ViTImageProcessor.from_pretrained(cfg["image_caption"]["feature_extractor"])
    vit_tokenizer = AutoTokenizer.from_pretrained(cfg["image_caption"]["tokenizer"])

    asr_model = whisper.load_model(cfg["asr"])

    labels_df = pd.read_parquet(os.path.join(cfg["dataset_path"], cfg["annotation_path"]))

    video_path = os.path.join(cfg["dataset_path"], cfg["video_path"])
    ic_kwargs = cfg["image_caption"]["kwargs"]

    audio_path = save_audio(video_path, target_path=cfg["work_path"])
    images = get_images(video_path, cfg["image_caption"]["fps"], cfg["image_caption"]["image_size"])
    ic_result = get_image_caption(images, vit_feature_extractor, vit_tokenizer, vit_model, **ic_kwargs)

    asr_result = get_text_recognition(asr_model, audio_path)
    prompt = create_prompt(asr_result, ic_result)
    print(prompt)

    response = openai.Completion.create(**dict({"prompt": prompt}, **dict(cfg["llm"])))

    print("Response from gpt:", response['choices'][0]['text'])
    tdf = transform_predictions(response['choices'][0]['text'])
    preds = fill_preds(ic_result, tdf)

    print("Sample of preds", preds)
    filename = os.path.basename(video_path).split(".")[0]
    annotation = labels_df[labels_df['video_id'] == filename].annotation.values[0]

    result_metric = eval_metric(preds, annotation)
    print(result_metric)


if __name__ == "__main__":
    main()
