import pandas as pd


def get_image_caption(images, feature_extractor, tokenizer, model, **kwags):
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values, **kwags)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    df = pd.DataFrame({"text": [pred.strip() for pred in preds]})
    df = df.reset_index().rename(columns={"index": "second"})
    return df

