import numpy as np
import pandas as pd


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
