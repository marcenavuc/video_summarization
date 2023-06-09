import numpy as np
import pandas as pd


def create_prompt(asr_result, ic_result):
    return """
        Summarize the video by description data
        Decription of each image per second:
        {ic_result}

        Speech recognition results:
        {asr_result}
        Return summarization as table with columns of second and importance. importance should be from 0 to 1.
        """.format(ic_result=ic_result.to_string(index=None),
                   asr_result=asr_result.to_string(index=None)
                   )


def create_batch_prompt(batch):
    return """
        Summarize the video by description data

        {batch}

        Return summarization as table with columns: second, importance. importance should be from 0 to 1.
        Summarization should has following structure:
        Second Importance
        <start_second>  <importance>
        ....
        <end_second>   <importance>
        """.format(batch=batch.to_string(index=None))


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


def find_ind(x, asr_result):
    start = asr_result['start_second'].values
    end = asr_result['end_second'].values
    res = np.where((start <= x) & (x <= end))[0]
    return np.nan if res.shape[0] == 0 else res[0]


def merge_results(ic_result, asr_result, max_tokens=2000):
    ic_result['wc'] = ic_result['text'].str.split().apply(len)
    ic_result['cwc'] = ic_result['wc'].cumsum()

    asr_result['wc'] = asr_result['text'].str.split().apply(len)
    asr_result['cwc'] = asr_result['wc'].cumsum()

    ic_result['index'] = ic_result['second'].apply(lambda x: find_ind(x, asr_result))

    all_texts = ic_result.join(asr_result.reset_index(), how='left', on='index', lsuffix='_ic', rsuffix='_asr')
    all_texts['wc_all'] = (all_texts['wc_ic'] + all_texts['wc_asr'].fillna(0)).cumsum()
    all_texts['batch'] = (all_texts['wc_all'] // max_tokens).astype(int)
    return all_texts



def create_batch_annotation_prompt(annotation, start, end):
    
    return """
Second Importance 
{}
""".format("\n".join([f"{num} {annotation[i]}" for i, num in enumerate(range(start, end))]))