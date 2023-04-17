import numpy as np
from sklearn.metrics import f1_score


def eval_metric(preds, annotation, t=0.5):
    post_preds = preds.copy()
    post_preds['reps'] = annotation.shape[0] // preds['second'].max()
    post_preds = post_preds.loc[post_preds.index.repeat(post_preds.reps)]
    post_preds = post_preds['prob'].values

    same_len = min(post_preds.shape[0], annotation.shape[0])
    y_true = np.where(annotation[:same_len] >= t, 1, 0)
    y_pred = np.where(post_preds[:same_len] >= t, 1, 0)
    return f1_score(y_true, y_pred)
