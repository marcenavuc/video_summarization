import numpy as np
from sklearn.metrics import f1_score


def spread_preds(preds, annotation):
    post_preds = preds.copy()
    post_preds['reps'] = annotation.shape[0] // preds['second'].max()
    post_preds = post_preds.loc[post_preds.index.repeat(post_preds.reps)]
    return post_preds['prob'].values


def eval_metric_dummy(preds, annotation, t=0.5):
    post_preds = spread_preds(preds, annotation)

    same_len = min(post_preds.shape[0], annotation.shape[0])
    y_true = np.where(annotation[:same_len] >= t, 1, 0)
    y_pred = np.where(post_preds[:same_len] >= t, 1, 0)
    return f1_score(y_true, y_pred)


def eval_metric_f1_canonical(preds, annotation, labels_count=5):
    labels = np.linspace(0, 1, labels_count + 1)[1:]
    post_preds = spread_preds(preds, annotation)

    same_len = min(post_preds.shape[0], annotation.shape[0])
    transform = np.vectorize(lambda x: (x >= labels).sum())
    y_true = transform(annotation[:same_len])
    y_pred = transform(post_preds[:same_len])
    return f1_score(y_true, y_pred, average='micro')