import torch as th
from sklearn.metrics import roc_auc_score, accuracy_score


def accuracy_metric(y_true, y_logits):
    y_pred = th.argmax(y_logits, dim=1)
    return accuracy_score(y_true.numpy(), y_pred.numpy())


def roc_auc_metric(y_true, y_logits):
    assert y_logits.shape[1] == 2
    y_p = th.exp(y_logits)
    return roc_auc_score(y_true, y_p[:,1].numpy())
