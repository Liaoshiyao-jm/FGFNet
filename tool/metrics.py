import numpy as np
from sklearn.metrics import confusion_matrix


def segmentation_metrics(y_true_list, y_pred_list, num_classes, foreground=None):
    if foreground is None:
        foreground = np.ones(num_classes, dtype=bool)
    foreground = np.asarray(foreground, dtype=bool)

    cm = confusion_matrix(
        y_true=np.concatenate([y.ravel() for y in y_true_list]),
        y_pred=np.concatenate([y.ravel() for y in y_pred_list]),
        labels=range(num_classes),
    )

    total = np.sum(cm)
    oa = np.trace(cm) / total * 100.0
    f1 = np.zeros(num_classes)
    iou = np.zeros(num_classes)
    for i in range(num_classes):
        precision_recall_den = np.sum(cm[i, :]) + np.sum(cm[:, i])
        union = np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i]
        f1[i] = 2.0 * cm[i, i] / precision_recall_den if precision_recall_den > 0 else 0.0
        iou[i] = cm[i, i] / union if union > 0 else 0.0

    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe) if pe < 1 else 0.0

    return {
        "OA": oa,
        "mF1": float(f1[foreground].mean()),
        "miou": float(iou[foreground].mean()),
        "Kappa": float(kappa),
        "confusion_matrix": cm,
    }
