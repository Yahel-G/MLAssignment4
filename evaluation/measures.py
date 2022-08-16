from sklearn.metrics import accuracy_score, matthews_corrcoef, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def evaluate(y_true, y_score, y_pred, multiclass=False, display=True):

    if multiclass:

        acc = accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        area = roc_auc_score(y_true, y_score, average='micro', multi_class='ovr')
        prauc = average_precision_score(y_true, y_score, average='micro')

    else:

        acc = accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        area = roc_auc_score(y_true, y_score)
        prauc = average_precision_score(y_true, y_score)

    if display:
        print(f"""Accuracy: {acc:.2f}
                  MCC:      {mcc:.2f}
                  AUC:      {area:.2f}
                  PR_AUC:   {prauc:.2f}""")

    return acc, mcc, area, prauc
