'''
@Project: MPK-GNN
@File   : ops_al.py
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc
    Evaluation metrics for classification tasks
'''
from sklearn import metrics


def accuracy(output, labels): # average of each batch
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def get_classification_results(labels_true, labels_pred):
    """
    :param y_true:
        data type: numpy.ndarray
        shape: (n_samples,)
        sample: [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
    :param y_pred:
        data type: numpy.ndarray
        shape: (n_samples,)
        sample: [ 0  1  2  3  4  5  6  7  8  9ops_io.py 10 11 12 13 14 15 16 17 18 19]
    :return:
    """
    ACC = metrics.accuracy_score(labels_true, labels_pred)
    MACRO_P = metrics.precision_score(labels_true, labels_pred, average='macro')
    MACRO_R = metrics.recall_score(labels_true, labels_pred, average='macro')
    MACRO_F1 = metrics.f1_score(labels_true, labels_pred, average='macro')
    MICRO_F1 = metrics.f1_score(labels_true, labels_pred, average='micro')

    return ACC, MACRO_P, MACRO_R, MACRO_F1, MICRO_F1