'''
@Project: MPK-GNN
@File   : ops_tt.py
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc
    
'''
import os
import torch
import numpy as np
import pandas as pd

from utils.ops_ev import accuracy


def adjust_learning_rate(optimizer, lr, decay, global_step, decay_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #   lr = args.lr * (0.1 ** (epoch // 30))
    lr = lr * pow(decay, float(global_step // decay_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def test_model(net, loader, device, L_1, L_2, L_3, num_classes):
    net.eval()
    test_acc = 0
    count = 0
    confusionGCN = np.zeros([num_classes, num_classes])
    predictions = pd.DataFrame()
    y_true = []

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        pred, _, _, _ = net(batch_x, L_1, L_2, L_3)

        test_acc += accuracy(pred, batch_y).item()
        count += 1
        y_true.append(batch_y.item())
        # y_pred.append(pred.max(1)[1].item())
        confusionGCN[batch_y.item(), pred.max(1)[1].item()] += 1
        px = pd.DataFrame(pred.detach().cpu().numpy())
        predictions = pd.concat((predictions, px), 0)

    preds_labels = np.argmax(np.asarray(predictions), 1)
    test_acc = test_acc / float(count)
    predictions.insert(0, 'trueLabels', y_true)

    return test_acc, confusionGCN, predictions, preds_labels
