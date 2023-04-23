'''
    @Project: MPK-GNN
    @File   : run_results.py
    @Author : Shunxin Xiao
    @Email  : xiaoshunxin.tj@gmail
    @Desc

'''
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

from models.mpk import MPK
from utils.ops_al import weight_init
from utils.ops_loss import SupConLoss
from utils.ops_tt import adjust_learning_rate, test_model
from utils.ops_io import process_data, load_processed_data
from utils.ops_ev import accuracy, get_classification_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Parameters of basic setting
    parser.add_argument('--seed', type=int, default=1009, help='Number of seed.')
    parser.add_argument('--n_repeated', type=int, default=10, help='Number of repeated experiments')
    parser.add_argument('--cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument('--cuda_device', type=str, default='3', help='The number of cuda device.')
    # Parameters of data loading
    parser.add_argument('--load_saved', action='store_true', default=True, help='Whether to load the saved data.')
    parser.add_argument('--dataset_name', type=str, default='1000', help='The number of cuda device.')
    parser.add_argument('--ratio', type=float, default=0.1, help='percentage training samples.')
    parser.add_argument('--batch_size', type=int, default=128, help='The batch size of training data')
    # Parameters of network framework
    parser.add_argument('--num_omics', type=int, default=2, help='Number of omics')
    parser.add_argument('--slm_dim_1', type=int, default=1024, help='the output dimension of the first layer (dim_1) of SLM')
    parser.add_argument('--slm_dim_2', type=int, default=256, help='the output dimension of the second layer (dim_1) of SLM')
    parser.add_argument('--flm_gcn_dim_1', type=int, default=64, help='The output dimension of the first layer of GCN in the FLM module')
    parser.add_argument('--flm_gcn_dim_2', type=int, default=8, help='The output dimension of the second layer of GCN in the FLM module')
    parser.add_argument('--pool_size', type=int, default=8, help='The size of pooling layer used in graph_max_pool. Must be a power of 2.')
    parser.add_argument('--flm_fl_dim', type=int, default=1024, help='The dimension of the flatten layer (fl) of the FLM module')
    parser.add_argument('--pm_dim_1', type=int, default=32, help='the output dimension of the Projection Module')
    parser.add_argument('--pm_dim_2', type=int, default=1024, help='the output dimension of the Projection Module')
    # Parameters of training process
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate.')
    parser.add_argument('--decay', type=float, default=0.95, help='The decay value of learning rate.')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='The value of L2 regularization.')
    parser.add_argument('--lambda_1', type=float, default=0.5, help='Weight of the first augmentation loss')
    parser.add_argument('--temperature', type=float, default=0.2, help='Parameter of contrastive learning')
    args = parser.parse_args()

    all_ACC = []
    all_MaP = []
    all_MaR = []
    all_MaF = []
    all_Time = []

    for i in range(args.n_repeated):
        if i == 0:
            # args.load_saved = False
            args.load_saved = True
        else:
            args.load_saved = True

        # Load data
        args.num_genes = int(args.dataset_name)
        if args.load_saved:
            train_data, train_labels, val_data, val_labels, test_data, test_labels, L_1, L_2, L_3, num_classes = \
                load_processed_data(num_genes=args.num_genes, ratio=args.ratio)
        else:
            train_data, train_labels, val_data, val_labels, test_data, test_labels, L_1, L_2, L_3, num_classes = \
                process_data(num_genes=args.num_genes, ratio=args.ratio)

        dset_train = Data.TensorDataset(train_data, train_labels)
        train_loader = Data.DataLoader(dset_train, batch_size=args.batch_size, shuffle=True)
        dset_val = Data.TensorDataset(val_data, val_labels)
        val_loader = Data.DataLoader(dset_val, shuffle=False)
        dset_test = Data.TensorDataset(test_data, test_labels)
        test_loader = Data.DataLoader(dset_test, shuffle=False)

        # Instantiate the network and optimizer
        net = MPK(num_omics=args.num_omics, num_genes=args.num_genes, num_classes=num_classes, slm_dim_1=args.slm_dim_1,
                  slm_dim_2=args.slm_dim_2, flm_gcn_dim_1=args.flm_gcn_dim_1, flm_gcn_dim_2=args.flm_gcn_dim_2,
                  pool_size=args.pool_size, flm_fl_dim=args.flm_fl_dim, pm_dim_1=args.pm_dim_1, pm_dim_2=args.pm_dim_2)
        net.apply(weight_init)
        optimizer = optim.SGD(net.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.weight_decay)
        device = torch.device("cuda:" + args.cuda_device if args.cuda else "cpu")
        criterion = SupConLoss(temperature=args.temperature, cuda_device=args.cuda_device)

        net = net.to(device)
        L_1 = L_1.to(device)
        L_2 = L_2.to(device)
        L_3 = L_3.to(device)

        # Begin to training....
        t_total_train = time.time()
        cur_lr = args.lr
        global_step = 0
        train_size = train_data.shape[0]

        for epoch in range(args.num_epochs):  # loop over the dataset multiple times
            net.train()
            cur_lr = adjust_learning_rate(optimizer, cur_lr, args.decay, global_step, train_size)  # update learning rate

            t_start = time.time()  # reset time

            # extract batches
            epoch_loss = 0.0
            epoch_acc = 0.0
            count = 0
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                optimizer.zero_grad()
                output, r_x_0, r_x_1, r_x_2 = net(batch_x, L_1, L_2, L_3)

                # batch_x = batch_x.view(batch_x.size()[0], -1)
                # loss = nn.MSELoss()(out_gae, batch_x)
                loss = nn.CrossEntropyLoss()(output, batch_y)
                loss += args.lambda_1 * criterion(
                    torch.cat([r_x_0.unsqueeze(1), r_x_1.unsqueeze(1), r_x_2.unsqueeze(1)], dim=1), batch_y)

                acc_batch = accuracy(output, batch_y).item()

                loss.backward()
                optimizer.step()

                count += 1
                epoch_loss += loss.item()
                epoch_acc += acc_batch
                global_step += args.batch_size

            epoch_loss /= count
            epoch_acc /= count
            t_stop = time.time() - t_start
            print('epoch= %d, loss(train)= %.3f, accuracy(train)= %.3f, time= %.3f, lr= %.5f' %
                  (epoch + 1, epoch_loss, epoch_acc, t_stop, cur_lr))
        t_total_train = time.time() - t_total_train
        print("Total training time: ", t_total_train)

        # Begin to testing...
        t_start_test = time.time()
        test_acc, confusionGCN, predictions, preds_labels = test_model(net, test_loader, device, L_1, L_2, L_3,
                                                                       num_classes)

        ACC, MACRO_P, MACRO_R, MACRO_F1, MICRO_F1 = get_classification_results(test_labels, preds_labels)
        all_ACC.append(ACC)
        all_MaP.append(MACRO_P)
        all_MaR.append(MACRO_R)
        all_MaF.append(MACRO_F1)
        all_Time.append(t_total_train)

    # fp = open("results.txt", "a+", encoding="utf-8")
    fp = open(str(args.num_genes) + ".txt", "a+", encoding="utf-8")
    fp.write("Ratio: {}\n".format(args.ratio))
    fp.write("ACC: {:.2f}\t{:.2f}\n".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
    fp.write("MaP: {:.2f}\t{:.2f}\n".format(np.mean(all_MaP) * 100, np.std(all_MaP) * 100))
    fp.write("MaR: {:.2f}\t{:.2f}\n".format(np.mean(all_MaR) * 100, np.std(all_MaR) * 100))
    fp.write("MaF: {:.2f}\t{:.2f}\n\n".format(np.mean(all_MaF) * 100, np.std(all_MaF) * 100))
    fp.write("Train Time: {:.2f}\t{:.2f}\n\n".format(np.mean(all_Time), np.std(all_Time)))
    fp.close()
