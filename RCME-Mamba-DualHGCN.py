#encoding=utf-8
import os
import time
import copy
import torch
import torch.optim as optim
from visual_data import load_feature_construct_H
import hypergraph_utils as hgut
from HGNN import HGNN
from sklearn.metrics import precision_score, recall_score, f1_score
import h5py
import numpy as np
from scipy.io import loadmat

'''
    作者：程文鑫 2024 12 14
'''

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# initialize data
data_dir = "C:/Users/DELL/Desktop/MPDB_Datasets/EEG_1500_EN/sub15/Feature_ME.mat"
lbls_dir = "C:/Users/DELL/Desktop/MPDB_Datasets/EEG_1500_EN/sub15/y.mat"


#处理特征矩阵、标签矩阵、训练集、测试集以及构建超图结构

fts, lbls, idx_train, idx_test, H = load_feature_construct_H(data_dir,lbls_dir)
#fts:64*5234   lbls:5234*1    idx_train:4187*1   idx_test:1047*1    H:64*128


G = hgut.generate_G_from_H(H)
n_class = int(lbls.max()) + 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# transform data to device
fts = torch.Tensor(fts).to(device)
lbls = torch.Tensor(lbls).squeeze().long().to(device)
G = torch.Tensor(G).to(device)
idx_train = torch.Tensor(idx_train).long().to(device)
idx_test = torch.Tensor(idx_test).long().to(device)


# 在每个phase之后计算精确度、召回率和F1分数
def compute_metrics(preds, lbls, num_classes=5):
    # 将preds和lbls转换为numpy数组，并展平为一维
    preds_np = preds.cpu().numpy().flatten()
    lbls_np = lbls.cpu().numpy().flatten()

    # 对于多分类问题，需要指定average参数，例如'macro', 'micro', 'weighted'等
    precision = precision_score(lbls_np, preds_np, average='macro', zero_division=1)
    recall = recall_score(lbls_np, preds_np, average='macro', zero_division=1)
    f1 = f1_score(lbls_np, preds_np, average='macro', zero_division=1)
    return precision, recall, f1

def train_model(model, criterion, optimizer, scheduler, num_epochs=50, print_freq=500):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0

    for epoch in range(num_epochs):
        if epoch % print_freq == 0:
            print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs - 1}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            idx = idx_train if phase == 'train' else idx_test

            # Iterate over data.
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(fts, G)
                loss = criterion(outputs[idx], lbls[idx])
                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * fts.size(0)
            running_corrects += torch.sum(preds[idx] == lbls.data[idx])

            epoch_loss = running_loss / len(idx)
            epoch_acc = running_corrects.double() / len(idx)

            if epoch % print_freq == 0:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if epoch % print_freq == 0:
            print(f'Best val Acc: {best_acc:4f}')
            print('-' * 20)

        precision, recall, f1 = compute_metrics(preds[idx], lbls[idx].data, num_classes=5)
        if epoch % print_freq == 0:
            print(
                f'{phase} Loss: {epoch_loss:.4f} Acc: {best_acc:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1 Score: {f1:.4f}')
        if precision > best_precision:
            best_precision = precision
        if recall > best_recall:
            best_recall = recall
        if f1 > best_f1:
            best_f1 = f1


    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_precision:4f}')
    print(f'Best val precision: {best_acc:4f}')
    print(f'Best val recall: {best_recall:4f}')
    print(f'Best val f1: {best_f1:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
def _main():
    model_ft = HGNN(in_ch=fts.shape[1],
                    n_class=n_class,
                    n_hid=128,
                    dropout=0.5)
    model_ft = model_ft.to(device)
    optimizer = optim.Adam(model_ft.parameters(), lr=0.001,
                           weight_decay=0.0005)
    schedular = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[100],
                                               gamma=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    model_ft = train_model(model_ft, criterion, optimizer, schedular, 2000, print_freq=50)

if __name__ == '__main__':
    _main()


