import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import os
import sys
import warnings
from tqdm import tqdm
import argparse

from PIL import Image
import numpy as np
import pandas as pd

import torch
import torch.utils.data as data
from torchvision import transforms

from sklearn.metrics import balanced_accuracy_score

from networks.SWP import SWP

from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss


class varifocal_loss(nn.Module):
    """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_

    Args:
        pred (Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (Tensor): The learning target of the iou-aware
            classification score with shape (N, C), C is the number of classes.
        alpha (float, optional): A balance factor for the negative part of
            Varifocal Loss, which is different from the alpha of Focal Loss.
            Defaults to 0.75.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        iou_weighted (bool, optional): Whether to weight the loss of the
            positive example with the iou target. Defaults to True.
    """

    def __init__(self, alpha=0.75, gamma=2.0, iou_weighted=False, use_sigmoid=True):
        super(varifocal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = iou_weighted
        self.sigmoid = use_sigmoid

    def forward(self, pred, target):
        if self.sigmoid:
            pred_new = F.sigmoid(pred)
        else:
            pred_new = pred
        target = target.cast(pred.dtype)
        if self.weight:
            focal_weight = target * (target > 0.0).cast('float32') + \
                           self.alpha * (pred_new - target).abs().pow(self.gamma) * \
                           (target <= 0.0).cast('float32')
        else:
            focal_weight = (target > 0.0).cast('float32') + \
                           self.alpha * (pred_new - target).abs().pow(self.gamma) * \
                           (target <= 0.0).cast('float32')

        if self.sigmoid:
            loss = nn.CrossEntropyLoss(weight=focal_weight)(pred, target)
        else:
            loss = nn.CrossEntropyLoss(weight=focal_weight)(pred, target)
            loss = loss.sum(axis=1)
        return loss


eps = sys.float_info.epsilon


def validate(test_loader, model, criterion, args, size_test_df, y_pred, y_true):
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            output = model(images)
            _, preds = torch.max(output, 1)  # preds是預測結果
            loss = criterion(output, target)

            y_pred.extend(preds.view(-1).detach().cpu().numpy())  # 將preds預測結果detach出來，並轉成numpy格式
            y_true.extend(target.view(-1).detach().cpu().numpy())  # target是ground-truth的label

    return y_pred, y_true

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default=r'C:\Users\s1810\Desktop\Database\RaFD\basic/', help='Raf-DB dataset path.')
    parser.add_argument('--checkpoint', type=str, default=r'C:\master\coder\thesis\SWP-main/best_rafdb.pth',help='weight path.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=1, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs.')
    return parser.parse_args()


class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform = None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path

        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None,names=['name','label'])

        if phase == 'train':
            self.data = df[df['name'].str.startswith('train')]
        else:
            self.data = df[df['name'].str.startswith('test')]

        file_names = self.data.loc[:, 'name'].values
        self.label = self.data.loc[:, 'label'].values - 1 # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        _, self.sample_counts = np.unique(self.label, return_counts=True)
        # print(f' distribution of {phase} samples: {self.sample_counts}')

        self.file_paths = []
        for f in file_names:
            f = f.split(".")[0]
            f = f +"_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.label[idx]

        if self.transform is not None:
            image = self.transform(image)
        return image, label



def run_training():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model = SWP()

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
    print('FLOPs:',flops,' ,Parm#:',params)


    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.RandomRotation(20),
            transforms.RandomCrop(224, padding=32)
        ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25)),
    ])
    train_dataset = RafDataSet(args.raf_path, phase = 'train', transform = data_transforms)
    print('Whole train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True,
                                               pin_memory = True)

    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    val_dataset = RafDataSet(args.raf_path, phase = 'test', transform = data_transforms_val)
    print('Validation set size:', val_dataset.__len__())

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,
                                               pin_memory = True)

    criterion_cls = torch.nn.CrossEntropyLoss()
    _focal_cls = FocalLoss(gamma=2, weight=None)
    _varifocal_cls = varifocal_loss(alpha=0.75, gamma=2.0, iou_weighted=False, use_sigmoid=True)



    params = list(model.parameters())
    # optimizer = torch.optim.ASGD(params,lr=args.lr, weight_decay = 1e-4)
    optimizer = torch.optim.SGD(params,lr=args.lr, weight_decay = 1e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


    best_acc = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
    # for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()

        for (imgs, targets) in train_loader:
            iter_cnt += 1
            optimizer.zero_grad()

            imgs = imgs.to(device)
            targets = targets.to(device)

            out1, out = model(imgs)
            # print(out,targets)

            _CE_loss = criterion_cls(out, targets)
            _focal_loss = _focal_cls(out, targets)
            # _varifocal_loss = _varifocal_cls(out, targets)
            loss = _focal_loss

            loss.backward()
            optimizer.step()

            running_loss += loss
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss/iter_cnt
        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (epoch, acc, running_loss,optimizer.param_groups[0]['lr']))

        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0

            ## for calculating balanced accuracy
            y_true = []
            y_pred = []

            model.eval()
            for (imgs, targets) in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)

                out1, out = model(imgs)
                _CE_loss = criterion_cls(out,targets)
                _focal_loss = _focal_cls(out,targets)
                # _varifocal_loss = _varifocal_cls(out, targets)
                loss = _focal_loss

                running_loss += loss
                iter_cnt+=1
                _, predicts = torch.max(out, 1)
                correct_num  = torch.eq(predicts,targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)

                y_true.append(targets.cpu().numpy())
                y_pred.append(predicts.cpu().numpy())

            running_loss = running_loss/iter_cnt
            scheduler.step()

            acc = bingo_cnt.float()/float(sample_cnt)
            acc = np.around(acc.numpy(),4)
            best_acc = max(acc,best_acc)

            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            balanced_acc = np.around(balanced_accuracy_score(y_true, y_pred),4)

            tqdm.write("[Epoch %d] Validation accuracy:%.4f. bacc:%.4f. Loss:%.3f" % (epoch, acc, balanced_acc, running_loss))
            tqdm.write("best_acc:" + str(best_acc))

            if acc > 0.89 and acc == best_acc:
                torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),},
                            os.path.join('checkpoints', "rafdb_epoch"+str(epoch)+"_acc"+str(acc)+"_bacc"+str(balanced_acc)+".pth"))
                tqdm.write('Model saved.')


if __name__ == "__main__":
    run_training()