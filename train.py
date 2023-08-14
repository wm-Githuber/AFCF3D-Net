import datetime
import numpy
import os
import copy
import time
import json
import argparse
import torch
import cv2
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from sklearn.metrics import precision_recall_fscore_support as prfs
# from utils.data import get_loader
from utils.func import AvgMeter, clip_gradient
from utils.lr_scheduler import get_scheduler
from utils.dataset import MyDataset
from utils.helper import set_metrics, get_mean_metrics, initialize_metrics
from utils.losses import get_criterion
from model.metric_tool import ConfuseMatrixMeter, AverageMeter, get_confuse_matrix
from model.cd3d import CD3D_Net
# from model.RD3D_ame import Net_arch
from model.Conv3D import Net_arch
from utils.loss_f import BCEDICE_loss
import xlsxwriter
import warnings
from thop import profile
warnings.filterwarnings("ignore")


def parse_option():
    parser = argparse.ArgumentParser()
    # data set
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--trainsize', type=int, default=256)
    parser.add_argument('--hflip', action='store_true', help='hflip data')
    parser.add_argument('--vflip', action='store_true', help='vflip data')
    parser.add_argument('--data_dir', type=str, default='E:\\AllData\\LEVERCD\\ABLabel')
    # parser.add_argument('--data_dir', type=str, default='E:\\AllData\\WHU\\ABLabel')
    # parser.add_argument('--data_dir', type=str, default='E:\\AllData\\SYSU-CD\\ABLable')
    # training
    parser.add_argument('--model', type=str, default='RD3D+', help='RD3D or RD3D+')
    parser.add_argument('--epochs', type=int, default=100, help='epoch number')
    parser.add_argument('--optim', type=str, default='adamW', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')       # ori_lr: 0.0001
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['step', 'cosine'])
    parser.add_argument('--warmup_epoch', type=int, default=-1, help='warmup epoch')
    parser.add_argument('--warmup_multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr_decay_epochs', type=int, default=[120, 160, 200], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr_decay_steps', type=int, default=20,
                        help='for step scheduler. step size to decay lr')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    # io
    parser.add_argument('--output_dir', type=str, default='./output', help='output director')

    opt, unparsed = parser.parse_known_args()
    opt.output_dir = os.path.join(opt.output_dir, str(int(time.time())))
    # opt.output_dir = os.path.join(opt.output_dir, f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')
    return opt

def build_loader(opt):
    train_data = MyDataset(opt.data_dir, "train")
    train_loader = DataLoader(train_data, batch_size=opt.batchsize, shuffle=True, num_workers=2, pin_memory=True)
    val_data = MyDataset(opt.data_dir, "val")
    val_loader = DataLoader(val_data, batch_size=opt.batchsize, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader

def build_model(opt):
    resnet = torchvision.models.resnet18(pretrained=True)
    # model = CD3D_Net(32, copy.deepcopy(resnet))
    model = Net_arch(32, copy.deepcopy(resnet))
    model = model.cuda()
    return model

def main(opt):
    train_loader, val_loader = build_loader(opt)
    n_data = len(train_loader.dataset)
    print(f"length of training dataset: {n_data}\n")
    print(f"length of val dataset: {len(val_loader.dataset)}\n")
    # build model
    model = build_model(opt)

    # print number of parameters
    parameters_tot = 0
    for nom, param in model.named_parameters():
        parameters_tot += torch.prod(torch.tensor(param.data.shape))
    print("Number of model parameters {}".format(parameters_tot))

    # 定义写出行列号
    train_hook = xlsxwriter.Workbook('WHU_0-001_train.xlsx')
    train_record = train_hook.add_worksheet()
    train_record.write('A1', 'epoch')
    train_record.write('B1', 'Pre')
    train_record.write('C1', 'Recall')
    train_record.write('D1', 'F1')
    train_record.write('E1', 'IoU')
    train_record.write('F1', 'acc')
    train_record.write('G1', 'loss')
    val_hook = xlsxwriter.Workbook('WHU_0-001_val.xlsx')
    val_record = val_hook.add_worksheet()
    val_record.write('A1', 'epoch')
    val_record.write('B1', 'Pre')
    val_record.write('C1', 'Recall')
    val_record.write('D1', 'F1')
    val_record.write('E1', 'IoU')
    val_record.write('F1', 'acc')
    val_record.write('G1', 'loss')
    row = 1
    col = 0

    # CE = torch.nn.BCEWithLogitsLoss().cuda()
    # CE = torch.nn.BCELoss()
    CE = BCEDICE_loss

    # build optimizer
    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), opt.lr, weight_decay=opt.weight_decay)
    elif opt.optim == 'adamW':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=opt.weight_decay)
    elif opt.optim == 'sdg':
        optimizer = torch.optim.SGD(model.parameters(), opt.lr / 10.0 * opt.batchsize, momentum=opt.momentum,
                                    weight_decay=opt.weight_decay)
    else:
        raise NotImplementedError
    scheduler = get_scheduler(optimizer, len(train_loader), opt)
    # routine
    for epoch in range(1, opt.epochs + 1):
        tic = time.time()
        tool4metric = ConfuseMatrixMeter(n_class=2)
        train(train_loader, model, optimizer, CE, scheduler, epoch, tool4metric, train_record, row, col)
        print('epoch {}, total time {:.2f}, learning_rate {}'.format(epoch, (time.time() - tic),
                                                                     optimizer.param_groups[0]['lr']))
        print('begin val')
        val(val_loader, model, CE, epoch, tool4metric, val_record, row, col)
        print('epoch {}, total time {:.2f}'.format(epoch, (time.time() - tic)))
        if epoch >= 30:
            torch.save(model.state_dict(), os.path.join(opt.output_dir, f"RD3D_{epoch}_ckpt.pth"))
            print("model saved {}!".format(os.path.join(opt.output_dir, f"RD3D_{epoch}_ckpt.pth")))
        row = row + 1

    torch.save(model.state_dict(), os.path.join(opt.output_dir, f"RD3D_last_ckpt.pth"))
    print("model saved {}!".format(os.path.join(opt.output_dir, f"RD3D_ckpt.pth")))
    train_hook.close()
    val_hook.close()
    os.path.join(opt.output_dir, f"RD3D_last_ckpt.pth")


def train(train_loader, model, optimizer, criterion, scheduler, epoch, tool4metric, train_record, row, col):
    tool4metric.clear()
    model.train()
#    train_metrics = initialize_metrics()
    loss_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        imageA, imageB, gts = pack
        imageA = imageA.cuda().float()
        imageB = imageB.cuda().float()
        gts = gts.cuda().float()

        imageA = imageA.unsqueeze(2)
        imageB = imageB.unsqueeze(2)
        images = torch.cat([imageA, imageB], 2)

        # flops, params = profile(model, (imageA, imageB,))
        # print('flops:', flops, 'params:', params)
        # print('flops: %.2f M, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

        # forward
        pred_s = model(images)
        # pred_s = pred_s.squeeze(1)
        gts = torch.unsqueeze(gts, dim=1)
        loss = criterion(pred_s, gts)
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        scheduler.step()

        loss_record.update(loss.data, opt.batchsize)
        bin_preds_mask = (pred_s.to('cpu') > 0.5).detach().numpy().astype(int)
        mask = gts.to('cpu').numpy().astype(int)

        # out_png = bin_preds_mask.squeeze(0)
        # out_mask = mask.squeeze(0)
        # cv2.imwrite('./pred.png', out_png)
        # cv2.imwrite('./mask.png', out_mask)

        tool4metric.update_cm(pr=bin_preds_mask, gt=mask)

        # gts_temp = gts.data.cpu().numpy().flatten()
        # prs_temp = bin_preds_mask.reshape(bin_preds_mask.shape[1] * bin_preds_mask.shape[2] * bin_preds_mask.shape[0])
        # cd_train_report = prfs(prs_temp, gts_temp, average='binary', pos_label=1)
        # train_metrics = set_metrics(train_metrics, cd_train_report,)
        # mean_train_metrics = get_mean_metrics(train_metrics)

        if i % 100 == 0 or i == len(train_loader):
            print('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}],'
                  'Loss: {:.4f}'.format(epoch, opt.epochs, i, len(train_loader), loss_record.show()))

    scores_dictionary = tool4metric.get_scores()
    # print("IoU for epoch {} is {}".format(epoch, scores_dictionary["iou"]))
    # print("F1 for epoch {} is {}".format(epoch, scores_dictionary["F1"]))
    # print("acc for epoch {} is {}".format(epoch, scores_dictionary["acc"]))
    # print("precision for epoch {} is {}".format(epoch, scores_dictionary["precision"]))
    # print("recall for epoch {} is {}".format(epoch, scores_dictionary["recall"]))
    print('---------------------------------------------')
    # print('pre of prfs is {}'.format(mean_train_metrics['cd_precisions']))
    # print('recall of prfs is {}'.format(mean_train_metrics['cd_recalls']))
    # print('F1 of prfs is {}'.format(mean_train_metrics['cd_f1scores']))
    # print("EPOCH {} TRAIN METRICS".format(epoch) + str(mean_train_metrics))
    train_record.write(row, col, epoch)
    train_record.write(row, col + 1, scores_dictionary['precision'])
    train_record.write(row, col + 2, scores_dictionary['recall'])
    train_record.write(row, col + 3, scores_dictionary['F1'])
    train_record.write(row, col + 4, scores_dictionary['iou'])
    train_record.write(row, col + 5, scores_dictionary['acc'])
    train_record.write(row, col + 6, loss_record.show())

def val(val_loader, model, criterion, epoch, tool4metric, val_record, row, col):
    model.eval()
    tool4metric.clear()
    loss_record = AvgMeter()
    # val_metrics = initialize_metrics()
    with torch.no_grad():
        for i, pack in enumerate(val_loader):
            imageA, imageB, gts = pack
            imageA = imageA.cuda().float()
            imageB = imageB.cuda().float()
            gts = gts.cuda().float()

            imageA = imageA.unsqueeze(2)
            imageB = imageB.unsqueeze(2)
            images = torch.cat([imageA, imageB], 2)

            pred_s = model(images)
            # pred_s = pred_s.squeeze(1)
            gts = torch.unsqueeze(gts, dim=1)

            loss1 = criterion(pred_s, gts)
            loss = loss1

            bin_preds_mask = (pred_s.to('cpu') > 0.5).detach().numpy().astype(int)
            mask = gts.to('cpu').numpy().astype(int)
            tool4metric.update_cm(pr=bin_preds_mask, gt=mask)
            loss_record.update(loss.data, opt.batchsize)

            # gts_temp = gts.data.cpu().numpy().flatten()
            # prs_temp = bin_preds_mask.reshape(
            #     bin_preds_mask.shape[1] * bin_preds_mask.shape[2] * bin_preds_mask.shape[0])
            # cd_train_report = prfs(prs_temp, gts_temp, average='binary', pos_label=1)
            # val_metrics = set_metrics(val_metrics, cd_train_report, )
            # mean_val_metrics = get_mean_metrics(val_metrics)

            if i % 100 == 0 or i == len(val_loader):
                print('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}],'
                      'Loss: {:.4f}'.format(epoch, opt.epochs, i, len(val_loader), loss_record.show()))

        scores_dictionary = tool4metric.get_scores()
        print("IoU for epoch {} is {}".format(epoch, scores_dictionary["iou"]))
        print("F1 for epoch {} is {}".format(epoch, scores_dictionary["F1"]))
        print("acc for epoch {} is {}".format(epoch, scores_dictionary["acc"]))
        print("precision for epoch {} is {}".format(epoch, scores_dictionary["precision"]))
        print("recall for epoch {} is {}".format(epoch, scores_dictionary["recall"]))
        print('---------------------------------------------')
        # print('pre of prfs is {}'.format(mean_val_metrics['cd_precisions']))
        # print('recall of prfs is {}'.format(mean_val_metrics['cd_recalls']))
        # print('F1 of prfs is {}'.format(mean_val_metrics['cd_f1scores']))
        val_record.write(row, col, epoch)
        val_record.write(row, col + 1, scores_dictionary['precision'])
        val_record.write(row, col + 2, scores_dictionary['recall'])
        val_record.write(row, col + 3, scores_dictionary['F1'])
        val_record.write(row, col + 4, scores_dictionary['iou'])
        val_record.write(row, col + 5, scores_dictionary['acc'])
        val_record.write(row, col + 6, loss_record.show())


if __name__ == '__main__':
    opt = parse_option()
    os.makedirs(opt.output_dir, exist_ok=True)
    path = os.path.join(opt.output_dir, 'config.json')
    with open(path, 'w') as f:
        json.dump(vars(opt), f, indent=2)
    print("\n full config save to {}".format(path))

    main(opt)




















