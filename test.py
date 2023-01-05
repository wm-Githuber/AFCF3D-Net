import argparse
import copy
import os
import warnings

import cv2
import torch
import torchvision
import tqdm
from torch.utils.data import DataLoader

from model.Networks import Netmodel
from model.metric_tool import ConfuseMatrixMeter
# from data import test_dataset
from utils.dataset import MyDataset
from utils.loss_f import BCEDICE_loss

warnings.warn('ignore')

# base_path = "PNG_result/LEVIR/"
# imglist = os.listdir(base_path)


# data_path = './output/WHU_BCE_DICE/WHU.pth'    # the WHU  record done!
# data_path = 'output/LEVIR_BCE_DICE/LEVIR.pth'    # the LEVIR record done!
# data_path = 'output/SYSU_BCE_DICE/SYSU.pth'    # the SYSU record done!

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=data_path, help='path to model file')
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--test_datasets', type=str, default=['NJU2000-test'], nargs='+', help='test dataset')
# parser.add_argument('--data_path', type=str, default='E:\\AllData\\SYSU-CD\\ABLable', help='test dataset')
# parser.add_argument('--data_path', type=str, default='E:\\AllData\\LEVERCD\\ABLabel', help='test dataset')
parser.add_argument('--data_path', type=str, default='E:\\AllData\\WHU\\ABLabel')
parser.add_argument('--save_path', type=str, help='test dataset')

# model
parser.add_argument('--multi_load', action='store_true', help='whether to load multi-gpu weight')
opt = parser.parse_args()

test_data = MyDataset(opt.data_path, "test")
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

if opt.save_path is not None:
    save_root = opt.save_path
else:
    mode_dir_name = os.path.dirname(opt.model_path)
    stime = mode_dir_name.split('\\')[-1]
    save_root = os.path.join(mode_dir_name, f'{stime}_results')

# build model
resnet = torchvision.models.resnet50(pretrained=True)
model = Netmodel(32, copy.deepcopy(resnet))

if opt.multi_load:
    state_dict_multi = torch.load(opt.model_path)
    state_dict = {k[7:]: v for k, v in state_dict_multi.items()}
else:
    state_dict = torch.load(opt.model_path)
model.load_state_dict(state_dict)
model.cuda()
model.eval()
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# loop to evaluate the model and print the metrics
bce_loss = 0.0
# criterion = torch.nn.BCELoss()
criterion = BCEDICE_loss
tool_metric = ConfuseMatrixMeter(n_class=2)

i = 0
c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}

save_path = './PNG_result/WHU/'
# textfile = 'E:\\AllData\\LEVERCD\\ABLabel\\list\\test.txt'
textfile = 'E:\\AllData\\WHU\\ABLabel\\list\\test.txt'
# textfile = 'E:\\AllData\\SYSU-CD\\ABLable\\list\\test.txt'
namelines = []
with open(textfile, 'r', encoding='utf-8') as file:
    for c in file.readlines():
        namelines.append(c.strip('\n').split(' ')[0])

with torch.no_grad():
    for reference, testimg, mask in tqdm.tqdm(test_loader):
        reference = reference.to(device).float()
        testimg = testimg.to(device).float()
        mask = mask.float()

        # pass refence and test in the model
        imageA = reference.unsqueeze(2)
        imageB = testimg.unsqueeze(2)
        images = torch.cat([imageA, imageB], 2)
        # generated_mask = model(images)
        generated_mask = model(images)
        generated_mask = generated_mask.squeeze(1)

        # compute the loss for the batch and backpropagate
        generated_mask = generated_mask.to("cpu")
        bce_loss += criterion(generated_mask, mask)

        ### Update the metric tool
        bin_genmask = (generated_mask > 0.5).numpy()
        bin_genmask = bin_genmask.astype(int)
        out_png = bin_genmask.squeeze(0)

        # savename = save_path + namelines[i]
        # cv2.imwrite(savename, out_png)

        i = i + 1

        mask = mask.numpy()
        mask = mask.astype(int)
        tool_metric.update_cm(pr=bin_genmask, gt=mask)

    bce_loss /= len(test_loader)
    print("Test summary")
    print("Loss is {}".format(bce_loss))
    print()

    scores_dictionary = tool_metric.get_scores()
    print(scores_dictionary)
