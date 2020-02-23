import BaseConfig as bc
import GeneralConfig as gc
import CNNConfig as cfg
import DataSampler as ds
import torch
import torch.nn as nn
import os
import glob
from torch.autograd import Variable
import argparse
import torchvision.models as models
import copy
from CNNTools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from CNNModels.MVCNN import MVCNN, MVCNN2, SVCNN
import numpy as np

model2 = MVCNN2("stage2", SVCNN("stage1", cnn_name=cfg.testingBaseModel), cnn_name=cfg.testingBaseModel)
model2.load(cfg.modelOutputFolder + "/" + cfg.testingModelName)
model2.cuda()
model2.eval()
loss_fn = nn.CrossEntropyLoss()

val_dataset = MultiviewImgDataset(cfg.testingImagePath + "/*/test", classNames=gc.classNames, scale_aug=False, rot_aug=False, num_views=3)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.testingBatchSize, shuffle=False, num_workers=0)

ds.LoadPfoData([]) # Load general info dataframe

all_correct_points = 0
all_points = 0
wrong_class = np.zeros(2)
samples_class = np.zeros(2)
all_loss = 0

for _, data in enumerate(val_loader, 0):
    print("Validating! (%s)" % (_+1))
    N,V,C,H,W = data[1].size()
    in_data = Variable(data[1]).view(-1,C,H,W).cuda()

    target = Variable(data[0]).cuda()

    out_data = model2(in_data)
    pred = torch.max(out_data, 1)[1]
    all_loss += loss_fn(out_data, target).cpu().data.numpy()
    results = pred == target

    for i in range(results.size()[0]):
        if not bool(results[i].cpu().data.numpy()):
            wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
        samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
    correct_points = torch.sum(results.long())

    all_correct_points += correct_points
    all_points += results.size()[0]

print ('Total # of test models: ', all_points)
val_mean_class_acc = np.mean((samples_class-wrong_class)/samples_class)
acc = all_correct_points.float() / all_points
val_overall_acc = acc.cpu().data.numpy()
loss = all_loss / len(val_loader)

print ('val mean class acc. : ', val_mean_class_acc)
print ('val overall acc. : ', val_overall_acc)
print ('val loss : ', loss)
