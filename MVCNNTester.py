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
import pandas as pd

model2 = MVCNN2("stage2", SVCNN("stage1", cnn_name=cfg.testingBaseModel), cnn_name=cfg.testingBaseModel)
model2.load(cfg.modelOutputFolder + "/" + cfg.testingModelName)
model2.cuda()
model2.eval()
loss_fn = nn.CrossEntropyLoss()

val_dataset = MultiviewImgDataset(cfg.testingImagePath + "/*/test", classNames=gc.classNames, scale_aug=False, rot_aug=False, num_views=3)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.testingBatchSize, shuffle=False, num_workers=0)

ds.LoadPfoData([]) # Load general info dataframe

dfPredictions = pd.DataFrame({"MVCNN": np.repeat(np.nan, len(ds.dfInputPfoData))})
dfPredictions.set_index(ds.dfInputPfoData['fileName'] + "_" + ds.dfInputPfoData['eventId'].astype(str) + "_" + ds.dfInputPfoData['pfoId'].astype(str), inplace = True)

all_correct_points = 0
all_points = 0
wrong_class = np.zeros(2)
samples_class = np.zeros(2)
all_loss = 0

for _, data in enumerate(val_loader, 0):
    print("Validating! (%s)" % (_+1))
    N,V,C,H,W = data[1].size()
    in_data = Variable(data[1]).view(-1,C,H,W).cuda()
    out_data = model2(in_data)
    pred = torch.max(out_data, 1)[1]
    for i, filepath in zip(range(pred.size()[0]), data[2][0]):
        index = os.path.basename(filepath[:-9])
        dfPredictions.loc[index, "MVCNN"] = int(pred[i].cpu().data.numpy())

ds.SavePfoData(dfPredictions, "CNN")
