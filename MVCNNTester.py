import BaseConfig as bc
import GeneralConfig as gc
import CNNConfig as cfg
import DataSampler as ds
import torch
import torch.nn as nn
import os
from torch.autograd import Variable
import torchvision.models as models
from CNNTools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from CNNModels.MVCNN import MVCNN2, SVCNN
import numpy as np
import pandas as pd

def TestCNN(val_loader, model, dfPredictions, name):
    nBatches = len(val_loader)
    for count, data in enumerate(val_loader, 0):
        print("Validating! %.1f%%" % (100 * count / nBatches))
        if name == "MVCNN":
            N,V,C,H,W = data[1].size()
            in_data = Variable(data[1]).view(-1,C,H,W).cuda()
        else:
            in_data = Variable(data[1]).cuda()
        out_data = model(in_data)
        pred = torch.max(out_data, 1)[1]
        for i, filepath in zip(range(pred.size()[0]), data[2][0]):
            index = os.path.basename(filepath[:-9])
            dfPredictions.loc[index, model.name] = int(pred[i].cpu().data.numpy())


ds.LoadPfoData([]) # Load general info dataframe
dfPredictions = pd.DataFrame({"MVCNN": np.repeat(np.nan, len(ds.dfInputPfoData))})
dfPredictions.set_index(ds.dfInputPfoData['fileName'] + "_" + ds.dfInputPfoData['eventId'].astype(str) + "_" + ds.dfInputPfoData['pfoId'].astype(str), inplace = True)

# Multi-View CNN
model2 = MVCNN2("MVCNN", SVCNN("SVCNN", cnn_name=cfg.testingBaseModel), cnn_name=cfg.testingBaseModel)
model2.load(cfg.modelOutputFolder + "/" + cfg.testingModelName)
#model2.cuda()
model2.eval()
val_dataset = MultiviewImgDataset(cfg.testingImagePath + "/*/test", classNames=gc.classNames, scale_aug=False, rot_aug=False, num_views=3)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.testingBatchSize, shuffle=False, num_workers=0)
TestCNN(val_loader, model2, dfPredictions, "MVCNN")

ds.SavePfoData(dfPredictions, "CNN")
