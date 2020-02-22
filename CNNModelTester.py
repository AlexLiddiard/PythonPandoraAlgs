import torch
import torch.nn as nn
import os
import glob
from torch.autograd import Variable
import BaseConfig as bc
import argparse
import torchvision.models as models
import copy
from mvcnn_pytorch.tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from mvcnn_pytorch.models.MVCNN import MVCNN, MVCNN2, SVCNN
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="ElectronPhoton")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=1)# it will be *3 images in each batch for mvcnn
parser.add_argument("-num_models", type=int, help="number of models per class", default=0)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.001)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="vgg19")
parser.add_argument("-num_views", type=int, help="number of views", default=3)
parser.add_argument("-train_path", type=str, default="../PythonPandoraAlgs/ElectronPhoton/SVGData/*/train")
parser.add_argument("-val_path", type=str, default="../PythonPandoraAlgs/ElectronPhoton/SVGData/*/test")
args = parser.parse_args()
pretraining = not args.no_pretraining

model1 = SVCNN("test1", nclasses=2, pretraining=pretraining, cnn_name=args.cnn_name)
model1.load(bc.analysisFolderFull + "/CNNModel")

model2 = MVCNN2("test2", model1, nclasses=2, cnn_name=args.cnn_name, num_views=args.num_views).cuda()
model2.load(bc.analysisFolderFull + "/CNNModel")

loss_fn = nn.CrossEntropyLoss()

#val_dataset = SingleImgDataset(args.val_path, scale_aug=False, rot_aug=False, test_mode=True)
val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, num_views=args.num_views)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize*3, shuffle=False, num_workers=0)

all_correct_points = 0
all_points = 0
wrong_class = np.zeros(2)
samples_class = np.zeros(2)
all_loss = 0

avgpool = nn.AvgPool1d(1, 1)

total_time = 0.0
total_print_time = 0.0
all_target = []
all_pred = []

for _, data in enumerate(val_loader, 0):
    print("Validating! (%s)" % (_+1))
    N,V,C,H,W = data[1].size()
    in_data = Variable(data[1]).view(-1,C,H,W).cuda()

    target = Variable(data[0]).cuda()

    out_data = model2(in_data)
    print(out_data)
    pred = torch.max(out_data, 1)[1]
    print(pred)
    print(target)
    all_loss += loss_fn(out_data, target).cpu().data.numpy()
    results = pred == target
    print(results)

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








