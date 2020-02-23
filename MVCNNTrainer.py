import BaseConfig as bc
import GeneralConfig as gc
import CNNConfig as cfg
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json

from CNNTools.Trainer import ModelNetTrainer
from CNNTools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from CNNModels.MVCNN import MVCNN, MVCNN2, SVCNN

def create_folder(log_dir, overwrite=True):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    elif overwrite:
        print('WARNING: Folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)

if __name__ == '__main__':
    create_folder(cfg.modelOutputFolder, False)
    log_dir = cfg.modelOutputFolder + "/" + cfg.trainingModelName
    create_folder(log_dir)

    # STAGE 1
    cnet = SVCNN('stage1', pretraining=cfg.pretraining, cnn_name=cfg.trainingBaseModel)

    optimizer = optim.Adam(cnet.parameters(), lr=cfg.learningRate, weight_decay=cfg.weightDecay)

    train_dataset = SingleImgDataset(cfg.trainingImagePath + "/*/train", classNames=gc.classNames, scale_aug=False, rot_aug=False, num_models=cfg.numPFOs*3, num_views=3)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.trainingBatchSize*3, shuffle=True, num_workers=0)

    val_dataset = SingleImgDataset(cfg.testingImagePath + "/*/test", classNames=gc.classNames, scale_aug=False, rot_aug=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.trainingBatchSize*3, shuffle=False, num_workers=0)
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))
    print("trainer getting set")
    trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'svcnn', log_dir, num_views=1)
    print("trainer starting")
    trainer.train(cfg.epochsStage1)
    print("trainer finished")

    # STAGE 2
    cnet_2 = MVCNN2('stage2', cnet, nclasses=2, cnn_name=cfg.trainingBaseModel, num_views=3)
    del cnet

    optimizer = optim.Adam(cnet_2.parameters(), lr=cfg.learningRate, weight_decay=cfg.weightDecay, betas=(0.9, 0.999))
    
    train_dataset = MultiviewImgDataset(cfg.trainingImagePath + "/*/train", classNames=gc.classNames, scale_aug=False, rot_aug=False, num_models=cfg.numPFOs*3, num_views=3)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.trainingBatchSize, shuffle=False, num_workers=0)# shuffle needs to be false! it's done within the trainer

    val_dataset = MultiviewImgDataset(cfg.testingImagePath + "/*/test", classNames=gc.classNames, scale_aug=False, rot_aug=False, num_views=3)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.trainingBatchSize, shuffle=False, num_workers=0)
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))
    trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'mvcnn', log_dir, num_views=3)
    trainer.train(cfg.epochsStage2)



