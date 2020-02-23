import torch
import torch.nn as nn
import os
import glob


class Model(nn.Module):

    def __init__(self, name):
        super(Model, self).__init__()
        self.name = name

    def save(self, path, epoch=0):
        torch.save(self.state_dict(), 
            os.path.join(path, "{}_epoch{}.pth".format(self.name, epoch)))

    def save_results(self, path, data):
        raise NotImplementedError("Model subclass must implement this method.")
        

    def load(self, path, epoch=None):
        if epoch is None:
            model_files = glob.glob(os.path.join(path, "{}_epoch*.pth".format(self.name)))
            mf = max(model_files)
        else:
            mf = os.path.join(path, "{}_epoch{}.pth".format(self.name, epoch))

        self.load_state_dict(torch.load(mf))


