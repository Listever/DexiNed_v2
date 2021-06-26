import torch

import yaml
from models.networks import get_nets
from models.losses import get_loss

class Trainer:
    def __init__(self,config):
        self.config = config

    def train(self):
        self._init_params(self)
        #print(config)

    @staticmethod
    def _init_params(self):
        self.criterionG = get_loss(self.config['model'])
        self.netG = get_nets(self.config['model'])
        self.netG().cuda()

        print(self.netG,self.criterionG)


if __name__ == '__main__':
    # Load config
    print(f"Number of GPU's available: {torch.cuda.device_count()}")
    print(f"Pytorch version: {torch.__version__}")
    with open('config/config.yaml','r') as f:
        config = yaml.load(f, Loader = yaml.FullLoader)

    trainer = Trainer(config)
    trainer.train()
