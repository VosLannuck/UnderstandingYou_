import torch
import argparse

import ModelBuilder
import DataPreps

from Trainer import Trainer
from Enums import ModelName

from torch.nn import Module
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torchsummary import summary
from torchvision import models
from torchvision.models.swin_transformer import SwinTransformer
from torch.nn import Linear
from omegaconf import OmegaConf, DictConfig, ListConfig
from typing import Union, List

config: Union[DictConfig, ListConfig] = OmegaConf.load("params.yaml")
device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
supportedModels: List[str] = [
    config.constant.vanilla_cnn,
    config.constant.vanilla_alex,
    config.constant.vanilla_vgg16,
    config.constant.vanilla_resnet
]


arg: argparse.ArgumentParser = argparse.ArgumentParser()
arg.add_argument("-m", "--model", type=str,
                 default=config.constant.vanilla_cnn,
                 choices=supportedModels)
arg.add_argument("-b", "--batch", type=int,
                 default=config.constant.batch_size)
arg.add_argument("-e", "--epoch", type=int,
                 default=config.constant.epoch)
arg.add_argument("-lr", "--learning_rate", type=int,
                 default=config.constant.lr)
arg.add_argument("-trp", "--training_path", type=str,
                 default=config.data.training_path)
arg.add_argument("-vp", "--valid_path", type=str,
                 default=config.data.validation_path)
arg.add_argument("-tsp", "--test_path", type=str,
                 default=config.data.testing_path)
arg.add_argument("-nc", "--num_classes", type=int,
                 default=config.constant.num_classes)
arg.add_argument("-img", "--image_size", type=int,
                 default=config.constant.img_size)
args = vars(arg.parse_args())


num_classes: int = args['num_classes']
modelName: ModelName = ModelBuilder.parseToModelName(config, args["model"])
model: Module = ModelBuilder.preserveModel(modelName, device, num_classes)

summary(model, (3, config.constant.img_size, config.constant.img_size))
trainDataLoader, validDataLoader, testDataLoader = DataPreps.makeDataset(args["training_path"],
                                                                         args["valid_path"],
                                                                         args["test_path"],
                                                                         args["num_classes"],
                                                                         args["image_size"],
                                                                         args["batch"],
                                                                         )
trainer: Trainer = Trainer(
    trainDataLoader,
    validDataLoader,
    testDataLoader,
)
cls = CrossEntropyLoss()
for _, _, _, _ in trainer.TrainModel(model, cls, 
                                     AdamW(params=model.parameters(),
                                     lr=config.constant.lr), 
                                     model_name=modelName.__str__()):
    ...
