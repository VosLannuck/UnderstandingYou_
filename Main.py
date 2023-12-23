import torch
import argparse

import ModelBuilder
import DataPreps

from Trainer import Trainer
from Enums import ModelName

from torch.nn import Module
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, lr_scheduler
from torchsummary import summary
from omegaconf import OmegaConf, DictConfig, ListConfig
from typing import Union, List
from distutils.util import strtobool

torch.manual_seed(0)
config: Union[DictConfig, ListConfig] = OmegaConf.load("params.yaml")
device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
supportedModels: List[str] = [
    config.constant.vanilla_cnn,
    config.constant.vanilla_alex,
    config.constant.vanilla_vgg16,
    config.constant.vanilla_resnet,
    config.constant.pre_vgg16,
    config.constant.pre_alex,
    config.constant.pre_resnet,
    config.constant.vit
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
arg.add_argument("-const", "--constant", action="store_true",
                 default=True, help="Pass TRUE or FALSE / T or F")
args = vars(arg.parse_args())


num_classes: int = args['num_classes']
modelName: ModelName = ModelBuilder.parseToModelName(config, args["model"])
model: Module = ModelBuilder.preserveModel(modelName, device, num_classes)
isConstant: bool = strtobool(args["constant"])
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
milestones: List[int] = [config.constant.lr_mile._1,
                         config.constant.lr_mile._2,
                         config.constant.lr_mile._3,
                         config.constant.lr_mile._4,
                         config.constant.lr_mile._5,
                         config.constant.lr_mile._6,
                         ]


def getMultistepScheduler(config: Union[DictConfig, ListConfig],
                          model: Module,
                          modelName: ModelName,
                          milestones: List[int] = milestones,
                          gamma=0.0001):
    if (modelName == ModelName.vanilla_alex):
        optimizer: torch.optim = AdamW(params=model.parameters(),
                                       lr=config.best_v_alex)
        scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer,
                                             milestones=milestones,
                                             gamma=gamma)
        return optimizer, scheduler
    elif (modelName == ModelName.pretrained_alex):
        optimizer: torch.optim = AdamW(params=model.parameters(),
                                       lr=config.best_p_alex)
        scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer,
                                             milestones=milestones,
                                             gamma=gamma)
        return optimizer, scheduler
    elif (modelName == ModelName.vanilla_vgg16):
        optimizer: torch.optim = AdamW(params=model.parameters(),
                                       lr=config.best_v_vgg16)
        scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer,
                                             milestones=milestones,
                                             gamma=gamma)
        return optimizer, scheduler
    elif (modelName == ModelName.pretrained_vgg):
        optimizer: torch.optim = AdamW(params=model.parameters(),
                                       lr=config.best_p_vgg16)
        scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer,
                                             milestones=milestones,
                                             gamma=gamma)
        return optimizer, scheduler
    elif (modelName == ModelName.vanilla_resnet):
        optimizer: torch.optim = AdamW(params=model.parameters(),
                                       lr=config.best_v_resnet)
        scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer,
                                             milestones=milestones,
                                             gamma=gamma)
        return optimizer, scheduler
    elif (modelName == ModelName.pretrained_resnet):
        optimizer: torch.optim = AdamW(params=model.parameters(),
                                       lr=config.best_p_resnet)
        scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer,
                                             milestones=milestones,
                                             gamma=0.001)
        return optimizer, scheduler
    elif (modelName == ModelName.vanilla_cnn):
        optimizer: torch.optim = AdamW(params=model.parameters(),
                                       lr=config.best_v_cnn)
        scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer,
                                             milestones=milestones,
                                             gamma=gamma)
        return optimizer, scheduler
    elif (modelName == ModelName.vit):
        optimizer: torch.optim = AdamW(params=model.parameters(),
                                       lr=config.best_p_vit)
        scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer,
                                             milestones=milestones,
                                             gamma=0.001)
        return optimizer, scheduler
    else:
        raise ("Wrong model name")


if (not isConstant):

    optimizer, multi_step_scheduler = getMultistepScheduler(config, model,
                                                            modelName)
    for _, _, _, _ in trainer.TrainModelHyperparamsTuner(model, cls,
                                                         optimizer,
                                                         multi_step_scheduler,
                                                         epoch=config.constant.epoch,
                                                         model_name=modelName.name):
        ...

else:
    optimizer = AdamW(params=model.parameters(),
                      lr=config.constant.lr)
    for _, _, _, _ in trainer.TrainModel(model, cls, optimizer,
                                         epoch=config.constant.epoch,
                                         model_name=modelName.name):
        ...
