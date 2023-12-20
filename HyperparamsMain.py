import torch
import os
import numpy as np

import ModelBuilder
import DataPreps
from datetime import datetime

from ray import train, tune
from ray.train._checkpoint import Checkpoint
from ray.train import ScalingConfig
from argparse import ArgumentParser
from typing import List, Union
from omegaconf import OmegaConf, DictConfig, ListConfig
from torch.nn import Module
from ray.tune.search.optuna import OptunaSearch
from typing import Dict

from DatasetMaker import MainSmokerDataset
from Trainer import Trainer
from Enums import ModelName

args: ArgumentParser = ArgumentParser(prog="UnderstandingYou",
                                      description="Finding optimum params")
config: Union[DictConfig, ListConfig] = OmegaConf.load("hyperparams.yaml")
supportedModels: List[str] = [
    config.constant.vanilla_cnn,
    config.constant.vanilla_alex,
    config.constant.vanilla_vgg16,
    config.constant.vanilla_resnet,
    config.constant.pretrained_alex,
    config.constant.pretrained_vgg16,
    config.constant.pretrained_resnet,
]

supportedDevices: List[str] = [
    config.cmd.device,
    "cuda"
]
args.add_argument('-m', '--model',
                  help="Model name", choices=supportedModels,
                  default=config.constant.vanilla_alex,
                  )
args.add_argument("-trp", "--training_path",
                  default=config.cmd.train_path, )
args.add_argument("-vp", "--valid_path",
                  default=config.cmd.valid_path, )
args.add_argument("-tsp", "--test_path",
                  default=config.cmd.test_path, )
args.add_argument("-nc", "--num_classes",
                  default=config.cmd.nc)
args.add_argument("-b", "--batch_size",
                  default=config.cmd.batch)
args.add_argument("-img", "--image_size",
                  default=config.cmd.img_size)
args.add_argument("-d", "--device",
                  default=config.cmd.device, choices=supportedDevices)
args.add_argument("-cp", "--checkpoint_path",
                  default=config.cmd.checkpoint_path)
args.add_argument("-ns", "--num_samples",
                  default=config.cmd.num_samples)
args.add_argument("-wks", "--workers",
                  default=config.cmd.num_workers)
args.add_argument("-gpus","--gpus",
                  default=config.cmd.num_gpus)

varargs = vars(args.parse_args())

device: str = "cuda" if (torch.cuda.is_available() and varargs["device"] != "cpu") else "cpu"
use_gpu: bool = False
if device == "cuda":
    use_gpu = True
abs_path: str = os.path.abspath(os.curdir)
train_path: str = os.path.join(abs_path, varargs["training_path"])
valid_path: str = os.path.join(abs_path, varargs["valid_path"])
test_path: str = os.path.join(abs_path, varargs["test_path"])
nc: int = varargs["num_classes"]
batch: int = varargs["batch_size"]
checkpoint_path: str = os.path.join(abs_path, varargs["checkpoint_path"])
img_size: int = varargs["image_size"]

os.makedirs(checkpoint_path, exist_ok=True)

modelName: ModelName = ModelBuilder.parseToModelName(config, varargs["model"])


def objective_optimizer(config):
    model: Module = ModelBuilder.preserveModel(modelName, device, nc)
    currTime: str = str(datetime.now().microsecond)
    checkpoint_path_model: str = os.path.join(checkpoint_path,
                                              str(modelName.name) + currTime)
    trainLoader, validLoader, testLoader = DataPreps.makeDataset(train_path,
                                                                 valid_path,
                                                                 test_path,
                                                                 nc,
                                                                 img_size,
                                                                 batch
                                                                 )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    trainer: Trainer = Trainer(trainLoader,
                               validLoader,
                               testLoader)
    for _, _, val_acc, val_loss in trainer.TrainModel(model, criterion,
                                                      optimizer,
                                                      model_name=modelName.__str__(),
                                                      is_hyperparams=True, epoch=10):
        reportResult(val_acc, val_loss)


def reportResult(val_acc: float,
                 val_loss: float, checkpoint: Checkpoint = None):
    repDict: Dict = {"loss": val_loss,
                     "acc": val_acc}
    train.report(repDict)


def run_hyperopts():
    search_space = {"lr": tune.loguniform(1e-4, 1e-2)}
    algo = OptunaSearch()

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(objective_optimizer),
            resources= {"cpu": varargs["workers"], "gpu": varargs["gpus"]}

        ),
        tune_config=tune.TuneConfig(
            num_samples=varargs["num_samples"],
            metric="loss",
            mode="min",
            search_alg=algo,
        ),
        param_space=search_space
    )
    results = tuner.fit()
    print("Best Configuration: ", results.get_best_result().config)


run_hyperopts()
