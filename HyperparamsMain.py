import torch
import os
import numpy as np

import ModelBuilder
import DataPreps
from datetime import datetime

from ray import train, tune
from ray.train._checkpoint import Checkpoint
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
    config.constant.vanilla_resnet
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
varargs = vars(args.parse_args())

device: str = "cuda" if (torch.cuda.is_available() and varargs["device"] != "cpu") else "cpu"
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
    loaded_checkpoint: Checkpoint = train.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(loaded_checkpoint_dir,
                                                      "checkpoint.pt")
            model.load_state_dict(optimizer_state)

    for _, _, val_acc, val_loss in trainer.TrainModel(model, criterion,
                                                      optimizer,
                                                      model_name=modelName.__str__(),
                                                      is_hyperparams=True):
        os.makedirs(checkpoint_path_model, exist_ok=True)
        modelSaveName: str = "checkpoint.pt"
        torch.save((model.state_dict(), optimizer.state_dict()),
                   os.path.join(checkpoint_path_model,
                                modelSaveName))
        checkpoint: Checkpoint = Checkpoint.from_directory(checkpoint_path_model)
        reportResult(val_acc, val_loss, checkpoint)


def reportResult(val_acc: float,
                 val_loss: float, checkpoint: Checkpoint):
    repDict: Dict = {"loss": val_loss,
                     "acc": val_acc}
    train.report(repDict, checkpoint=checkpoint)


def run_hyperopts():
    search_space = {"lr": tune.loguniform(1e-4, 1e-2)}
    algo = OptunaSearch()

    tuner = tune.Tuner(
        objective_optimizer,
        tune_config=tune.TuneConfig(
            num_samples=1,
            metric="loss",
            mode="min",
            search_alg=algo,
        ),
        param_space=search_space
    )
    results = tuner.fit()
    print("Best Configuration: ", results)


run_hyperopts()
