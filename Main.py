import torch
import argparse

from CNN_testing import CNN_Testing
from Baseline_CNN import CNN_Baseline
from Baseline_Resnet import BaselineResnet, ResidualBlock
from Baseline_VGG import VGG_Baseline_16
from Baseline_AlexNet import AlexNet
from DatasetMaker import MainSmokerDataset
from Trainer import Trainer

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torchsummary import summary
from torchvision import models
from torchvision.models.swin_transformer import SwinTransformer
from torch.nn import Linear

INIT_LR: float = 1e-3
BATCH_SIZE: int = 64
EPOCH_SIZE: int = 10
IMG_SIZE: int = 224
IMG_SIZE_TESTING: int = 64

TRAIN_SPLIT: float = 0.75
VAL_SPLIT: float = 1 - TRAIN_SPLIT
NUM_CLASSES: int = 2

TRAINING_PATH: str = "./Dataset/Training/"
VALIDATION_PATH: str = "./Dataset/Validation/"
TESTING_PATH: str = "./Dataset/Testing/"

VANILLA_CNN: str = "vanilla_cnn"
VANILLA_VGG16: str = "vanilla_VGG16"
VANILLA_RESNET: str = "vanilla_resnet"
VANILLA_ALEX: str = "vanilla_alex"
VIT: str = "vit_swin"
device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

arg: argparse.ArgumentParser = argparse.ArgumentParser()
arg.add_argument("-b", "--batch", type=int, default=BATCH_SIZE)
arg.add_argument("-e", "--epoch", type=int, default=EPOCH_SIZE)
arg.add_argument("-lr", "--learning_rate", type=int, default=INIT_LR)
arg.add_argument("-trp", "--training_path", type=str, default=TRAINING_PATH)
arg.add_argument("-vp", "--valid_path", type=str, default=VALIDATION_PATH)
arg.add_argument("-tsp", "--test_path", type=str, default=TESTING_PATH)
arg.add_argument("-nc", "--num_classes", type=str, default=2)
arg.add_argument("-img", "--image_size", type=int, default=IMG_SIZE)
args = vars(arg.parse_args())

cnnModel: CNN_Baseline = CNN_Baseline(3, classes=NUM_CLASSES).to(device)
alexNetModel: AlexNet = AlexNet(num_classes=NUM_CLASSES).to(device)
vggModel: VGG_Baseline_16 = VGG_Baseline_16(numClasses=NUM_CLASSES).to(device)
resnetModel: BaselineResnet = BaselineResnet(ResidualBlock,
                                             numClasses=NUM_CLASSES).to(device)
vit_model: SwinTransformer = models.swin_v2_s(weights='DEFAULT').to(device)
vit_model.head = Linear(in_features=vit_model.head.in_features, out_features=NUM_CLASSES).to(device)
cnn_testing: CNN_Testing = CNN_Testing(3, classes=NUM_CLASSES).to(device)

summary(vit_model, (3, IMG_SIZE_TESTING, IMG_SIZE_TESTING))
exit(1)

mainSmokerDatasetMaker: MainSmokerDataset = MainSmokerDataset(
    trainData=args['training_path'],
    validationData=args['valid_path'],
    testingData=args['test_path'],
    numClasses=args['num_classes'],
    columns_df=["img_path", "class_name", "label"],
)

mainSmokerDatasetMaker.makeDataFrame(args["training_path"])
mainSmokerDatasetMaker.makeDataFrame(args["valid_path"])
mainSmokerDatasetMaker.makeDataFrame(args["test_path"])
mainSmokerDatasetMaker.makeDataset()
mainSmokerDatasetMaker.makeDataLoader(batchSize=BATCH_SIZE)

trainDataLoader: DataLoader = mainSmokerDatasetMaker.trainDataLoader
validDataLoader: DataLoader = mainSmokerDatasetMaker.validationDataLoader
testDataLoader: DataLoader = mainSmokerDatasetMaker.testDataLoader

trainer: Trainer = Trainer(
    trainDataLoader,
    validDataLoader,
    testDataLoader,
)
cls = CrossEntropyLoss()
trainer.TrainModel(cnn_testing, cls,
                   AdamW(params=alexNetModel.parameters(),
                   lr=INIT_LR), model_name=VANILLA_CNN)
