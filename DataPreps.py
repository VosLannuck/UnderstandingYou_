
from DatasetMaker import MainSmokerDataset
from torch.utils.data import DataLoader
from typing import Tuple


def makeDataset(training_path: str,
                validation_path: str,
                testing_path: str,
                num_classes: int,
                img_size: int,
                batch_size: int,
                columns=["img_path", "class_name", "label"]
                ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    mainSmokerDatasetMaker: MainSmokerDataset = MainSmokerDataset(
        trainData=training_path,
        validationData=validation_path,
        testingData=testing_path,
        numClasses=num_classes,
        columns_df=columns,
    )

    mainSmokerDatasetMaker.makeDataFrame(training_path)
    mainSmokerDatasetMaker.makeDataFrame(validation_path)
    mainSmokerDatasetMaker.makeDataFrame(testing_path)
    mainSmokerDatasetMaker.makeDataset(img_size)
    mainSmokerDatasetMaker.makeDataLoader(batch_size)

    trainDataLoader: DataLoader = mainSmokerDatasetMaker.trainDataLoader
    validDataLoader: DataLoader = mainSmokerDatasetMaker.validationDataLoader
    testDataLoader: DataLoader = mainSmokerDatasetMaker.testDataLoader
    return trainDataLoader, validDataLoader, testDataLoader
