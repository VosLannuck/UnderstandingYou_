#%%
import torch
import mlflow
import os
import numpy as np

from DatasetMaker import MainSmokerDataset, SmokerDataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from typing import List, Tuple, Dict

#%%
device : str = "cuda" if torch.cuda.is_available() else "cpu" 
class Trainer():
    def __init__(self,
                 trainDataLoader : DataLoader,
                 validationDataLoader : DataLoader,
                 testDataLoader : DataLoader,
                 ):
        self.trainDataLoader : DataLoader = trainDataLoader
        self.validationDataLoader : DataLoader = validationDataLoader
        self.testDataLoader : DataLoader = testDataLoader
        
        self.logs : Dict[str, List[float]] = {
            "train_loss" : [],
            "train_acc" : [],
            "val_loss" : [],
            "val_acc" : [],
        }       
        
    def train_step(self, model : torch.nn.Module, loss_fn : CrossEntropyLoss , optimizer : torch.optim.Optimizer) -> Tuple[torch.Tensor, torch.Tensor]:
        totalObservation : int = 0
        currentLoss : float = 0.0
        correct_prediction : int = 0
        
        numBatches : int = len(self.trainDataLoader)
        model.train()

        for data, target in self.trainDataLoader:
            target : torch.Tensor = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)
            
            # Clean previous calculations
            optimizer.zero_grad()
            
            # Forward prop
            outputs : torch.Tensor = self.model(data)
            
            loss : torch.Tensor = loss_fn(outputs, target)
            
            ## Back prop
            loss.backward()

            ## Optimzing model
            optimizer.step()

            currentLoss += loss.item()
            _, pred_indices = torch.max(outputs, dim=1) # [0 , 1]
            correct_prediction += torch.sum(pred_indices == target).item()
            totalObservation += target.shape[0]
        
        return correct_prediction / totalObservation, currentLoss / numBatches
            
            
    def test_or_eval_step(self, model : torch.nn.Module,loss_fn : CrossEntropyLoss, dataLoader : DataLoader ) -> Dict[torch.Tensor, torch.Tensor]:
        totalObservation : int = 0
        currentLossTotal : float = 0.0
        num_batches : int = len(dataLoader)
        correct_prediction : int = 0 
        with torch.no_grad():
            model.eval()
            
            for data, target in dataLoader:
                target : torch.Tensor = target.type(torch.LongTensor)
                data : torch.Tensor = data.to(device)
                target = target.to(device)
                
                outputs : torch.Tensor = model(data)
                
                loss : torch.Tensor = loss_fn(outputs, target)
                
                currentLossTotal += loss.item()
                
                ## Get the max from the outputs predictions, that is why dim is 1 not 0
                _, predIndices = torch.max(outputs,dim=1) 
                correct_prediction += torch.sum(target == predIndices).item() 
                totalObservation += 1
            
        return correct_prediction / totalObservation, currentLossTotal / num_batches
    
    def TrainModel(self, model : torch.nn.Module, loss_fn : CrossEntropyLoss, optimizer : torch.optim.Optimizer,
                   epoch : int = 5 , lr : float = 1e-4, checkpointPath : str = "checkpoints"):
        if(not os.path.isdir(checkpointPath)):
            os.mkdir(checkpointPath)
        
        model.to(device)
        best_loss : float = np.inf
        
        for epoch in tqdm(range(epoch)):
            train_acc, train_loss = self.train_step(model, loss_fn, optimizer)
            val_acc, val_loss = self.test_or_eval_step(model, loss_fn, self.validationDataLoader)
            print("Epoch : %g "% (epoch))
            print(f"TrainLoss : {train_loss:.4f}, TrainAcc : {train_acc:.4f}")
            print(f"ValLoss : {val_loss:.4f}, ValAcc : {val_loss:.4f}")

            self.logs["train_loss"].append(train_loss)
            self.logs["train_acc"].append(train_acc)
            
            self.logs["val_loss"].append(val_loss)
            self.logs["val_acc"].append(val_acc)
            
            torch.save(model.state_dict(), "checkpoints/last.pth")
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), "checkpoints/best.pth")
        print("Training done ")
