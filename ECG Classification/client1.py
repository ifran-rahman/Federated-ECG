import os
import sys
import timeit
from collections import OrderedDict
from typing import Dict, List, Tuple
from cen_train import *
import flwr as fl
import numpy as np
import torch
import torchvision
import cen_train
# import cifar

USE_FEDBN: bool = True

# pylint: disable=no-member
DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member

# Flower Client
class CustomClient(fl.client.NumPyClient):
    """Flower client implementing ECG classification using
    PyTorch."""

    def __init__(
        self,
        model: ecg_net,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        num_examples: Dict,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_examples = num_examples

    def get_parameters(self, config) -> List[np.ndarray]:
        print("Entered get_parameters")
        self.model.train()  
        if USE_FEDBN:
            # Return model parameters as a list of NumPy ndarrays, excluding parameters of BN layers when using FedBN
            return [
                val.cpu().numpy()
                for name, val in self.model.state_dict().items()
                if "bn" not in name
            ]
        else:
            # Return model parameters as a list of NumPy ndarrays
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        
        # Set model parameters from a list of NumPy ndarrays
        print("Entered set_parameters")
        self.model.train()
        if USE_FEDBN:
            keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=False)
        else:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        
        print("Entered Fit Method")
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        # cen_train.train(self.model, self.trainloader, epochs=1, device=DEVICE)

        train_client(model= self.model, train_loader = self.train_loader, valid_loader=self.test_loader, epochs=2)

        return self.get_parameters(config=config), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        # loss, accuracy = cen_train.validate(self.model, self.testloader, device=DEVICE)
        criterion = nn.CrossEntropyLoss()
        loss, accuracy = cen_train.validate(self.model, self.test_loader, criterion)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}

def main() -> None:
    # load dataset
    abnormal = pd.read_csv('datasets/ptbdb_abnormal.csv', header = None)
    normal = pd.read_csv('datasets/ptbdb_normal.csv', header = None)

    train_dataset, val_dataset, num_examples = prepare__dataset(abnormal=abnormal, normal=normal, val_split_factor=val_split_factor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    print("Num example", num_examples)

    model = ecg_net(2).to(device=device)

    # Perform a single forward pass to properly initialize BatchNorm
    _ = model(next(iter(train_loader))[0].to(DEVICE))

    # Start client
    print("Starting Client 1")
    client = CustomClient(model, train_loader, val_loader, num_examples)
    fl.client.start_numpy_client(server_address="127.0.0.1:8081", client=client)

if __name__ == "__main__":
    main()
