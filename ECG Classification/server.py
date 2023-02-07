from typing import Dict, Optional, Tuple, List, Union
from collections import OrderedDict
import numpy as np
import flwr as fl
from cen_train import *
import cen_train
from flwr.common import (
    Scalar,
)

model = ecg_net(2).to(device=device)

_, _, test_loader, _ = my_DataLoader('datasets/ptbdb_abnormal.csv',
                                    'datasets/ptbdb_normal.csv',
                                    batch_size=batch_size,
                                    val_split_factor=0.2)


class SaveModelStrategy(fl.server.strategy.FedAvg):

        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
        ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Scalar]]:

            # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

            if aggregated_parameters is not None:
                
                # Save aggregated_ndarrays
                print(f"Saving round {server_round} aggregated_ndarrays...")

                # Convert `Parameters` to `List[np.ndarray]`
                aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

                np.savez(f"round-{server_round}-weights.npz", *aggregated_ndarrays)

                # Convert `List[np.ndarray]` to PyTorch`state_dict`
                params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                model.load_state_dict(state_dict, strict=True)

                # Save the model
                torch.save(model.state_dict(), f"model_round_{server_round}.pth")

            return aggregated_parameters, aggregated_metrics

def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        criterion = nn.CrossEntropyLoss()
        loss, accuracy = 0,0 #cen_train.validate(model, test_loader, criterion)
        return loss, {"accuracy": accuracy}

        # return evaluate

def fit_config(server_round: int):
        """Return training configuration dict for each round.
        Keep batch size fixed at 32, perform two rounds of training with one
        local epoch, increase to two local epochs afterwards.
        """
        config = {
            "batch_size": 16,
            "local_epochs": 1 if server_round < 2 else 2,
        }
        return config

def main():
  
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        evaluate_fn=evaluate,
        on_fit_config_fn=fit_config,
    )
    
    fl.server.start_server(server_address="https://5682-119-148-3-101.in.ngrok.io ", config=fl.server.ServerConfig(num_rounds=3), strategy=strategy)
  

if __name__ == "__main__":
    main()                              