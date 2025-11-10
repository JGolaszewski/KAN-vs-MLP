import numpy as np
from abc import ABC, abstractmethod
from mlp_architecture import MLP
import random
from typing import Callable

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm

from loguru._logger import Logger
import pandas as pd
import time

def create_reg_data(
        func:Callable,
        x_range:tuple = (-10,10),
        ammount:int = 200,
        split:float = 0.8,
        batch_size:int = 64,
        noise_factor:float = 0.1,
        standardize:bool = True,
        torch_dtype:torch.dtype = torch.float32,
        seed = None,
        ) -> tuple[DataLoader, DataLoader, StandardScaler]:
    
    # Seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Create regression data in range for specified 1D function
    X = np.linspace(x_range[0], x_range[1], ammount).reshape(-1,1)
    y = func(X) + np.random.normal(0,noise_factor,X.shape)

    # Split data info train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=seed)

    # Standardize if needed
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Convert to torch DataLoaders
    train_data = TensorDataset(
        torch.tensor(X_train, dtype=torch_dtype),
        torch.tensor(y_train, dtype=torch_dtype)
    )
    test_data = TensorDataset(
        torch.tensor(X_test, dtype=torch_dtype),
        torch.tensor(y_test, dtype=torch_dtype)
    )

    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_dl, test_dl, scaler

class EvolutionAlgorithm(ABC):
    @abstractmethod
    def initialize(self) -> list[np.ndarray]:
        pass

    @abstractmethod
    def selection(self, genomes: list[np.ndarray], scores) -> list[np.ndarray]:
        pass

    @abstractmethod
    def mutate(self, genomes: list[np.ndarray]) -> list[np.ndarray]:
        pass

class GridEvo(EvolutionAlgorithm):
    def __init__(self, number):
        self.number = number

    def initialize(self):
        genomes = [np.array([1])]

        steps = [
            lambda g: np.append(g[:-1], g[-1] + 1),
            lambda g: np.append(g, 1),
        ]

        n = self.number
        extend = 1
        i = 0
        while True:
            new = []
            for s in steps:
                for g in genomes[-extend:]:
                    new.append(s(g))
                    i+=1
                    if i == n-1:
                        break
                if i == n-1:
                    break

            genomes.extend(new)
            extend = len(new)
            if i == n-1:
                break
            
        return genomes
    
    def mutate(self, genomes):
        def increase_random_layer(g: np.ndarray) -> np.ndarray:
            if g.size == 0:
                return g


            g_copy = g.copy()
            idx_to_modify = random.randint(0, g.size - 1)
            amount_to_add = random.randint(1, 5)
            
            g_copy[idx_to_modify] += amount_to_add
            
            return g_copy

        def decrease_random_layer(g: np.ndarray) -> np.ndarray:
            if g.size == 0:
                return g

            g_copy = g.copy()
            idx_to_modify = random.randint(0, g.size - 1)
            amount_to_subtract = random.randint(1, 5)
            
            g_copy[idx_to_modify] = max(1, g_copy[idx_to_modify] - amount_to_subtract)
            
            return g_copy

        steps = [
            lambda g: np.append(g[:-1], g[-1] + 1),
            lambda g: np.append(g, 1),
            lambda g: np.append(g, 1),

            # Delete random layer
            lambda g: np.delete(g, np.random.choice(len(g))) if len(g) > 1 else g,

            # Duplicate random layer
            lambda g: np.insert(g, (idx := np.random.choice(len(g))), g[idx]) if len(g) > 0 else g,

            decrease_random_layer,
            increase_random_layer
        ]
        original_genomes = genomes.copy()

        while len(genomes) < self.number:
            genomes.append(
                random.choice(steps)(random.choice(original_genomes))
            )

        return genomes
    
    # We select best 50% of genomes
    def selection(self, genomes, scores):
        scores_arr = np.array(scores)
        
        sorted_indices = np.argsort(scores_arr)
        
        num_to_select = len(genomes) // 2        
        best_indices = sorted_indices[:num_to_select]
        
        best_genomes = [genomes[i] for i in best_indices]
        
        return best_genomes

class TrainingHandler:
    def __init__(
            self, 
            sim_in_size,
            sim_out_size,
            test_data: DataLoader,
            train_data: DataLoader,
            criterion: torch.nn.Module,
            optimizer_class: torch.optim.Optimizer
            ):
        
        # In-Out params
        self.sim_in_size = sim_in_size
        self.sim_out_size = sim_out_size

        # Data
        self.train_d = train_data
        self.test_d = test_data

        # Training and Testing
        self.criterion = criterion
        self.opt_c = optimizer_class

    def train(
            self,
            genome: np.ndarray,
            num_epochs: int,
            ):

        mlp = MLP(
            input_size =    self.sim_in_size,
            output_size =   self.sim_out_size,
            hidden_sizes =  genome,
        )

        optimizer = self.opt_c(mlp.parameters(), lr=0.001)
        train_losses = []
        times = []

        for epoch in range(num_epochs):
            mlp.train()
            running_loss = 0.0

            start_time = time.perf_counter()
            for inputs, targets in self.train_d:
                optimizer.zero_grad()
                
                outputs =  mlp(inputs)
                loss = self.criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(self.train_d.dataset)
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
            train_losses.append(epoch_loss)
        
        self.last = mlp
        return train_losses, times

    def score_last(self):
        if self.last is None:
            raise TypeError('No last model trained!')

        total_size = 0
        total_error = 0
        with torch.inference_mode():
            start_time = time.perf_counter()
            for data, targets in self.test_d:
                pred = self.last(data)
                loss = self.criterion(pred, targets)
                
                total_error += loss.item() * data.size(0)
                total_size += targets.size(0)

            end_time = time.perf_counter()
    
        return total_error / total_size, end_time - start_time

class Simulation:
    def __init__(
            self, 
            evolution_algoritm: EvolutionAlgorithm,
            training_handler: TrainingHandler,
            logger: Logger = None
            ):

        self.evolution_algoritm = evolution_algoritm
        self.training_handler = training_handler

    def run(self, epoches = 30):
        genomes = self.evolution_algoritm.initialize()
        
        sim_data = pd.DataFrame(columns=[
            'genome',

            # Training data
            'avg_train_loss',
            'std_train_loss',
            'max_train_loss',
            'min_train_loss',
            'avg_train_time',
            'std_train_time',

            # Testing data
            'test_loss',
            'inference_time'
        ])

        for i in range(epoches):
            train_losses = []
            mse = []
            
            best = {'genome': None, 'score': np.inf}

            pbar = tqdm(genomes, desc=f"Simulation {i+1} Training")
            for genome in pbar:

                last_train_losses, train_times = self.training_handler.train(
                    genome=genome,
                    num_epochs=50,
                )
                
                last_loss, inference_time = self.training_handler.score_last()
                pbar.set_postfix(last_loss=f"{last_loss:.4f}")
                
                if last_loss < best['score']:
                    best['genome'] = genome
                    best['score'] = last_loss

                train_losses.append(last_train_losses)
                mse.append(last_loss)

                ltl_np = np.array(last_train_losses)
                tt_np = np.array(train_times)

                sim_data.loc[len(sim_data)] = [
                    "_".join(genome.astype(str)),
                    ltl_np.mean(),
                    ltl_np.std(),
                    ltl_np.max(),
                    ltl_np.min(),
                    tt_np.mean(),
                    tt_np.std(),
                    last_loss,
                    inference_time
                ]

            print(best)

            selected = self.evolution_algoritm.selection(genomes, mse)
            genomes = self.evolution_algoritm.mutate(selected)

        return sim_data
        
        