import numpy as np
import typer
import sim_util
from loguru import logger

import torch
from pathlib import Path 
from datetime import datetime

app = typer.Typer()

try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass


@app.command()
def run(save_path: str):
    sp = Path(save_path)
    
    if not sp.exists() or not sp.is_dir():
        raise typer.BadParameter(f'Path {sp} does not exist or is not a directory path !')


    train_data, test_data, _ = sim_util.create_reg_data(
        lambda x: np.sin(x),
        seed = 42
    )


    evolution_logic = sim_util.GridEvo(10)
    training_handler = sim_util.TrainingHandler(
        sim_in_size = 1,
        sim_out_size = 1,
        train_data = train_data,
        test_data = test_data,
        criterion = torch.nn.L1Loss(),
        optimizer_class = torch.optim.Adam
    )

    sim = sim_util.Simulation(
        evolution_logic,
        training_handler
    )

    data = sim.run(epoches = 30)

    timestamp = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")

    data.to_csv(sp / f'sim_run_{timestamp}')


if __name__ == "__main__":
    app()