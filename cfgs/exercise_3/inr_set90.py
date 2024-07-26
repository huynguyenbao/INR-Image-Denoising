from functools import partial

import torch
import torch.nn as nn

from src.data_loaders.single_image_dataset import SingleImageDataset
from src.data_loaders.pyramid_image_dataset import PyramidImageDataset
from src.trainers.baseline_trainer import BaselineTrainer
from src.trainers.boosting_trainer import BoostingTrainer
from src.trainers.earlystop_trainer import EarlystopTrainer
from src.trainers.sboost_trainer import SBoostTrainer
from src.trainers.pyramid_trainer import PyramidTrainer
from src.models.INR.siren import Siren
from src.models.INR.wire import Wire
from src.models.INR.finer import Finer

from src.models.INR.metric import PSNR

q1_experiment = dict(
    name = 'baseline_siren',

    model_arch = Siren,
    model_args = dict(
        in_features=2, 
        out_features=1, 
        hidden_features=256, 
        hidden_layers=3, 
        outermost_linear=True
    ),

    datamodule = SingleImageDataset,
    data_args = dict(
        image_path = "/content/drive/MyDrive/HLCV-Assignments/Project/data/Lenna.bmp", 
        noise_level = 25,
        RGB_mode = False,
        target_range=[-1,1],
    ),

    optimizer = partial(
        torch.optim.Adam,
        lr=1e-4
    ),
    lr_scheduler = partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),

    criterion = nn.MSELoss,
    criterion_args = dict(),

    metrics=dict(
        psnr = PSNR(1),
    ),

    trainer_module = BaselineTrainer,
    trainer_config = dict(
        n_gpu = 1,
        epochs = 1000,
        chunk = 256*256,
        eval_period = 50,
        save_dir = "./Saved/",
        save_period = 200,
        monitor = "off",
        early_stop = 0,
        log_step = 20,
        tensorboard=True,
        wandb=False,
    ),
)


q2_experiment = dict(
    name = 'earlystop_siren',

    model_arch = Siren,
    model_args = dict(
        in_features=2, 
        out_features=1, 
        hidden_features=256, 
        hidden_layers=3, 
        outermost_linear=True
    ),

    datamodule = SingleImageDataset,
    data_args = dict(
        image_path = "/content/drive/MyDrive/HLCV-Assignments/Project/data/Lenna.bmp", 
        noise_level = 25,
        RGB_mode = False,
        target_range=[-1,1],
    ),

    optimizer = partial(
        torch.optim.Adam,
        lr=1e-4
    ),
    lr_scheduler = partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),

    criterion = nn.MSELoss,
    criterion_args = dict(),

    metrics=dict(
        psnr = PSNR(1),
    ),

    trainer_module = EarlystopTrainer,
    trainer_config = dict(
        n_gpu = 1,
        epochs = 1000,
        chunk = 256*256,
        eval_period = 50,
        save_dir = "./Saved/",
        save_period = 200,
        monitor = "on",
        early_stop = 1,
        log_step = 20,
        tensorboard=True,
        wandb=False,
    ),
)


q3_experiment = dict(
    name = 'boosting_siren',

    model_arch = Siren,
    model_args = dict(
        in_features=2, 
        out_features=1, 
        hidden_features=256, 
        hidden_layers=3, 
        outermost_linear=True
    ),

    datamodule = SingleImageDataset,
    data_args = dict(
        image_path = "/content/drive/MyDrive/HLCV-Assignments/Project/data/Lenna.bmp", 
        noise_level = 25,
        RGB_mode = False,
        target_range=[-1,1],
    ),

    optimizer = partial(
        torch.optim.Adam,
        lr=1e-4
    ),
    # lr_scheduler = partial(
    #     torch.optim.lr_scheduler.StepLR,
    #     step_size=5, gamma=0.8
    # ),

    criterion = nn.MSELoss,
    criterion_args = dict(),

    metrics=dict(
        psnr = PSNR(1),
    ),

    trainer_module = BoostingTrainer,
    trainer_config = dict(
        n_gpu = 1,
        epochs = 1000,
        update_cycle = 400,
        chunk = 256*256,
        eval_period = 50,
        save_dir = "./Saved/",
        save_period = 100,
        monitor = "off",
        early_stop = 0,
        log_step = 100,
        tensorboard=True,
        wandb=False,
    ),
)


q4_experiment = dict(
    name = 'sboosting_siren',

    model_arch = Siren,
    model_args = dict(
        in_features=2, 
        out_features=1, 
        hidden_features=256, 
        hidden_layers=3, 
        outermost_linear=True
    ),

    datamodule = SingleImageDataset,
    data_args = dict(
        image_path = "/content/drive/MyDrive/HLCV-Assignments/Project/data/Lenna.bmp", 
        noise_level = 25,
        RGB_mode = False,
        pyramid_levels = 4,
        target_range = [-1,1],
    ),

    optimizer = partial(
        torch.optim.Adam,
        lr=1e-4
    ),
    # lr_scheduler = partial(
    #     torch.optim.lr_scheduler.StepLR,
    #     step_size=5, gamma=0.8
    # ),

    criterion = nn.MSELoss,
    criterion_args = dict(),

    metrics=dict(
        psnr = PSNR(1),
    ),

    trainer_module = SBoostTrainer,
    trainer_config = dict(
        n_gpu = 1,
        epochs = 1000,
        update_cycle = 100,
        chunk = 256*256,
        eval_period = 50,
        save_dir = "./Saved/",
        save_period = 100,
        monitor = "off",
        early_stop = 0,
        log_step = 100,
        tensorboard=True,
        wandb=False,
    ),
)

q5_experiment = dict(
    name = 'baseline_wire',

    model_arch = Wire,
    model_args = dict(
        in_features=2, 
        out_features=1, 
        hidden_features=256, 
        hidden_layers=3, 
        outermost_linear=True
    ),

    datamodule = SingleImageDataset,
    data_args = dict(
        image_path = "/content/drive/MyDrive/HLCV-Assignments/Project/data/Lenna.bmp", 
        noise_level = 25,
        RGB_mode = False,
        target_range=[-1,1],
    ),

    optimizer = partial(
        torch.optim.Adam,
        lr=5e-3
    ),
    lr_scheduler = partial(
        torch.optim.lr_scheduler.LambdaLR,
        step_size=5, gamma=0.8
    ),

    criterion = nn.MSELoss,
    criterion_args = dict(),

    metrics=dict(
        psnr = PSNR(1),
    ),

    trainer_module = BaselineTrainer,
    trainer_config = dict(
        n_gpu = 1,
        epochs = 10000,
        chunk = 256*256,
        eval_period = 50,
        save_dir = "./Saved/",
        save_period = 200,
        monitor = "off",
        early_stop = 0,
        log_step = 20,
        tensorboard=True,
        wandb=False,
    ),
)
