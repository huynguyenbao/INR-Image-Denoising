from functools import partial

import torch
import torch.nn as nn

from src.data_loaders.single_image_dataset import SingleImageDataset
from src.data_loaders.pyramid_image_dataset import PyramidImageDataset
from src.trainers.siren_trainer import SIRENTrainer
from src.trainers.boosting_siren_trainer import BOOSTING_SIRENTrainer
from src.trainers.earlystop_siren_trainer import EARLYSTOP_SIRENTrainer
from src.trainers.sboost_siren_trainer import SBOOST_SIRENTrainer
from src.trainers.pyramid_siren_trainer import PYRAMID_SIRENTrainer
from src.models.INR.siren import Siren
from src.models.INR.metric import PSNR

q1_experiment = dict(
    name = 'camera_man_siren',

    model_arch = Siren,
    model_args = dict(
        in_features=2, 
        out_features=1, 
        hidden_features=256, 
        hidden_layers=2, 
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

    trainer_module = SIRENTrainer,
    trainer_config = dict(
        n_gpu = 1,
        epochs = 700,
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
    name = 'camera_man_early_siren',

    model_arch = Siren,
    model_args = dict(
        in_features=2, 
        out_features=1, 
        hidden_features=256, 
        hidden_layers=2, 
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

    trainer_module = EARLYSTOP_SIRENTrainer,
    trainer_config = dict(
        n_gpu = 1,
        epochs = 700,
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
    name = 'camera_man_boosting_bsiren',

    model_arch = Siren,
    model_args = dict(
        in_features=2, 
        out_features=1, 
        hidden_features=256, 
        hidden_layers=2, 
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

    trainer_module = BOOSTING_SIRENTrainer,
    trainer_config = dict(
        n_gpu = 1,
        epochs = 700,
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
    name = 'camera_man_sboost_bsiren',

    model_arch = Siren,
    model_args = dict(
        in_features=2, 
        out_features=1, 
        hidden_features=256, 
        hidden_layers=2, 
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

    trainer_module = SBOOST_SIRENTrainer,
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
    name = 'camera_man_psboost_siren',

    model_arch = Siren,
    model_args = dict(
        in_features=2, 
        out_features=1, 
        hidden_features=256, 
        hidden_layers=2, 
        outermost_linear=True
    ),

    datamodule = PyramidImageDataset,
    data_args = dict(
        image_path = "/content/drive/MyDrive/HLCV-Assignments/Project/data/Lenna.bmp", 
        noise_level = 25,
        RGB_mode = False,
        pyramid_levels=4,
        target_range=[-1,1],
    ),

    optimizer = partial(
        torch.optim.Adam,
        lr=1e-4
    ),

    criterion = nn.MSELoss,
    criterion_args = dict(),

    metrics=dict(
        psnr = PSNR(1),
    ),

    trainer_module = PYRAMID_SIRENTrainer,
    trainer_config = dict(
        n_gpu = 1,
        epochs = 1000,
        chunk = 256*256,
        iters_per_level=100,
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
