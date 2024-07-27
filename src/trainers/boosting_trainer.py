import torch

from tqdm.auto import tqdm

from .base_trainer import BaseTrainer
from src.utils.utils import MetricTracker
from src.utils.img_utils import normalize, save_image
import numpy as np
import os

class BoostingTrainer(BaseTrainer):

    def __init__(self, config, log_dir, train_eval_set):
        """
        Create the model, loss criterion, optimizer, and dataloaders
        And anything else that might be needed during training. (e.g. device type)
        """
        super().__init__(config, log_dir)

        self.model = self.config['model_arch'](**self.config['model_args'])
        self.model.to(self._device)
        if len(self._device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self._device_ids)

        # Simply Log the model
        self.logger.info(self.model)

        # Prepare Losses
        self.criterion = self.config['criterion'](**self.config['criterion_args'])

        # Prepare Optimizer
        trainable_params = self.model.parameters()

        # Configure the optimizer and lr scheduler
        # These are usually Python Partial() objects that have all the options already inserted.
        self.optimizer = self.config['optimizer'](trainable_params)
        self.lr_scheduler = self.config.get('lr_scheduler', None) 

        # Set Dataset
        self.train_eval_set = train_eval_set
        self.chunk = self.config['trainer_config']['chunk']
        
        self.log_step = self.trainer_config['log_step']

        # Prepare Metrics
        self.metric_functions = self.config['metrics']

        self.train_metrics = MetricTracker(
            keys=['loss'] + [metric_key for metric_key in self.metric_functions.keys()],
            writer=self.writer)
        
        self.eval_N = 0
        self.eval_metrics = MetricTracker(
            keys=[metric_key for metric_key in self.metric_functions.keys()],
            writer=self.writer)

        # Baseline Result
        eval_results = {}
        self.image_shape = train_eval_set.image_shape
        if len(self.image_shape) == 2:
            self.image_shape = [1, 1, self.image_shape[0], self.image_shape[1]]
        elif len(self.image_shape) == 3:
            # TODO: WRONG
            self.image_shape = [1, self.image_shape[0], self.image_shape[1], self.image_shape[2]]
        else:
            raise Exception()
            exit()

        for metric_key, metric_func in self.metric_functions.items():
            result = metric_func.compute(
                torch.Tensor(train_eval_set.noisy_image).reshape(self.image_shape), 
                torch.Tensor(train_eval_set.clean_image).reshape(self.image_shape))
            self.eval_metrics.update(metric_key, result)

            eval_results[metric_key] = result

        for metric_key, result in eval_results.items():
            print(f"Noisy Image Quality Measurement: {metric_key}: {result}")
        
        # Additional Config
        self.update_cycle = self.config['trainer_config']['update_cycle']
        self.update_counter = 0
        self.psuedo_target = None

    def _train_epoch(self):
        """
        Training logic for an epoch. Only takes care of doing a single training loop.

        :return: A dict that contains average loss and metric(s) information in this epoch.
        """

        # Set model to train mode
        self.model.train()
        self.train_metrics.reset()

        self.logger.debug(f"==> Start Training Epoch {self.current_epoch}/{self.epochs}, lr={self.optimizer.param_groups[0]['lr']:.6f} ")

        image_res = self.train_eval_set.image_res
        chunk = self.chunk

        pbar = tqdm(total=image_res, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        indices = torch.randperm(image_res)
        
        coords = self.train_eval_set.coords.to(self._device)
        gt_noisy = self.train_eval_set.gt_noisy.to(self._device)
        reconstruct = torch.zeros_like(gt_noisy).to(self._device)

        if self.psuedo_target is None:
            self.psuedo_target = gt_noisy.clone()

        for batch_idx in range(0, image_res, chunk):

            batch_indices = indices[batch_idx:min(image_res, batch_idx+chunk)]

            batch_coords = coords[:, batch_indices, ...]
            batch_gt_noisy = self.psuedo_target[:, batch_indices, :]
            
            output = self.model.forward(batch_coords)
            reconstruct[:, batch_indices, :] = output

            loss = self.criterion(output, batch_gt_noisy)
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.writer is not None:
                self.writer.set_step((self.current_epoch - 1) * image_res + batch_idx)
            
            # Update all the train_metrics with new values.
            self.train_metrics.update('loss', loss.item())

            pbar.set_description(f"Train Epoch: {self.current_epoch} Loss: {loss.item():.4f}")

            pbar.update(chunk)

            del loss

        if self.lr_scheduler:
            self.lr_scheduler.step()

        # Update target to prevent overfitting
        self.update_target(gt_noisy, reconstruct)
        

        # Compute PNSR/ ... between reconstruction and noisy image
        output_range = self.config['data_args']['target_range']
        reconstruct = normalize(reconstruct, output_range, [0, 1]).reshape(self.image_shape)
        gt_noisy = normalize(gt_noisy, output_range, [0, 1]).reshape(self.image_shape)

        for metric_key, metric_func in self.metric_functions.items():
            result = metric_func.compute(reconstruct, gt_noisy)
            self.train_metrics.update(metric_key, result)

        log_dict = self.train_metrics.result()

        self.logger.debug(f"==> Finished Epoch {self.current_epoch}/{self.epochs}.")
        
        del reconstruct, coords, gt_noisy
        torch.cuda.empty_cache()

        return log_dict
    
    @torch.no_grad()
    def evaluate(self):
        """
        Evaluate the model on the val_loader given at initialization

        :param loader: A Dataloader to be used for evaluatation. If not given, it will use the 
        self._eval_loader that's set during initialization..
        :return: A dict that contains metric(s) information for validation set
        """
        
        self.model.eval()
        self.eval_metrics.reset()

        self.logger.debug(f"++> Evaluate at epoch {self.current_epoch} ...")

        image_res = self.train_eval_set.image_res
        chunk = self.chunk

        pbar = tqdm(total=image_res, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        indices = torch.randperm(image_res).to(self._device)
        
        coords = self.train_eval_set.coords.to(self._device)
        # gt_noisy = self.train_eval_set.gt_noisy.to(self._device)
        gt_clean = self.train_eval_set.gt_clean.to(self._device)
        reconstruct = torch.zeros_like(gt_clean).to(self._device)
        for batch_idx in range(0, image_res, chunk):

            batch_indices = indices[batch_idx:min(image_res, batch_idx+chunk)]
            batch_coords = coords[:, batch_indices, ...]

            reconstruct[:, batch_indices] = self.model(batch_coords)

            pbar.update(chunk)
        pbar.close()

        # Compute PNSR/ ... between reconstruction and clean image
        eval_results = {}
        output_range = self.config['data_args']['target_range']
        reconstruct = normalize(reconstruct, output_range, [0, 1]).reshape(self.image_shape)
        gt_clean = normalize(gt_clean, output_range, [0, 1]).reshape(self.image_shape)

        for metric_key, metric_func in self.metric_functions.items():
            result = metric_func.compute(reconstruct, gt_clean)
            self.eval_metrics.update(metric_key, result)

            eval_results[metric_key] = result
        # Save denoised image  
        denoised = reconstruct.cpu().detach().numpy()
        denoised = denoised.reshape(self.train_eval_set.image_shape)
        
        denoised = normalize(denoised, self.config['data_args']['target_range'], [0, 255])
        denoised = denoised.astype(np.int32)

        image_path = os.path.join(
            self.config['trainer_config']['save_dir'], 
            self.config['name'] ,
            'eval.png')
        
        save_image(image_path, denoised, self.config['data_args']['RGB_mode'])

        for metric_key, result in eval_results.items():
            print(f"Metric {metric_key}: {result}")

        log_dict = self.eval_metrics.result()

        self.logger.debug(f"++> Finished evaluating epoch {self.current_epoch}.")

        del reconstruct, coords, gt_clean
        torch.cuda.empty_cache()

        return log_dict
    
    def update_target(self, gt_noisy, reconstruction):
        self.update_counter += 1

        if self.update_counter >= self.update_cycle:
            print(f"Updating Target.")
            self.update_counter = 0
            self.psuedo_target = (reconstruction.detach() + gt_noisy) / 2
