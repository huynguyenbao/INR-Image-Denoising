import logging
import os
from abc import abstractmethod

import torch
from numpy import inf
from os.path import join as ospj
from src.logger import TensorboardWriter
from src.utils.utils import prepare_device

try:
    import wandb
except:
    pass
class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, config, log_dir):

        self.config = config # General config (model arch, optimizer, lr_scheduler, etc.)
        self.trainer_config = self.config['trainer_config'] # Training details such as num_epochs etc.

        # Setup a logger (just for cleaner log files)
        self._configure_logging(log_dir)

        # Read and save some of the configs
        self.epochs = self.trainer_config['epochs']
        self.save_period = self.trainer_config['save_period']
        self.eval_period = self.trainer_config['eval_period']

        # Configure how to monitor training and how to make checkpoints
        self._configure_monitoring()

        # Setup the checkpoint directory (where to save checkpoints)
        self.checkpoint_dir = ospj(
            self.trainer_config['save_dir'], self.config['name']
        )
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        else:
            print(f'Warning! Save dir {self.checkpoint_dir} already exists!'+\
                'Existing checkpoints will be overwritten!')
        

        # Setup visualization writer instance (Tensorboard, WandB)
        self.writer = None
        if self.trainer_config['tensorboard']:
            self.writer = TensorboardWriter(ospj(log_dir, 'tensorboard'), self.logger)
        self.wandb_enabled = self.trainer_config['wandb']

        # Prepare for (multi-device) GPU training
        # This part doesn't do anything if you don't have a GPU.
        self._device, self._device_ids = prepare_device(self.trainer_config['n_gpu'])
        
        self.start_epoch = 1
        self.best_epoch = 1
        self.current_epoch = 1

        self.counter = 0
        self.min_delta = 0

        self.eval_N = 0

        self.train_result_history = {}
        self.eval_result_history = {}

    def _configure_logging(self, log_dir):
        self.logger = logging.getLogger()
        self.logger.setLevel(1)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s : %(message)s')
        if not os.path.exists(ospj(log_dir)):
            os.mkdir(log_dir)
        _log_file = ospj(log_dir, self.config['name']+".log")
        if os.path.exists(_log_file):
            print(f'Warning! Log file {_log_file} already exists! The logs will be appended!')
        file_handler = logging.FileHandler(_log_file)
        file_handler.setFormatter(formatter)
        if (self.logger.hasHandlers()):
            self.logger.handlers.clear()
        self.logger.addHandler(file_handler)
    
    def _configure_monitoring(self):
        self.monitor = self.trainer_config.get('monitor', 'off')
        if self.monitor == 'off':
            self.monitor_mode = 'off'
        else:
            self.early_stop = self.trainer_config.get('early_stop', inf)
            if self.early_stop > 0:
               self.monitor_mode = 'on'
 
    @abstractmethod # To be implemented by the child classes!
    def _train_epoch(self):
        """
        Training logic for an epoch. Only takes care of doing a single training loop.

        :return: A dict that contains average loss and metric(s) information in this epoch.
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        self.logger.info("------------ New Training Session ------------")
        self.not_improved_count = 0        
        if self.wandb_enabled: wandb.watch(self.model, self.criterion, log='all')

        for epoch in range(self.start_epoch, self.epochs + 1):
            self.current_epoch = epoch
            train_result = self._train_epoch()
            # store the train_result to plot
            self.train_result_history[self.current_epoch] = train_result
                
            # save logged informations into log dict - ONLY CURRENT EPOCH
            log = {'epoch': self.current_epoch}
            log.update(train_result)

            if self.should_evaluate():
                eval_result = self.evaluate()
                # save eval information to the log dict as well
                log.update({f'eval_{key}': value for key, value in eval_result.items()})

                # store the eval_result to plot
                self.eval_result_history[self.current_epoch] = eval_result

            if self.monitor_mode != 'off' : # Then there is a metric to monitor

                if self.is_early_stopping():
                    print("X-X Early stopping at epoch: {}".format(self.current_epoch))
                    self.logger.info("X-X Early stopping at epoch: {}".format(self.current_epoch))
                    
                    eval_result = self.evaluate()
                    # save eval information to the log dict as well
                    log.update({f'eval_{key}': value for key, value in eval_result.items()})

                    # store the eval_result to plot
                    self.eval_result_history[self.current_epoch] = eval_result
                    
                    break

       
            # print logged information to the screen
            for key, value in log.items():
                self.logger.info(f'    {key[:15]}: {value:0.5f}')

            if self.wandb_enabled: wandb.log(log)

            if self.current_epoch % self.save_period == 0:
                # Just to regularly save the model every save_period epochs
                path = os.path.join(self.checkpoint_dir, f'E{self.current_epoch}_model.pth')
                self.save_model(path=path)

                STATE_PATH = os.path.join(self.checkpoint_dir, f'E{self.current_epoch}_state.pth')
                self.save_state(STATE_PATH)
            

        # Always save the last model
        path = os.path.join(self.checkpoint_dir, f'last_model.pth')
        self.save_model(path=path)
        STATE_PATH = os.path.join(self.checkpoint_dir, f'last_state.pth')
        self.save_state(STATE_PATH)
        
        return self.train_result_history, self.eval_result_history

    def should_evaluate(self):
        """
        Based on the self.current_epoch and self.eval_interval, determine if we should evaluate.
        You can take hint from saving logic implemented in BaseTrainer.train() method

        returns a Boolean
        """
        ###  TODO  ################################################
        # Based on the self.current_epoch and self.eval_interval, determine if we should evaluate.
        # You can take hint from saving logic implemented in BaseTrainer.train() method
        if self.current_epoch == 1:
            return True
        if self.current_epoch % self.eval_period == 0:
            # Just to regularly evaluate the model every eval_period epochs
            return True
            
        return False
        #########################################################
    
    @abstractmethod # To be implemented by the child classes!
    def evaluate(self, loader=None):
        """
        Evaluate the model on the val_loader given at initialization

        :param loader: A Dataloader to be used for evaluation. If not given, it will use the 
        self._eval_loader that's set during initialization..
        :return: A dict that contains metric(s) information for validation set
        """
        raise NotImplementedError
    
    def save_model(self, path=None):
        """
        Saves only the model parameters.
        : param path: path to save model (including filename.)
        """
        self.logger.info("Saving checkpoint: {} ...".format(path))
        torch.save(self.model.state_dict(), path)
        self.logger.info("Checkpoint saved.")
    
    def load_model(self, path=None):
        """
        Loads model params from the given path.
        : param path: path to save model (including filename.)
        """
        self.logger.info("Loading checkpoint: {} ...".format(path))
        self.model.load_state_dict(torch.load(path))
        self.logger.info("Checkpoint loaded.")
    
    @abstractmethod # To be implemented by the child classes!
    def is_early_stopping(self):
        """
        Checks early stopping from the history
        : param eval_result_history: history of evaluation results
        """
        raise NotImplementedError


    @abstractmethod # To be implemented by the child classes!
    def get_model_state(self):
        raise NotImplementedError

    @abstractmethod # To be implemented by the child classes!
    def set_model_state(self, checkpoint):
        raise NotImplementedError
    
    def save_state(self, PATH):
        print(f"Saving to: {PATH}")

        model_state = self.get_model_state()
        model_state['train_result_history'] = self.train_result_history
        model_state['eval_result_history'] = self.eval_result_history
        model_state['current_epoch'] = self.current_epoch
        model_state['eval_N'] = self.eval_N

        torch.save(model_state, PATH)

    def load_state(self, PATH):
        if PATH is None:
            return
        print(f"Loading from: {PATH}")
        checkpoint = torch.load(PATH)

        self.train_result_history = checkpoint['train_result_history']
        self.eval_result_history = checkpoint['eval_result_history']
        self.start_epoch = checkpoint['current_epoch']
        self.eval_N = checkpoint['eval_N']

        self.set_model_state(checkpoint)