import torch
import torch.distributed as dist
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
from utils import is_master,get_logger


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config,train_sampler=None):
        self.config = config
        if dist.is_initialized():
            logger_name="{}{}".format(__name__,dist.get_rank())
        else:
            logger_name=__name__
        self.logger = get_logger(name=logger_name, log_dir=config.log_dir, verbosity=config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.train_sampler=train_sampler

        # configuration to monitor model performance and save best
        if not is_master() or self.monitor == 'off':
            self.mnt_mode = 'off'
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        if is_master():
            self.writer = TensorboardWriter(config, cfg_trainer['tensorboard'])
        else:
            self.writer = TensorboardWriter(config, False)

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            if dist.is_initialized():
                self.train_sampler.set_epoch(epoch-1)
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1
                    self.logger.info("Monitor metric did\'t improve for epoch#: {}".format(epoch))

                if not_improved_count > self.early_stop:
                    self.logger.info("Monitor metric didn\'t improve for {} epochs Training stops.".format(self.early_stop))
                    if dist.is_initialized():
                        dist.destroy_process_group()
                    exit(0)
            if is_master() and (epoch % self.save_period == 0 or best):
                state=self._generate_model_state(epoch)
                if epoch % self.save_period == 0:
                    self._save_checkpoint(epoch, state)
                if best:
                    self._save_best(state)
            if dist.is_initialized():
                self.logger.debug("Barrier after saving")
                dist.barrier()
            

    def _save_best(self, state):
        """
        Saves state as 'model_best.pth'
        :param state: state to be saved as 'model_best.pth'
        """
        best_path = str(self.checkpoint_dir / 'model_best.pth')
        torch.save(state, best_path)
        self.logger.info("Saving current best: model_best.pth ...")

    def _save_checkpoint(self, epoch, state):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param state: state to be saved as checkpoint
        """
        filename = str(self.checkpoint_dir /
                       'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _generate_model_state(self, epoch):
        """
        Returns dict contaning model state

        :param epoch: current epoch number
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        return state
