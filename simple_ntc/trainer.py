from copy import deepcopy
from os import stat
from typing import Callable

import numpy as np
import torch

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage, metric
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from simple_ntc.utils import get_grad_norm, get_paramter_norm

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

class MyEngine(Engine):

    def __init__(self, func: Callable, model, crit, optimizer, config):
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.config = config 

        super().__init__(func)

        self.best_loss = np.inf
        self.best_model = None

        # gpu settings
        self.device = next(model.parameters()).device
    
    @staticmethod
    def train(engine, mini_batch: int):
        engine.model.train()
        engine.optimizer.zero_grad()

        # step1: Feed Forward
        x, y = mini_batch.text, mini_batch.label
        x, y = x.to(engine.device), y.to(engine.device)

        x = x[:, :engine.config.max_length]

        y_hat = engine.model(x)

        # step2: Calculate Loss
        loss = engine.crit(y_hat, y)
        # step2: Back-propagation
        loss.backward()

        if isinstance(y, torch.LongTensor) or isinstance(y, torch.LongTensor):
            accuracy = (torch.argmax(y_hat, dim=-1).sum() / float(y.size(0)))
        else:
            accuracy = 0
        
        p_norm = float(get_paramter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        # step3: Gradient descent
        engine.optimizer.step()
        
        # step4: Return current state
        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            '|param|': p_norm,
            '|g_param|': g_norm
        }
    
    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            x, y = mini_batch.text, mini_batch.label
            x, y = x.to(engine.device), y.to(engine.device)

            x = x[:, :engine.config.max_length]

            y_hat = engine.model(x)

            loss = engine.crit(y_hat, y)

            if isinstance(y, torch.LongTensor) or isinstance(y_hat, torch.LongTensor):
                accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
            else:
                accuracy = 0
            
        return {
            'loss': float(loss),
            'accuarcy': float(accuracy)
        }
    
    @staticmethod
    def attach(train_engine, validation_engine, verbose=VERBOSE_BATCH_WISE):
        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name
            )
        
        training_metric_names = ['loss', 'accuracy', '|params|', '|g_param|']

        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)
        
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names)
        
        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                print('Validation - loss={:.4e} accuracy={:.4f} best_loss={:.4e}'.format(
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                    engine.best_loss,
                ))
    
    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss:
            engine.best_loss = loss
            engine.best_model = deepcopy(engine.model.state_dict())
        
    @staticmethod
    def save_model(engine, train_engine, config, **kwargs):
        torch.save(
            {
                'model': engine.best_model,
                'config': config,
                **kwargs
            }, config.model_fn
        )

class Trainer():

    def __init__(self, config) -> None:
        self.config = config

    def train(
        self,
        model,
        crit,
        optimizer,
        train_loader, valid_loader
    ):
        train_engine = MyEngine(
            MyEngine.train,
            model, crit, optimizer, self.config
        )
        validation_engine = MyEngine(
            MyEngine.validate,
            model, crit, optimizer, self.config
        )

        MyEngine.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )

        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epoch=1)
        
        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            run_validation,
            validation_engine, valid_loader
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            MyEngine.check_best
        )

        train_engine.run(
            train_loader,
            max_epochs=self.config.n_epochs
        )

        model.load_state_dict(validation_engine.best_model)

        return model
