from typing import Any, Callable, Dict

import torch
import torch.nn.utils as torch_utils

from ignite.engine import Events

from simple_ntc.utils import get_grad_norm, get_parameter_norm
from solution.simple_ntc.bert_trainer import VERBOSE_EPOCH_WISE

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

from simple_ntc.trainer import Trainer, MyEngine


class EngineForBert(MyEngine):

    def __init__(self, func: Callable, model, crit, optimizer, scheduler, config):
        self.scheduler = scheduler

        super().__init__(func, model, crit, optimizer, config)
    
    @staticmethod
    def train(engine, mini_batch: Dict[str, Any]) -> Dict[str, float]:
        engine.model.train()
        engine.optimizer.zero_grad()

        x, y = mini_batch['input_ids'], mini_batch['labels']
        x, y = x.to(engine.device), y.to(engine.device)
        mask = mini_batch['attention_mask']
        mask = mask.to(engine.device)

        x = x[:, :engine.config.max_length]

        y_hat = engine.model(x, attention_mask=mask).logits

        loss = engine.crit(y_hat, y)
        loss.backward()

        if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
            accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
        else:
            accuracy = 0
        
        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        engine.optimizer.step()
        engine.scheduler.step()

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            '|param|': p_norm,
            '|g_param|': g_norm,
        }
    
    @staticmethod
    def validate(engine, mini_batch: Dict[str, Any]) -> Dict[str, float]:
        engine.model.eval()

        with torch.no_grad():
            x, y = mini_batch['input_ids'], mini_batch['labels']
            x, y = x.to(engine.device), y.to(engine.device)
            mask = mini_batch['attention_mask']
            mask = mask.to(engine.device)

            x = x[:, :engine.config.max_length]

            y_hat = engine.model(x, attention_mask=mask).logits

            loss = engine.crit(y_hat, y)

            if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
            else:
                accuracy = 0
        
        return {
            'loss': float(loss),
            'accuarcy': float(accuracy)
        }


class BertTrainer(Trainer):

    def __init__(self, config) -> None:
        self.config = config

    def train(
        self,
        model, crit, optimizer, scheulder,
        train_loader, valide_loader
    ):
        train_engine = EngineForBert(
            EngineForBert.train,
            model, crit, optimizer, scheulder, self.config
        )
        validation_engine = EngineForBert(
            EngineForBert.validate,
            model, crit, optimizer, scheulder, self.config
        )

        EngineForBert.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )

        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valide_loader, max_epochs=1)
        
        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            run_validation,
            validation_engine, valide_loader
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            EngineForBert.check_best
        )

        train_engine.run(
            train_loader,
            max_epochs=self.config._epochs
        )

        model.load_state_dict(validation_engine.best_model)

        return model