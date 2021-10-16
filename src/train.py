import os
from typing import List, Optional

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase
import pandas as pd

from src.utils import utils

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    try:
        # Set seed for random number generators in pytorch, numpy and python.random
        if "seed" in config:
            seed_everything(config.seed, workers=True)

        # fore pytorch to use deterministic algorithms
        if "deterministic" in config:
            if config.deterministic:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
            if torch.__version__.startswith("1.7"):
                torch.set_deterministic(config.deterministic)
            else:
                torch.use_deterministic_algorithms(config.deterministic)

        # Init Lightning datamodule
        log.info(f"Instantiating training datamodule <{config.datamodule._target_}>")
        datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule, _recursive_=False, _convert_="partial")

        # Create datamodules for testing
        testmodules: List[LightningDataModule] = {"default": datamodule}
        if "testmodules" in config:
            for test_case_name, test_conf in config["testmodules"].items():
                if "_target_" in test_conf:
                    log.info(f"Instantiating testmodule <{test_conf._target_}>")
                    test_module = hydra.utils.instantiate(test_conf, _recursive_=False,  _convert_="partial")
                    testmodules[test_case_name] = test_module

        # Init Lightning model
        log.info(f"Instantiating model <{config.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(config.model, _recursive_=False,  _convert_="partial")

        # Init Lightning callbacks
        callbacks: List[Callback] = []
        if "callbacks" in config:
            for _, cb_conf in config["callbacks"].items():
                if "_target_" in cb_conf:
                    log.info(f"Instantiating callback <{cb_conf._target_}>")
                    callbacks.append(hydra.utils.instantiate(cb_conf))

        # Init Lightning loggers
        logger: List[LightningLoggerBase] = []
        if "logger" in config:
            for _, lg_conf in config["logger"].items():
                if "_target_" in lg_conf:
                    log.info(f"Instantiating logger <{lg_conf._target_}>")
                    logger.append(hydra.utils.instantiate(lg_conf))

        # Init Lightning trainer
        log.info(f"Instantiating trainer <{config.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(
            config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
        )

        # Send some parameters from config to all lightning loggers
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(
            config=config,
            model=model,
            datamodule=datamodule,
            trainer=trainer,
            callbacks=callbacks,
            logger=logger,
        )

        # Train the model
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

        log.info(f"Fitting model finished.")

        # save datamodule
        # do this after the first call to setup, but before the second one in test
        # torch.save(datamodule, "datamodule.pt")

        # gather results
        results = []

        # Evaluate model on test set after training
        if not config.trainer.get("fast_dev_run"):
            log.info("Starting testing!")
            for test_case_name, module in testmodules.items():
                result = trainer.test(datamodule=module)[0]
                result["test_case_name"] = test_case_name
                results.append(result)

        # Make sure everything closed properly
        log.info("Finalizing!")
        utils.finish(
            config=config,
            model=model,
            datamodule=datamodule,
            trainer=trainer,
            callbacks=callbacks,
            logger=logger,
        )

        df = pd.DataFrame(results)
        df.to_csv("results.csv")

        # Print path to best checkpoint
        log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

        # Return metric score for hyperparameter optimization
        optimized_metric = config.get("optimized_metric")
        if optimized_metric:
            return trainer.callback_metrics[optimized_metric]
    except Exception as e:
        log.exception(e)
        log.error(f"Training terminated by exception")
