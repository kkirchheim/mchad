import argparse
import logging
from typing import Any, Dict, Optional, Union

import pytorch_lightning.loggers as plog
from omegaconf import OmegaConf


log = logging.getLogger(__name__)


class YamlLogger(plog.LightningLoggerBase):
    """
    Logs hyperparameters to a yaml file
    """

    def __init__(self, filename="config.yaml", name="yaml", version=None, prefix="", save_dir="."):
        super(YamlLogger, self).__init__()
        self.filename = filename
        self._version = version
        self._name = name
        self._save_dir = save_dir

    def log_hyperparams(self, params: argparse.Namespace, *args, **kwargs):
        log.info(f"Writing hyperparameters to '{self.filename}'")
        with open(self.filename, "w") as f:
            f.write(OmegaConf.to_yaml(params))

    def experiment(self) -> Any:
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        pass

    def name(self) -> str:
        return self._name

    def version(self) -> Union[int, str]:
        return self._version

    def save_dir(self) -> Optional[str]:
        return self._save_dir
    