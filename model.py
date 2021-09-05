from typing import List, Dict, Any
from dataclasses import dataclass, field
import dataclasses
import logging
import copy
import os
import json


logger = logging.getLogger(__name__)


@dataclass
class TagConfiguration:
    name: str
    alphabet: List[str]
    weight: float
    lookup: Dict[str, int] = field(init=False, repr=False)
    num_values: int = field(init=False, repr=False)

    def __post_init__(self):
        self.num_values = len(self.alphabet)
        self.lookup = {v: i for i, v in enumerate(self.alphabet)}

    def asdict(self):
        d = dataclasses.asdict(self)
        del d['lookup']
        del d['num_values']
        return d


@dataclass
class LemmatagConfig:
    architectures: List[str] = None
    config_class: str = None
    dropout: float = 0.5
    word_dropout: float = 0.5
    encoder_layers: int = 2
    we_dim: int = 768
    epochs: int = 10
    label_smoothing: float = 0.1
    learning_rate: float = 0.001
    grad_clip: float = 3.0
    num_words: int = None
    num_chars: int = None
    unknown_char: int = None
    bow: int = None
    eow: int = None
    num_target_chars: int = None
    tag_configurations: List = None

    def __post_init__(self):
        self.config_class = type(self).__name__

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        output['tag_configurations'] = [x.asdict() for x in output['tag_configurations']]
        return output

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> 'LemmatagConfig':
        if 'tag_configurations' in config_dict:
            config_dict['tag_configurations'] = [TagConfiguration(**x) for x in config_dict['tag_configurations']]
        config = cls(**config_dict)
        logger.info("Model config %s", str(config))
        return config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> 'LemmatagConfig':
        with open(os.path.join(pretrained_model_name_or_path, 'config.json'), 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
