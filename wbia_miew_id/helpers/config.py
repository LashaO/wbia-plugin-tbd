import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

from dataclasses import asdict

def dataclass_to_dict(dataclass_instance):
    return asdict(dataclass_instance)

class DictableClass:
    def __iter__(self):
        yield from dataclass_to_dict(self).items()

@dataclass
class Data(DictableClass):
    images_dir: str
    train_anno_path: str
    val_anno_path: str
    image_size: Tuple[int, int]
    viewpoint_list: List = None
    train_n_filter_min: int = None
    val_n_filter_min: int = 2
    train_n_subsample_max: int = None
    val_n_subsample_max: int = None
    name_keys: List = field(default_factory=['name'])


@dataclass
class Engine(DictableClass):
    train_batch_size: int
    valid_batch_size: int
    epochs: int
    seed: int
    device: str
    loss_module: str
    use_wandb: bool
    num_workers: int = 0



@dataclass
class SchedulerParams(DictableClass):
    lr_start: float
    lr_max: float
    lr_min: float
    lr_ramp_ep: int
    lr_sus_ep: int
    lr_decay: float

@dataclass
class ModelParams(DictableClass):
    model_name: str
    use_fc: bool
    fc_dim: int
    dropout: float
    loss_module: str
    s: float
    margin: float
    ls_eps: float
    theta_zero: float
    pretrained: bool
    n_classes: int

@dataclass
class TestParams():
    batch_size: int = 4
    fliplr: bool = False
    fliplr_view: List = field(default_factory=list)

@dataclass
class Config(DictableClass):
    exp_name: str
    project_name: str
    checkpoint_dir: str
    comment: str
    data: Data
    engine: Engine
    scheduler_params: SchedulerParams
    model_params: ModelParams
    test: TestParams
    

def get_config(file_path: str) -> Config:
    print(f"Loading config from path: {file_path}")
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    config_dict['data'] = Data(**config_dict['data'])
    config_dict['engine'] = Engine(**config_dict['engine'])
    config_dict['scheduler_params'] = SchedulerParams(**config_dict['scheduler_params'])
    config_dict['model_params'] = ModelParams(**config_dict['model_params'])
    config_dict['test'] = TestParams(**config_dict['test'])
    

    config = Config(**config_dict)
    return config