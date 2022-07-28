from typing import Optional

from pydantic import BaseModel
from pydantic.typing import Literal


class Settings(BaseModel):

    # ----------------------------------------------------
    search_type: Literal["local", "global"] = "local"

    POOL = {"local": False, "global": True}

    # ---------------------Graph creation--------------------

    neighbors: int = 12
    rcut: float = 3.0
    search_delta: float = 1.0
    n_classification: int = 2

    # --------------dataloader parameter--------------------

    train_size: Optional[int] = None
    test_size: Optional[int] = None
    val_size: Optional[int] = None
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    return_test: bool = True
    num_workers: int = 1
    pin_memory: bool = False
    batch_size: int = 64

    # ----------------- model training parameters-----------------

    bond_fea_len: int = 80
    angle_fea_len: int = 80
    n_conv_edge: int = 3

    gbf_bond: dict = {'dmin': 0, 'dmax': 8, 'steps': bond_fea_len}
    gbf_angle: dict = {'dmin': -1, 'dmax': 1, 'steps': angle_fea_len}

    h_fea_edge: int = 128  # hidden feature len
    h_fea_angle: int = 128
    # Number of hidden layer

    @property
    def pooling(self):
        return self.POOL[self.search_type]

    embedding: bool = False
    checkpoint_every: int = 1

    # ----------------- model run parameters-----------------

    resume: bool = False
    epochs: int = 100
    optimizer: Literal["adam", "sgd"] = "adam"
    weight_decay: float = 0
    momentum: float = 0.9
    learning_rate: float = 1e-2

    scheduler: bool = True
    gamma: float = 0.1
    step_size: int = 30

    write_checkpoint: bool = True
    progress: bool = True
