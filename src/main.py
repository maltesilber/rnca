"""
Example main script to configure and start training the RNCA model.
"""
from dataclasses import dataclass
from train import train
from pathlib import Path

@dataclass
class Config:
    epochs: int = 30
    batch_size: int = 16
    replace_n: int = 2
    
    state_channels: int = 16
    img_kernel_size: tuple = (3, 3)
    hidden_dim: int = 128
    dropout_rate: float = 0.5
    alive_threshold: float = 0.1

    pool_size: int = 256
    nca_steps: int = 64
    
    learning_rate: float = 1e-4
    
    data_dir: Path = Path('src/data')
    ckpt_dir: Path = Path('src/ckpt')
    seed: int = 0

def main():
    cfg = Config()
    train(cfg)

if __name__ == "__main__":
    main()