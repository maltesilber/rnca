# rNCA: Self-Repairing Segmentation Masks
[![arXiv](https://img.shields.io/badge/arXiv-2512.13397-b31b1b.svg?style=flat)](https://arxiv.org/abs/2512.13397)

This repository implements **rNCA** (self-repairing segmentation masks) for image segmentation tasks.

## Overview
<p align="center">
<img width="2100" height="870" alt="rnca" src="https://github.com/user-attachments/assets/db7f4b57-7cd5-4f29-b051-a94b9a09c239" />
</p>

**Refinement Neural Cellular Automata (rNCA)** is a lightweight post-processing method for repairing segmentaiton errors. It uses Neural Cellular Automata trained on imperfect predictions and ground truth masks to iteratively refine coarse segmentations using only local information. The learned dynamics reconnect fragmented regions, remove spurious components, and converge to stable, topologically consistent masks, making rNCA a task-agnostic and model-agnostic refinement module.

## Examples
<p align="center">
<img alt="vessel" src="https://github.com/user-attachments/assets/a5860b94-c519-4bdc-84aa-a5b2353fcd7b" />
</p>

## Usage
This repository contains a minimal example for training the **rNCA** (self-repairing segmentation masks) model on a synthetic dataset.

To get started:
1. **Install dependencies**  
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the training script**  
   To begin training the model on the synthetic dataset, simply run the following command:
   ```bash
   python src/main.py
   ```

   This will start the training process. By default, the script will use the synthetic dataset and save the trained model and outputs to the specified output directory.

## Citation
If you use **rNCA** in your research and need to reference it, please cite it as follows:

```
@misc{silbernagel2025rncaselfrepairingsegmentationmasks,
      title={rNCA: Self-Repairing Segmentation Masks}, 
      author={Malte Silbernagel and Albert Alonso and Jens Petersen and Bulat Ibragimov and Marleen de Bruijne and Madeleine K. Wyburd},
      year={2025},
      eprint={2512.13397},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.13397}, 
}
```

## License
rNCA is licensed under the MIT License.
