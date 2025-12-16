# rNCA: Self-Repairing Segmentation Masks
[![arXiv](https://img.shields.io/badge/arXiv-2503.14525-b31b1b.svg?style=flat)](https://arxiv.org/abs/2503.14525)

This repository implements **rNCA** (self-repairing segmentation masks) for image segmentation tasks.

## Overview
<p align="center">
  <img src="figures/rnca.png" alt="rNCA figure" width="900"/>
</p>

**Refinement Neural Cellular Automata (rNCA)** is a lightweight post-processing method for repairing segmentaiton errors. It uses Neural Cellular Automata trained on imperfect predictions and ground truth masks to iteratively refine coarse segmentations using only local information. The learned dynamics reconnect fragmented regions, remove spurious components, and converge to stable, topologically consistent masks, making rNCA a task-agnostic and model-agnostic refinement module.

## Examples
<p align="center"> <img src="figures/vessel.gif" alt="rNCA in action for vessel segmentation" width="600"/> </p>

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
