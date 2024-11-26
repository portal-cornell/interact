# InteRACT: Transformer Models for Human Intent Prediction Conditioned on Robot Actions

[Kushal Kedia](https://kushal2000.github.io/), [Atiksh Bhardwaj](https://portal-cornell.github.io/), [Prithwish Dan](https://portfolio-pdan101.vercel.app/), [Sanjiban Choudhury](https://www.sanjibanchoudhury.com/)

Cornell University

**ICRA 2024** 

[Project Page](https://portal-cornell.github.io/interact/) | [arxiv](https://arxiv.org/abs/2311.12943)


## Installation

Follow these steps to install `InteRACT`:

1. Create and activate the conda environment:
   ```bash
   cd interact
   conda create --name interact python=3.8.16
   conda activate interact
   pip install -r requirements.txt
   pip install -e . 
   ```

## Preliminaries

1. Set the `base_dev_dir` to your working directory in all of the config files

2. Create a new directory for data under interact so that the repo has the following structure:
  ### Repo Structure
  ```
  ├── config
  │   ├── *.yaml files 
  ├── interact
  |   ├── checkpoints
  |     ├── HH_checkpoints
  |     ├── HR_checkpoints
  |   ├── data
  |     ├── comad_data
  |     ├── amass
  |     ├── cmu_mocap (optional)     
  |   ├── model
  |     ├── model architecture files...
  |   ├── utils
  |     ├── utility files...
  |   ├── scripts
  |     ├── eval_hh.py / eval_hr.py        <- evaluation scripts
  |     ├── pretrain_intent_forecaster.py  <- pretraining on H-H 
  |     |── finetune_intent_forecaster.py  <- finetuning on H-H
  |     |── hr_transfer.py                 <- transferring to H-R
  |   ├── mapping
  |     ├── files for joint mapping...
  |   ├── body_models
  |     ├── SMPL skeleton file (used for AMASS data)
  |
  ├── environment.yml
  ├── README.md
  ├── setup.py

  ```

## Dataset Installation and Preprocessing

### AMASS 
Download datasets listed in ```configs/synthetic_amass.yaml``` from the official [AMASS website](https://amass.is.tue.mpg.de/).

The AMASS dataset contains data of single human motion. Preprocess this data to create synthetic two-human data:
```
python scripts/create_synthetic_amass.data.py
```
Update the config file ```configs/synthetic_amass.yaml``` before running this script.

### CoMaD
Download the data from this link [Data](https://cornell.app.box.com/s/jb0wau30dqotcjsak78ks64ea1o88yan) into the correct data directory.

You can also wget the .zip file using:
```
wget https://cornell.box.com/shared/static/6ss0mfojdof8q1z9ru7go58rwxqbnel5.zip -O comad_data.zip
```


## Training

1. Run the pretraining script on large-scale H-H data:
   ```bash
   python scripts/pretrain_intent_forecaster.py
   ```
2. Run the finetuning script on H-H interaction data. 
    ```bash
    python scripts/finetune_intent_forecaster.py
    ```
3. Run the script to transfer the model to  the H-R setting. 
    ```bash
    python scripts/hr_transfer.py
    ```

## Evaluation

1. Run the evaluation script for H-H:
   ```bash
   python scripts/eval_hh.py
   ```
2. Run the evaluation script for H-R: 
    ```bash
    python scripts/eval_hr.py
    ```


### BibTeX
   ```bash
   @article{kedia2023interact,
    title={InteRACT: Transformer Models for Human Intent Prediction Conditioned on Robot Actions},
    author={Kedia, Kushal and Bhardwaj, Atiksh and Dan, Prithwish and Choudhury, Sanjiban},
    journal={arXiv preprint arXiv:2311.12943},
    year={2023}
  }
   ``` 

### Acknowledgement
* MRT is adapted from [MRT](https://github.com/jiashunwang/MRT)
