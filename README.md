# ELLA: Exploration through Learned Language Abstraction

*Authors*: Suvir Mirchandani, Siddharth Karamcheti, Dorsa Sadigh

In this work, we introduce ELLA: Exploration through Learned Language Abstraction, a reward shaping approach for instruction following. ELLA correlates high-level instructions with simpler low-level instructions to enrich the sparse rewards afforded by the environment. ELLA has two key elements: 1) A termination classifier that identifies when agents complete low-level instructions, and 2) A relevance classifier that correlates low-level instructions with success on high-level tasks. We learn the termination classifier offline from pairs of instructions and terminal states, and we learn the relevance classifier online. Please see our preprint on Arxiv for details.

## Installation

Clone this repository.

```
git clone https://github.com/Stanford-ILIAD/ELLA.git
cd ELLA
```

Create a new environment and activate it.

```
conda env create -f env.yml -n ella
conda activate ella
```

Install BabyAI and MiniGrid.

```
cd babyai
pip install -e .
cd ../gym-minigrid
pip install -e .
```

`wandb` integration is built into this codebase: optionally, run `wandb login`, and uncomment the `--wb` option in the training scripts.

## Sample Runs:

These examples are for the PutNext-Room (PNL) Level with GoTo-Room (GTL) as a subtask.

### Generate demos

These are used by ELLA's termination classifier, ELLA's relevance classifier (for validation), and LEARN.

```
scripts/sample_runs/gen_demos/gtl_term_cls.sh
scripts/sample_runs/gen_demos/pnl_rel_cls_oracle.sh
scripts/sample_runs/gen_demos/gtl_learn.sh
```

### Train classifiers

Train ELLA's termination classifier, initialize ELLA's relevance classifier, and train LEARN's classifier.

```
scripts/sample_runs/train_cls/gtl_term_cls.sh
scripts/sample_runs/train_cls/pnl_rel_cls_init.sh
scripts/sample_runs/train_cls/gtl_learn.sh
```

### Run RL

Run vanilla PPO, PPO with ELLA, or PPO with LEARN.

```
scripts/sample_runs/rl/pnl_ppo.sh
scripts/sample_runs/rl/pnl_ella.sh
scripts/sample_runs/rl/pnl_learn.sh
```

## Code Structure & Notable Files:

* `babyai`: Contains our customized version of the package
    * `./babyai`: Internal logic for BabyAI
        * `./levels`: Level definitions
        * `./rl/algos/ppo.py`: PPO implementation
        * `./shaped_env.py`: Custom vectorized environment that parallelizes reward shaping computations
        * `./model.py`: FiLM architecture used in RL; repurposed for termination classifier
    * `./scripts`: External scripts to define and train models
        * `./subtask_prediction_model.py`: Defines relevance classifier
        * `./subtask_prediction.py`: Defines online dataset and boilerplate for training relevance classifier
        * `./train_subtask_prediction_model.py`: Script for training/pre-training relevance classifier
        * `./train_rl.py`: Script for training RL agents
        * `./imitation.py`: Boilerplate for training IL models / termination classifier
        * `./train_il.py`: Script for training IL models / termination classifier
* `scripts`: Scripts for training models
    * `./sample_runs`: Sample scripts (see above)

## Acknowledgements
- [babyai](https://github.com/mila-iqia/babyai/tree/iclr19)
- [gym-minigrid](https://github.com/maximecb/gym-minigrid)
- [rl-learn](https://github.com/prasoongoyal/rl-learn)
