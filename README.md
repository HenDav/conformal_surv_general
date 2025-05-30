# Conformalized Survival Analysis for General Right-Censored Data

This repository contains the code and materials for the paper:

> **Conformalized Survival Analysis for General Right-Censored Data**  
> *Hen Davidov, Shai Feldman, Gil Shamai, Ron Kimmel, Yaniv Romano*  
> Presented at **International Conference on Learning Representations (ICLR) 2025**  
> [[OpenReview](https://openreview.net/forum?id=JQtuCumAFD)]

## Overview
We develop a framework to quantify predictive uncertainty in survival analysis, providing a reliable lower predictive bound (LPB) for the true, unknown patient survival time. Our approach extends conformal prediction techniques to general right-censored data.

## Installation
To set up the environment, install dependencies via:
```bash
conda env create -f environment.yml
```

Experiments are reprodicible through `experiments.ipynb'

## Citation

```
@inproceedings{davidovconformalized,
  title={Conformalized Survival Analysis for General Right-Censored Data},
  author={Davidov, Hen and Feldman, Shai and Shamai, Gil and Kimmel, Ron and Romano, Yaniv},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=JQtuCumAFD}
}
```
