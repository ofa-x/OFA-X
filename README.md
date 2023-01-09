# Harnessing the Power of Multi-Task Pretraining for Ground-Truth Level Natural Language Explanations

_Demo page coming soon!_

This repository contains the code for the publication ["Harnessing the Power of Multi-Task Pretraining for Ground-Truth 
Level Natural Language Explanations"](https://arxiv.org/abs/2212.04231) by Björn Plüster, Jakob Ambsdorf, Lukas Braach, Jae Hee Lee and Stefan Wermter.

It includes a fork of [OFA-Sys/OFA](https://github.com/OFA-Sys/OFA) (found in the `./OFA` directory) and all necessary code to train OFA on VL-NLE tasks (such as VQA-X, e-SNLI-VE, and VCR) for the [e-ViL benchmark](https://github.com/maximek3/e-ViL).

`./training` contains the code and configurations for training and evaluating the models. The
[training README](./training/Training.md) contains more information on how to run the training scripts.

`./dataset_preparation` contains the code for generating the datasets and where to get all required files.
See the [dataset preparation README](./dataset_preparation/Datasets.md) for more information.

The `./survey` directory contains all data related to the human evaluation conducted in the paper, with more information in the survey [survey README](./survey/Survey.md).

If you are using OFA-X in your work, please consider citing:

```
@article{pluster2022harnessing,
  title={Harnessing the Power of Multi-Task Pretraining for Ground-Truth Level Natural Language Explanations},
  author={Pl{\"u}ster, Bj{\"o}rn and Ambsdorf, Jakob and Braach, Lukas and Lee, Jae Hee and Wermter, Stefan},
  journal={arXiv preprint arXiv:2212.04231},
  year={2022}
}

@inproceedings{wang2022ofa,
  title={Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework},
  author={Wang, Peng and Yang, An and Men, Rui and Lin, Junyang and Bai, Shuai and Li, Zhikang and Ma, Jianxin and Zhou, Chang and Zhou, Jingren and Yang, Hongxia},
  booktitle={International Conference on Machine Learning},
  pages={23318--23340},
  year={2022},
  organization={PMLR}
}
```