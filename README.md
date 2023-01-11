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

## Model Weights

Please see the links in the table to download the trained model weights. The base-size model is only available with OFA-pretraining, while we selected the huge-size model depending on BERTScore performance of the Large model.

| Training               | Pretraining | Model Weights                                                                                                                                                                                                                                                                                |
|------------------------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| VQA-X                  | OFA         | [Base](https://drive.google.com/file/d/1Sw1I_4L0cZbaglZ4ZotsqB8tRSJ_-hqB/view?usp=sharing), [Large](https://drive.google.com/file/d/1g1CascIuRhTVFzPd2kE1DhNfnf3j7hZf/view?usp=share_link)                                                                                                   |
| VQA-X                  | Caption     | [Large](https://drive.google.com/file/d/1MP-QWl9rTUC4xk8VZ8DdXLkPoEta83B-/view?usp=share_link), [Huge](https://drive.google.com/file/d/19iJ-722TBfo7NZvRunBK4MDD3dVxIkeC/view?usp=share_link)                                                                                                |
| e-SNLI-VE              | OFA         | [Base](https://drive.google.com/file/d/1Kf-qJ3pVHSgkUFSjlwZwBc8XBQMn_WKN/view?usp=share_link), [Large](https://drive.google.com/file/d/1WUu88G5Bm94Yyx--Tkoq2lo_qBD0T9uW/view?usp=share_link), [Huge](https://drive.google.com/file/d/13H4NUoNfcrfo85STjdfxY2S2t7oE8Z7o/view?usp=share_link) |
| e-SNLI-VE              | Caption     | [Large](https://drive.google.com/file/d/1WHkDREL0jpMnyHtxOcrfw0q97x-2kSeX/view?usp=share_link)                                                                                                                                                                                               |
| VCR                    | OFA         | [Base](https://drive.google.com/file/d/1iApz-0mj_i4I3senPxUIPfsZvUqZWA_6/view?usp=share_link), [Large](https://drive.google.com/file/d/19CaqUcLWmEkKxb2bZMV0oqMRhrBK0QdJ/view?usp=share_link), [Huge](https://drive.google.com/file/d/133Dl851hXN2F2z1m3_RhEOeJmCiUJI4Q/view?usp=share_link) |
| VCR                    | Caption     | [Large](https://drive.google.com/file/d/1P4IrLOYEcp35WfG3NhV0Q6td4PLLYpqA/view?usp=share_link)                                                                                                                                                                                               |
| OFA-X_MT (e-ViL-comb.) | OFA         | [Large](https://drive.google.com/file/d/1lpuxBSdzTgn3cKP-Qsk_GqL8j0Mar4jW/view?usp=share_link)                                                                                                                                                                                               |
