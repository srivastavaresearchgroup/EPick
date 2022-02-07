# EPick
This is an implementation of the EPick model (https://arxiv.org/abs/2109.02567).

# Requirements
```
Python 3.8
Tensorflow 2.5
CUDA 11.2
```
# Data
## * data format: .tfrecords

## * data source

  [1] The details of the STEAD dataset can be found in (https://github.com/smousavi05/STEAD)
  ```
  @article{mousavi2019stanford,
    title={STanford EArthquake Dataset (STEAD): A Global Data Set of Seismic Signals for AI},
    author={Mousavi, S Mostafa and Sheng, Yixiao and Zhu, Weiqiang and Beroza, Gregory C},
    journal={IEEE Access},
    year={2019},
    publisher={IEEE}
  }
  ```

  [2] The INSTANCE dataset can be downloaded from (http://doi.org/10.13127/instance) or (https://github.com/ingv/instance).
  ```
  @article{michelini2021instance,
    title={INSTANCE--the Italian seismic dataset for machine learning},
    author={Michelini, Alberto and Cianetti, Spina and Gaviano, Sonja and Giunchi, Carlo and Jozinovi{\'c}, Dario and Lauciani, Valentino},
    journal={Earth System Science Data},
    volume={13},
    number={12},
    pages={5509--5544},
    year={2021},
    publisher={Copernicus GmbH}
  }
  ```

# Cite
If you use this code for your research, please cite it with the following bibtex entry.
```
@article{li2021epick,
  title={EPick: Multi-Class Attention-based U-shaped Neural Network for Earthquake Detection and Seismic Phase Picking},
  author={Li, Wei and Chakraborty, Megha and Fenner, Darius and Faber, Johannes and Zhou, Kai and Ruempker, Georg and Stoecker, Horst and Srivastava, Nishtha},
  journal={arXiv preprint arXiv:2109.02567},
  year={2021}
}
```
