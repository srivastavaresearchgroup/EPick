# EPick
This is an implementation of the EPick model (https://arxiv.org/abs/2109.02567).

# Requirements
```
Python 3.8
Tensorflow-gpu 2.5
CUDA 11.2
```
# Training
python ./Train/train.py --tfrecords_dir = datapath --checkpoint_dir=model_saving_path hyperparameters setting like batch_size,  epochs

# Testing
[1] Epick testing: python ./Test/Epick/epick_test.py --tfrecords_dir datapath ---checkpoint_path saveed_model
[2] PhaseNet: the code can be accessed in https://github.com/wayneweiqiang/PhaseNet
[3] ARRU:The code can be accessed in https://github.com/tso1257771/Attention-Recurrent-Residual-U-Net-for-earthquake-detection.

# Data
## data format: 
>.tfrecords

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
