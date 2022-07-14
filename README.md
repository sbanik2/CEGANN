# CEGAN: Crystal Edge Graph Attention Network for multiscale classification of materials environment

Software implementation of Crystal Edge Graph Attention Network (CEGAN) workflow that uses graph attention-based architecture to perform multiscale classification of materials.

The following paper describes the details of the CGCNN framework:

## Table of Contents
- [Introduction](#Introduction)
- [Prerequisites](#prerequisites)
- [Installation] (#installation)
- [Setting up a model] (#setting up a model)
	- [training a model]
	- [make predictions]
  - [customize model parameters]
- [Using a pre trained model]
- [ Data availability]
- [License](#license)

## Introduction
Crystal Edge Graph Attention Network (CEGAN) [cite] workflow that uses graph attention-based architecture to learn unique feature representations and perform classification of materials belonging to different classes and scales. The edge-graph representation of the structures is passed to a Hierarchical message passing block for the convolution operations. The output of the convolved feature vectors from the edge and angle convolution layers are then passed to the aggregation block where feature representations of each of the structures are generated for the prediction task.


<a href="url"><img src="https://github.com/sbanik2/CEGAN/blob/main/Figs/Workflow.png" align="centre" height="600" width="1000" ></a>



## Prerequisites
This package requires:
- [PyTorch](http://pytorch.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pymatgen]( https://pymatgen.org/)
- [pytorch-ignite](https://pytorch.org/ignite/index.html)

*Note that the the code may not be compatible with recent pymatgen version. The recommended version is 2021.2.16.

## installation
First, install anaconda pacckege (https://docs.anaconda.com/anaconda/install/). Then, create and activate a conda environment for CEGAN using
```
conda create --name cegan python=3.8
conda activate cegan
```
Now, install the packeges, E.g.,
```
pip install pymatgen==2021.2.16
```.


