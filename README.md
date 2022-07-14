# CEGAN: Crystal Edge Graph Attention Network for multiscale classification of materials environment

Software implementation of Crystal Edge Graph Attention Network (CEGAN) workflow that uses graph attention-based architecture to perform multiscale classification of materials.

The following paper describes the details of the CGCNN framework:

## Table of Contents
- [Introduction](#Introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Setting up a model](#setting)
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

## Installation
First, install anaconda pacckege (https://docs.anaconda.com/anaconda/install/). Then, create and activate a conda environment for CEGAN using
```
conda create --name cegan python=3.8
conda activate cegan
```
Now, install the packages. E.g.,
```
pip install pymatgen==2021.2.16
```
### To install CEGAN code
```
git init
git clone git@github.com:sbanik2/CEGAN.git
```
## Setting up a model
To set up 3 things are necessary.
-	1. Copy CEGAN code in the run directory.
-	2. The code accepts the training data structures as POSCAR format. Create a directory containing the all the POSCAR files. The class label should me mentioned within the POSCAR file. For example, for a directory 
``` 
Training/0.POSCAR
 ```
-	The class labels re mentioned within the POSCAR file as comments. There can be two scenaris for the (a) global classification task (b) local classification task. For the global classification only one label for the whole  structure  is required which is provided as
```
0 # Class label
1.0
9.758649 0.000000 0.000000
0.000000 4.338197 0.000000
1.739575 0.000000 9.596382
Cu
4
```
-	For a local level classification i.e., labels for each class
```
0,1,1,0 # Class label
1.0
9.758649 0.000000 0.000000
0.000000 4.338197 0.000000
1.739575 0.000000 9.596382
Cu
4
```
