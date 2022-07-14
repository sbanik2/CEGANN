# CEGAN: Crystal Edge Graph Attention Network for multiscale classification of materials environment

<p align="justify"> Software implementation of Crystal Edge Graph Attention Network (CEGAN) workflow that uses graph attention-based architecture to perform multiscale classification of materials. </p>

The following paper describes the details of the CGCNN framework:

## Table of Contents
- [Introduction](#Introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Setting up a model](#Setting-up-a-model)
	- [training a model](#training-a-model)
	- [make predictions](#make-predictions)
- [Customize model parameters](#customize-model-parameters)
- [Using a pretrained model](#using-a-pretrained-model)
- [ Data availability] (#data-availability)
- [License](#license)

## Introduction
<p align="justify"> Crystal Edge Graph Attention Network (CEGAN) [cite] workflow that uses graph attention-based architecture to learn unique feature representations and perform classification of materials belonging to different classes and scales. The edge-graph representation of the structures is passed to a Hierarchical message passing block for the convolution operations. The output of the convolved feature vectors from the edge and angle convolution layers are then passed to the aggregation block where feature representations of each of the structures are generated for the prediction task. </p>



<p align="center"> <a href="url"><img src="https://github.com/sbanik2/CEGAN/blob/main/Figs/Workflow.png" align="centre" height="400" width="600" ></a> </p>



## Prerequisites
This package requires:
- [PyTorch](http://pytorch.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pymatgen]( https://pymatgen.org/)
- [pytorch-ignite](https://pytorch.org/ignite/index.html)

*Note that the code may not be compatible with the recent pymatgen version. The recommended version is 2021.2.16.

## Installation
First, install the anaconda package (https://docs.anaconda.com/anaconda/install/). Then, create and activate a Conda environment for CEGAN using
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

 To set up a model 3 things are necessary.
-	Copy CEGAN code in the run directory.
-	<p align="justify"> The code accepts the training data structures in POSCAR format. Create a directory containing all the POSCAR files. The class label should be mentioned within the POSCAR file itself. For example, a directory "Training" will contain 0.POSCAR,1.POSCAR … etc.  </p>
``` 
Training/0.POSCAR
 ```
-	<p align="justify"> The class labels are mentioned within the POSCAR file as comments. There can be two scenarios. (a) global classification task (b) local classification task. For the global classification, only one label for the whole structure  is required which is provided as.  </p>
```
0 # Class label
1.0
9.758649 0.000000 0.000000
0.000000 4.338197 0.000000
1.739575 0.000000 9.596382
Cu
4
```
-	For a local level classification i.e., labels for each atom in a structure should be provided in a comma-separated format. E.g.,
```
0,1,1,0 # Class label
1.0
9.758649 0.000000 0.000000
0.000000 4.338197 0.000000
1.739575 0.000000 9.596382
Cu
4
```

### training a model
To train a model simply go running directory with the cegan code and use the following
```
python train.py <path-to-the-training-data-directory> <output-checkpoint-path> <log-file-path>
```
<p align="justify"> During the training, the model checkpoints the current model parameters for a given epoch, and the best set obtained so face (“model_best.pt”). The path for the directories may be provided or it will dump the parameters in a default “model_checkpoints” directory in the current path. The same goes for the log file where the training loss and the validation accuracy are stored (default “log.model”).   </p>

### make predictions

To make predictions 
```
python predict.py <path-to-the-prediction-data-directory> <path-to-best-model-checkpoint-parameters>
```
### Customize model parameters
The customizable model parameters with the default values are as follows
```
**search_type**: ["local","global"] <default “local”> # For the type of search
**neighbors <default 12> # Number of nearest neighbors for graph construction
rcut <default 3> # Intial cutoff for finding neighbors
search_delta <default 1> # Incerment in rcut for finding nearest neighbors
n_classification <default 2> # Number of classfication classes
train_size <default None> # Size of training data
test_size <default None> # Size of test data
val_size <default None> # Size of validation data
 train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    return_test: bool = True
    num_workers: int = 1
    pin_memory: bool = False
    batch_size: int = 64
        
    
    bond_fea_len: int = 80
    angle_fea_len: int = 80
    n_conv_edge: int = 3
   
      
    
    h_fea_edge: int = 128                    # hidden feature len
    h_fea_angle: int = 128 
                                             # Number of hidden layer 
        
        
    @property
    def pooling(self):
        return self.POOL[self.search_type]
                  
    embedding : bool = False
    checkpoint_every : int = 1
    
        resume: bool = False
    epochs: int = 100
    optimizer: Literal["adam", "sgd"] = "adam"
    weight_decay: float = 0
    momentum: float = 0.9
    learning_rate: float = 1e-2
    scheduler: bool = True
    gamma: float = 0.1
    step_size: int = 30
    

    write_checkpoint: bool = True
    progress: bool = True 
```

