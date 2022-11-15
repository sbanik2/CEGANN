# CEGANN: Crystal Edge Graph Attention Neural Network 

<p align="justify"> Implementation of Crystal Edge Graph Attention Network (CEGAN) workflow that uses graph attention-based architecture to perform multiscale classification of materials. </p>

The following paper describes the details of the CGCNN framework:

## Table of Contents
- [Introduction](#Introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Setting up a model](#Setting-up-a-model)
	- [training a model](#training-a-model)
	- [make predictions](#make-predictions)
- [Customize model parameters](#customize-model-parameters)
- [Using a pre-trained model](#using-a-pretrained-model)
- [Data availability](#data-availability)
- [Citation]( #data-availability)
- [License](#license)

## Introduction
<p align="justify"> Crystal Edge Graph Attention Network (CEGAN) [https://doi.org/10.48550/arXiv.2207.10168] workflow that uses graph attention-based architecture to learn unique feature representations and perform classification of materials belonging to different classes and scales. The edge-graph representation of the structures is passed to a Hierarchical message passing block for the convolution operations. The output of the convolved feature vectors from the edge and angle convolution layers are then passed to the aggregation block where feature representations of each of the structures are generated for the prediction task. </p>



<p align="center"> <a href="url"><img src="https://github.com/sbanik2/CEGAN/blob/main/Figs/Workflow.png" align="centre" height="400" width="600" ></a> </p>



## Prerequisites
This package requires:
- [PyTorch](http://pytorch.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pymatgen]( https://pymatgen.org/)
- [pytorch-ignite](https://pytorch.org/ignite/index.html)
- [scikit-learn](https://scikit-learn.org/stable/)
- [pydantic](https://pydantic-docs.helpmanual.io/)
- [natsort](https://natsort.readthedocs.io/en/master/)

*Note that the code may not be compatible with the recent pymatgen version. The recommended version is 2021.2.16.

## Installation
First, install the anaconda package (https://docs.anaconda.com/anaconda/install/). Then, create and activate a Conda environment for CEGAN using
```
conda env create --name cegan -f environment.yml
conda activate cegan
```
This will also install all the prerequisites in the cegan environment.

### To install CEGAN code
```
git clone git@github.com:sbanik2/CEGAN.git
```
## Setting up a model

 To set up a model
-	Copy CEGAN code in the run directory.
-	<p align="justify"> The code accepts the training data structures in POSCAR format. Create a directory containing all the POSCAR files. The class label should be mentioned within the POSCAR file itself. For example, a directory "Training" will contain 0.POSCAR,1.POSCAR … etc.  </p>
``` 
Training/0.POSCAR
 ```
-	<p align="justify"> The class labels are mentioned within the POSCAR file as comments. A script for adding class labels to POSCAR is  (Label. ipynb)  provide in the post directory. It should be noted that for multiclass classification the class labels should start from 0. For 5 class classification task,the class labels should be (0,1,2,3,4). For the classification task, there can be two scenarios. (a) global classification task (b) local classification task. For the global classification, only one label for the whole structure is required which is provided as.  </p>
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
To train a model simply go to the run directory with the cegan code and use the following
```
python train.py <path-to-the-training-data-directory> <output-checkpoint-path> <log-file-path>
```
<p align="justify"> During the training, the model checkpoints the current model parameters for a given epoch, and the best set obtained so face (“model_best.pt”). The path for the directories may be provided or it will dump the parameters in a default “model_checkpoints” directory in the current path. The same goes for the log file where the training loss and the validation accuracy are stored (default “log.model”).   </p>

### make predictions

<p align="justify">To make predictions, all the structures should be in the id.POSCAR format. Where “id” corresponds to the crystal ids and can take any value.  Run, </p>

```
python predict.py <path-to-the-prediction-data-directory> <path-to-best-model-checkpoint-parameters>

```
<p align="justify">The results will be stored in “predictions.json” with the key of the dictionary as the “id” of the crystal. Model dumps two outputs in the JSON file (1)  is the embedding of the structures (Feature representation) (2) Class label of the Structure. </p>

### Customizable model parameters
The model has its own default set of parameters for training and predictions.
```
search_type ["local","global"]   <default “local”>                  # For the type of search
neighbors                        <default 12>                       # Number of nearest neighbors for graph construction
rcut                             <default 3>                        # Initial cutoff for finding neighbors
search_delta                     <default 1>                        # Incerment in rcut for finding nearest neighbors
n_classification                 <default 2>                        # Number of classfication classes
train_size                       <default None>                     # Size of training data
test_size                        <default None>                     # Size of test data
val_size                         <default None>                     # Size of validation data
train_ratio                      <default 0.8>                      # Training ratio
val_ratio                        <default 0.1>                      # Validation ratio
test_ratio                       <default 0.1>                      # test ratio
return_test                      <default False>                    # Whether to return test loader or not
num_workers                      <default 1>                        # Data loader option
pin_memory                       <default False>                    # Data loader option
batch_size                       <default 64>                       # Data loader option
bond_fea_len                     <default 80>                       # Size of edge feature vector
angle_fea_len                    <default 80>                       # Size of the angle feaure vector
n_conv_edge                      <default 3>                        # number of edge convolution
h_fea_edge                       <default 128>                      # Hideen features for edge dense layer output
h_fea_angle                      <default 128>                      # Hideen features for angle dense layer output
embedding                        <default False>                    # Whether the predict the embeddings of the strutures or not
checkpoint_every                 <default 1>                        # Write checkpoint after this many epochs
resume                           <default False>                    # To resume search
epochs                           <default 100>                      # number epochs
optimizer: ["adam", "sgd"]       <default "adam">                   # optimizer option
weight_decay                     <default 0>                        # optimizer option
momentum                         <default 0.9>                      # optimizer option
learning_rate                    <default 1e-2>                     # optimizer option
scheduler                        <default True>                     # using a step learning rate reduction
gamma                            <default 0.1>                      # scheduler option
step_size                        <default 30>                       # scheduler option
write_checkpoint                 <default True>                     # Whether to write checkpoint or not
progress                         <default True>                     # Show progress bar
```

<p align="justify">To use parameter values different than the default ones, one must specify the parameters in a separate YAML file named "custom_config.yaml" and the file should be in the same directory as train.py. An example of training and prediction with the custom parameters can be found in the example directory. </p>

### Using a pre-trained model
<p align="justify"> All training data along with the pre-trained models for the classification tasks in the paper [https://doi.org/10.48550/arXiv.2207.10168] have been provided in the prediction directory. Each directory contains the training data and the validation data as “training.zip”, and “targets.zip”. The best model parameters and the custom parameters used for the training have also been provided. To make predictions on a new validation dataset copy the cegan code in the existing path. The run  </p>

```

python predict.py <path-to-the-prediction-data-directory>  model_best.pt

```

### Data availability
All the datasets used for training are available in the pretrained directory of the code.

### Citation
```
@article{banik2022cegan,
  title={CEGAN: Crystal Edge Graph Attention Network for multiscale classification of materials environment},
  author={Banik, Suvo and Dhabal, Debdas and Chan, Henry and Manna, Sukriti and Cherukara, Mathew and Molinero, Valeria and Sankaranarayanan, Subramanian KRS},
  journal={arXiv preprint arXiv:2207.10168},
  year={2022}
}
```
### License
CEGAN is licensed under the MIT License




