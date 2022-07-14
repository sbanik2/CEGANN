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


<img src="https://github.com/sbanik2/CEGAN/blob/main/Figs/Workflow.png" width="200">
