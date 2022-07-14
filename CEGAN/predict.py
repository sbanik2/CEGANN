import sys
import os
import torch
import shutil
import pandas as pd
import torch.nn as nn
from graph import CrystalGraphDataset,prepare_batch_fn
from utilis import Normalizer
from dataloader import get_train_val_test_loader
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, MeanAbsoluteError
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.metrics import Accuracy,Precision,Recall,ConfusionMatrix
from torch.optim.lr_scheduler import StepLR
from ignite.handlers.param_scheduler import LRScheduler
from ignite.handlers import Checkpoint, global_step_from_engine
#from ignite.contrib.handlers.tensorboard_logger import *
from settings import Settings
#from utilis import ROC_AUC_multiclass
import numpy as np
from pymatgen import Structure,Lattice
from natsort import natsorted
from glob import glob
from random import *
import yaml
from pymatgen.io.vasp.inputs import Poscar,Incar

validation_data = sys.argv[1]

try:
    checkpoint_best = sys.argv[2]
except:
    raise Exception("model parameters are needed for prediction")
    


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




poscars = natsorted(glob("target/*.POSCAR"))


# In[28]:


if os.path.exists("custom_config.yaml"):
    with open('custom_config.yaml', 'r') as file:
        custom_dict = yaml.full_load(file)
        settings = Settings(**custom_dict)
else:
    settings = Settings()




# In[26]:

classification_dataset = []

for file in poscars:
    poscar = Poscar.from_file(file)
      

    dictionary = {'structure':poscar.structure,
                  'target':  np.array([float(lab) for lab in  poscar.comment.split(",")],dtype="int"),
                     }
    
    
    classification_dataset.append(dictionary)


# In[27]:


#-----------data loading-----------------------


       
graphs = CrystalGraphDataset(
               classification_dataset,
               neighbors = settings.neighbors,
               rcut = settings.rcut,
               delta = settings.search_delta,
               mp_load = False,
               mp_pool=None,
               mp_cpu_count=100
              )






# # Load model

# In[30]:


from model import CEGAN


net = CEGAN(
             settings.gbf_bond,
             settings.gbf_angle,
             n_conv_edge=settings.n_conv_edge,
             h_fea_edge=settings.h_fea_edge,
             h_fea_angle=settings.h_fea_angle,
             n_classification=settings.n_classification,
             pooling=settings.pooling,
             embedding=True,
            )

net.to(device)

best_checkpoint = torch.load(checkpoint_best,map_location=torch.device(device))


net.load_state_dict(best_checkpoint['model'])


# In[ ]:





# In[31]:



embeddings = []
targets = []
predictions = []


for i in range(graphs.size):
    outdata = graphs.collate([graphs[i]])
    
    x,y = prepare_batch_fn(outdata,device,non_blocking=False)
    predict,embedding = net(x)
    
    embeddings.append(embedding.cpu().detach().numpy())
    targets.append(y.cpu().detach().numpy())
    predictions.append(predict.cpu().detach().numpy())
    
    
    


# In[30]:


embeddings = np.concatenate(embeddings,axis=0)
targets = np.concatenate(targets,axis=0)
predictions = np.concatenate(predictions,axis=0)

predictions = np.argmax(predictions,axis=1)



# In[32]:


np.savetxt("predict.dat",predictions,fmt='%d')


# In[33]:


np.savetxt("embeddings.dat",embeddings)


# In[34]:


np.savetxt("targets.dat",targets,fmt='%d')


# In[ ]:




