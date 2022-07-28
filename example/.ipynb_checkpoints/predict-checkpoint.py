import json
import os
import sys
from glob import glob
from random import *

# from utilis import ROC_AUC_multiclass
import numpy as np
import torch
import yaml
from natsort import natsorted
from pymatgen.io.vasp.inputs import Poscar

from graph import CrystalGraphDataset, prepare_batch_fn

# from ignite.contrib.handlers.tensorboard_logger import *
from settings import Settings

validation_data = sys.argv[1]

try:
    checkpoint_best = sys.argv[2]
except:
    raise Exception("model parameters are needed for prediction")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


poscars = natsorted(glob("{}/*.POSCAR".format(validation_data)))


# In[28]:


if os.path.exists("custom_config.yaml"):
    with open('custom_config.yaml', 'r') as file:
        custom_dict = yaml.full_load(file)
        settings = Settings(**custom_dict)
else:
    settings = Settings()


# In[26]:

classification_dataset = []
struct_label = []

for file in poscars:
    label = file.split("/")[-1].replace(".POSCAR", "")
    poscar = Poscar.from_file(file)
    struct_label.append(label)

    dictionary = {
        'structure': poscar.structure,
        'target': np.array(
            [1]
        ),  # dummy target where target of the vaidation data is not known
    }

    classification_dataset.append(dictionary)


# In[27]:


# -----------data loading-----------------------


graphs = CrystalGraphDataset(
    classification_dataset,
    neighbors=settings.neighbors,
    rcut=settings.rcut,
    delta=settings.search_delta,
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

best_checkpoint = torch.load(
    checkpoint_best, map_location=torch.device(device)
)


net.load_state_dict(best_checkpoint['model'])


# In[ ]:


# In[31]:


predictions = {}


for i in range(graphs.size):
    outdata = graphs.collate([graphs[i]])
    label = struct_label[i]

    x, y = prepare_batch_fn(outdata, device, non_blocking=False)
    predict, embedding = net(x)

    predictions[label] = {
        "class": np.argmax(predict.cpu().detach().numpy(),axis=1).tolist(),
        "embeddings": embedding.cpu().detach().numpy().tolist(),
    }


# In[30]:

with open('predictions.json', 'w') as f:
    json.dump(predictions, f)
