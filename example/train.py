import os
import shutil
import sys
from glob import glob

# from utilis import ROC_AUC_multiclass
import numpy as np
import torch
import torch.nn as nn
import yaml
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.handlers.param_scheduler import LRScheduler
from ignite.metrics import Accuracy
from natsort import natsorted
from pymatgen.io.vasp.inputs import Poscar
from torch.optim.lr_scheduler import StepLR

from dataloader import get_train_val_test_loader
from graph import CrystalGraphDataset, prepare_batch_fn

# from ignite.contrib.handlers.tensorboard_logger import *
from settings import Settings

training_data = sys.argv[1]

try:
    checkpoint_dir = sys.argv[2]
except:
    checkpoint_dir = "model_checkpoints"

try:
    logfile = sys.argv[3]
except:
    logfile = "log.model"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)


def train(
    graphs,
    settings,
    model,
    output_dir=os.getcwd() + "model",
    screen_log="log.model",
):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    global best_val_accuracy
    global total_batch_loss
    global iteration_count

    best_val_accuracy, total_batch_loss, iteration_count = -1e300, 0, 0

    train_loader, val_loader, test_loader = get_train_val_test_loader(
        graphs,
        collate_fn=graphs.collate,
        batch_size=settings.batch_size,
        train_ratio=settings.train_ratio,
        val_ratio=settings.val_ratio,
        test_ratio=settings.test_ratio,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
        train_size=settings.train_size,
        test_size=settings.test_size,
        val_size=settings.val_size,
    )

    # print(len(train_loader),len(test_loader),len(val_loader))

    # print(len(train_loader),len(test_loader),len(val_loader))

    # -----------losss--------------------

    criterion = nn.CrossEntropyLoss()
    val_metrics = {
        # "rocauc": ROC_AUC_multiclass()
        "accuracy": Accuracy(),
    }

    # ---------model----------------

    model.to(device)

    if settings.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )

    if settings.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=settings.learning_rate,
            momentum=settings.momentum,
        )

    # ------------tariner---------------

    trainer = create_supervised_trainer(
        model,
        optimizer,
        criterion,
        prepare_batch=prepare_batch_fn,
        device=device,
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global total_batch_loss
        global iteration_count
        total_batch_loss += engine.state.output
        iteration_count += 1

    # ----------scheduler-------------------

    if settings.scheduler:

        torch_lr_scheduler = StepLR(
            optimizer, step_size=settings.step_size, gamma=settings.gamma
        )
        scheduler = LRScheduler(torch_lr_scheduler)
        trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)

        # @trainer.on(Events.EPOCH_STARTED)
        # def print_lr():
        #    print(optimizer.param_groups[0]["lr"])

    if settings.progress:
        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {"loss": x})

    # ----------Evaluator---------------------

    evaluator = create_supervised_evaluator(
        model,
        prepare_batch=prepare_batch_fn,
        metrics=val_metrics,
        device=device,
    )

    test_evaluator = create_supervised_evaluator(
        model,
        prepare_batch=prepare_batch_fn,
        metrics=val_metrics,
        device=device,
    )

    # ----------Checkpoint-------------------

    def Checkpoint(epoch, is_best, path=os.getcwd(), filename="checkpoint.pt"):

        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
                "trainer": trainer.state_dict(),
                "best_accuracy": best_val_accuracy,
            },
            "{}/{}_{}".format(path, filename, epoch),
        )

        if is_best:
            shutil.copyfile(
                "{}/{}_{}".format(path, filename, epoch),
                "{}/model_best.pt".format(path),
            )

    # ----------------resume---------------------

    if settings.resume:

        if os.path.isfile("{}/checkpoint.pt".format(output_dir)):
            print(
                "=> loading checkpoint '{}'".format(
                    "{}/checkpoint.pt".format(output_dir)
                )
            )
            checkpoint = torch.load("{}/checkpoint.pt".format(output_dir))
            epoch = checkpoint["epoch"]
            settings.epochs = settings.epochs - epoch - 1
            best_val_error = checkpoint["best_accuracy"]
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["lr_scheduler"])
            trainer.load_state_dict(checkpoint["trainer"])

        else:
            print("=> no checkpoint file found")

    else:
        os.system("rm {}".format(screen_log))

    # -----------logging resulst after every epoch--------------

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):

        global best_val_accuracy
        global total_batch_loss
        global iteration_count

        # print("validating")
        epoch = engine.state.epoch

        avg_batch_loss = total_batch_loss / iteration_count
        total_batch_loss, iteration_count = 0, 0

        # print("validating")
        epoch = engine.state.epoch

        evaluator.run(val_loader)

        valmetrics = evaluator.state.metrics
        val_accuracy = valmetrics["accuracy"]

        if len(test_loader) != 0:
            test_evaluator.run(test_loader)
            testmetrics = test_evaluator.state.metrics
            test_accuracy = testmetrics["accuracy"]

        if epoch % settings.checkpoint_every == 0:
            Checkpoint(epoch, is_best=False, path=output_dir)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(
                "Saving the data for best accuracy of {} obtained at epoch {}".format(
                    best_val_accuracy, epoch
                )
            )
            Checkpoint(
                epoch, is_best=True, path=output_dir
            )  # add checkpoint function here

        if len(test_loader) != 0:
            pbar.log_message(
                "Avg_Train_loss:{:4f},  Accuracy_val: {:4f},  Accuracy_test: {:4f}".format(
                    avg_batch_loss, val_accuracy, test_accuracy
                )
            )

            with open(screen_log, "a") as outfile:
                outfile.write(
                    "{:4f},{:4f},{:4f}\n".format(
                        avg_batch_loss, val_accuracy, test_accuracy
                    )
                )

        else:
            pbar.log_message(
                "Avg_Train_loss:{:4f},  Accuracy_val: {:4f}".format(
                    avg_batch_loss, val_accuracy
                )
            )
            with open(screen_log, "a") as outfile:
                outfile.write(
                    "{:4f},{:4f}\n".format(avg_batch_loss, val_accuracy)
                )

        epoch += 1

    trainer.run(train_loader, max_epochs=settings.epochs)


# # Settings

# In[20]:


if os.path.exists("custom_config.yaml"):
    with open("custom_config.yaml", "r") as file:
        custom_dict = yaml.full_load(file)
        settings = Settings(**custom_dict)
else:
    settings = Settings()


# # Loading dataset


poscars = natsorted(glob("{}/*.POSCAR".format(training_data)))


classification_dataset = []

for file in poscars:
    poscar = Poscar.from_file(file)

    dictionary = {
        "structure": poscar.structure,
        "target": np.array(
            [float(lab) for lab in poscar.comment.split(",")], dtype="int"
        ),
    }

    classification_dataset.append(dictionary)


# In[22]:


# -----------data loading-----------------------

graphs = CrystalGraphDataset(
    classification_dataset,
    neighbors=settings.neighbors,
    rcut=settings.rcut,
    delta=settings.search_delta,
)


# # Model

# In[23]:


from model import CEGAN

net = CEGAN(
    settings.gbf_bond,
    settings.gbf_angle,
    n_conv_edge=settings.n_conv_edge,
    h_fea_edge=settings.h_fea_edge,
    h_fea_angle=settings.h_fea_angle,
    n_classification=settings.n_classification,
    pooling=settings.pooling,
    embedding=settings.embedding,
)

train(graphs, settings, net, output_dir=checkpoint_dir, screen_log=logfile)
