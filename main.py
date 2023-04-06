import itertools
import os
from pathlib import Path

import numpy
import torch
import torchvision
import torchinfo

import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms, models
from moduls import data_prep, engine, utils
from torchmetrics.classification import MulticlassAccuracy
import wandb
import random
from mlxtend.plotting import plot_confusion_matrix

print(f"Torch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")

log_wandb = False
EPOCHS = 15
MODEL = models.efficientnet_b2
WEIGHTS = models.EfficientNet_B2_Weights
DATASET = "data/pizza_steak_sushi_20_percent"
IN_FEATURES = 1408 #B0 1280

if log_wandb:
    wandb.init(
        # set the wandb project where this run will be logged
        project="Base_line_tinyVGG_04",
        name=f"{EPOCHS}_epochs_trans_B2_20%",
        # track hyperparameters and run metadata
        config={

            "learning_rate": 0.001,
            "architecture": "EffitientNetB2",
            "dataset": "CustomDataset_Food101_20%_3classes",
            "epochs": EPOCHS,
        }
    )

device = utils.setup_device()


def manual_way():
    # important for imagenet
    normalize = transforms.Normalize(std=[0.485, 0.456, 0.406],
                                     mean=[0.229, 0.224, 0.255])

    train_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    return train_transform


def automatic_way():
    # From version of pytochvision 0.13+
    weights = WEIGHTS.DEFAULT  # using best version of Weights
    auto_transforms = weights.transforms()
    return auto_transforms


maunal_transforms = manual_way()
auto_trans = automatic_way()
print(auto_trans)  # See the transform we apply
train_dataloader, test_dataloader, class_names = data_prep.data_prep_imgfolder(path=DATASET,
                                                                               transforms=(auto_trans, auto_trans),
                                                                               batch_size=32)
"""
def the_old_way():
    #before 0.13 will be removed in 0.15
    model = models.efficientnet_b0(pretrained = True)
"""

def changes_to_model(model):
    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=IN_FEATURES, out_features=3, bias=True)
    )
    return model

def model_prep(model,weights=None):
    if weights is not None:
        model = model(weights=weights)
        model = changes_to_model(model=model)
    else:
        model = changes_to_model(model=model())

    return model


def train_model():

    model = model_prep(model=MODEL, weights=WEIGHTS.DEFAULT)

    torchinfo.summary(model, input_size=(32, 3, 224, 224),
                      col_names=["input_size", "output_size", "num_params", "trainable"],
                      col_width=20, row_settings=["var_names"])

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(params=model.parameters(), lr=0.001)
    acc_fn = MulticlassAccuracy(num_classes=len(class_names)).to(device)

    engine.train(model=model, device=device, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                 loss_fn=loss_fn, acc_fn=acc_fn, epochs=EPOCHS, optim=optim, log_to_wandb=log_wandb)

    num_images_to_plot = 3
    test_image_path_list = list(Path(os.path.join(DATASET, "test")).glob("*/*.jpg"))
    sample_imgs = random.sample(population=test_image_path_list,
                                k=num_images_to_plot)

    for img in sample_imgs:
        utils.pred_plot_image(model, img, class_names, device, transform=auto_trans)

    utils.pred_plot_image(model=model, class_names=class_names, device=device, transform=auto_trans,
                          image_path="data/pizza_steak_sushi/testing/Pizza_test.jpg")

    utils.save_model_dict(model, "models/", "TransferLearning_EfffitientB2_base_15e_20.pth")


def test_model():
    model = model_prep(model=MODEL)
    model = utils.load_model("models/TransferLearning_EfffitientB2_base_15e_20.pth",model=model,device=device)
    #torchinfo.summary(model, input_size=(32, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"],col_width=20, row_settings=["var_names"])

    #utils.pred_plot_image(model,"data/pizza_steak_sushi/testing/newspaper.jpg",class_names,device,transform=auto_trans)
    #utils.pred_plot_image(model, "data/pizza_steak_sushi/testing/burger.jpg", class_names, device,transform=auto_trans)

    utils.plot_most_wrong(model,test_dataloader,device,class_names,5,auto_trans)
    utils.plot_confmat_classification(model,test_dataloader,device,class_names)




train_model()
test_model()
