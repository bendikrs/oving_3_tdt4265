import torchvision
from collections import OrderedDict
import pathlib
import matplotlib.pyplot as plt
from torch.utils import data
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy, weights_init
from datetime import datetime
import torch
import torch.nn.functional as F
from torchsummary import summary

class Model4a(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 10)  # No need to apply softmax,
                                            # as this is done in nn.CrossEntropyLoss

        for param in self.model.parameters():# Freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters():# Unfreeze the last fully-connected
            param.requires_grad = True          # layer
        for param in self.model.layer4.parameters():# Unfreeze the last 5 convolutional
            param.requires_grad = True              # layers
    
    def forward(self, x):
        x = self.model(x)
        return x

def create_plots(trainer: Trainer, name: str, header: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.suptitle(header, fontsize = 14) # Added header to plots
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()

if __name__ == "__main__":
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 32
    learning_rate = 5e-4
    early_stop_count = 4
    Optimizer = "Adam Optimizer"
    dataloaders = load_cifar10(batch_size, task="4a_model") # using the Model4a transforms
    model = Model4a()
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()

    
    final_val_acc = list(trainer.validation_history["accuracy"].values())[-1]
    final_test_acc = list(trainer.test_history["accuracy"].values())[-1]
    final_train_acc = list(trainer.train_history["accuracy"].values())[-1]
    final_train_loss = list(trainer.train_history["loss"].values())[-1]
    print(f'Final validation accuracy {final_val_acc}')
    print(f'Final train accuracy {final_train_acc}')
    print(f'Final test accuracy {final_test_acc}')
    print(f'Final train loss {final_train_loss}')

    plotName = "task4a_model_" + datetime.now().strftime("%a_%H_%M")
    header = "Task 4a"

    summary(trainer.model, (3,32,32))
    f = open(pathlib.Path("plots").joinpath("plotlogs.txt"), "a")
    f.write("\n--------------------------------------------------------------------" + \
        plotName + "\n" + \
        f'Final validation accuracy {final_val_acc}\n' + \
        f'Final train accuracy {final_train_acc}\n' + \
        f'Final test accuracy {final_test_acc}\n' +\
        f'Final train loss {final_train_loss}\n' + \
        f'Batch size: {batch_size}, Learning rate: {learning_rate}\n' +\
        f'Optimizer: {Optimizer} \n' +\
        str(model) + \
        "\n--------------------------------------------------------------------\n\n\n")
    f.close()
    create_plots(trainer, plotName, header)
