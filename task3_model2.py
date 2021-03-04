from collections import OrderedDict
import pathlib
import matplotlib.pyplot as plt
from torch.utils import data
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy
from datetime import datetime
import torch
from torchsummary import summary

class Model2(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super(Model2, self).__init__() 
        # TODO: Implement this function (Task  2a)
        num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes

        '''
            Differences from task 2 model:
                - Added som relu and conv
                - Added a batchNorm
                - Some difference in in/out channels
        '''
        self.modelList = OrderedDict([
            ('conv1', nn.Conv2d(
                in_channels=image_channels, 
                out_channels=64, 
                kernel_size=3, stride=1, padding=1
            )),
            ('relu1', nn.ReLU()),
            ('maxPool1', nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )),
            ('conv2', nn.Conv2d(
                in_channels=64, 
                out_channels=128, 
                kernel_size=5, stride=1, padding=2
            )),
            ('relu2', nn.ReLU()),
            ('maxPool2', nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )),
            ('conv3', nn.Conv2d(
                in_channels=128, 
                out_channels=256, 
                kernel_size=3, stride=1, padding=1
            )),
            ('relu3', nn.ReLU()),
            ('maxPool3', nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )),
            # ('conv4', nn.Conv2d(
            #     in_channels=128, 
            #     out_channels=256, 
            #     kernel_size=5, stride=1, padding=2
            # )),
            # ('relu4', nn.ReLU()),
            ('flattern', nn.Flatten(start_dim=1
            )),
            ('fc1', nn.Linear(
                in_features=256*4*4,
                out_features=128
            )),
            ('relu5', nn.ReLU()),
            #  ('fc2', nn.Linear(
            #     in_features=128,
            #     out_features=64
            # )),
            # ('relu6', nn.ReLU()),
            ('out', nn.Linear(
                in_features=128,
                out_features=10
            ))            
        ])
        self.model = nn.Sequential(self.modelList)


    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        batch_size = x.shape[0]
        out = self.model(x)
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out

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
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    Optimizer = "Average SGD, weight decay = 0.001"
    dataloaders = load_cifar10(batch_size, task="3_model2")
    model = Model2(image_channels=3, num_classes=10)
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

    plotName = "task3_model2_" + datetime.now().strftime("%a_%H_%M")
    header = "Task 3, Model 2"

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
