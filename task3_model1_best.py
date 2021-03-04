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

class Model1_test(nn.Module):
    
    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super(Model1_test, self).__init__() 
        # TODO: Implement this function (Task  2a)
        self.num_classes = num_classes
    
        # Define the convolutional layers
        self.modelList = OrderedDict([
            ('conv1', nn.Conv2d(
                in_channels=image_channels, 
                out_channels=32, 
                kernel_size=3, stride=1, padding=1
            )),
            ('batchNorm1', nn.BatchNorm2d(32)),
            ('relu1', nn.ReLU()),
            ('convx', nn.Conv2d(
                in_channels=32, 
                out_channels=64, 
                kernel_size=3, stride=1, padding=1
            )),
            ('batchNormx', nn.BatchNorm2d(64)),
            ('relux', nn.ReLU()),
            ('maxPoolx', nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )),
            ('conv2', nn.Conv2d(
                in_channels=32*2, 
                out_channels=64*2, 
                kernel_size=3, stride=1, padding=1
            )),
            ('batchNorm2', nn.BatchNorm2d(64*2)),
            ('dropout1', nn.Dropout2d(p=0.15)),
            ('relu2', nn.ReLU()),
            ('avgPool1', nn.AvgPool2d(
                kernel_size=2
            )),
            ('conv3', nn.Conv2d(
                in_channels=64*2, 
                out_channels=128*2, 
                kernel_size=3, stride=1, padding=1
            )),
            ('batchNorm3', nn.BatchNorm2d(128*2)),
            ('maxPool3', nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(
                in_channels=128*2, 
                out_channels=256*2, 
                kernel_size=5, stride=1, padding=2
            )),
            ('batchNorm4', nn.BatchNorm2d(256*2)),
            ('relu4', nn.ReLU()),
            ('flattern', nn.Flatten(start_dim=1
            )),
            ('dropout2', nn.Dropout(p=0.5)),
            ('fc1', nn.Linear(
                in_features=512*4*4,
                out_features=2048
            )),
            ('relu5', nn.ReLU()),
            ('batchNorm5', nn.BatchNorm1d(2048)),
            ('fc2', nn.Linear(
               in_features=2048,
               out_features=512
            )),
            ('relu6', nn.ReLU()),
            ('dropout3', nn.Dropout(p=0.1)),
            ('out', nn.Linear(
                in_features=512,
                out_features=10
            ))            
        ])
        self.model = nn.Sequential(self.modelList)
        self.model.apply(weights_init)

        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 32*32*32
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss


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
    learning_rate = 0.03
    early_stop_count = 100
    Optimizer = "Average SGD, weight decay = 0.002"
    dataloaders = load_cifar10(batch_size, task="3_model1") # using the model1 transforms
    model = Model1_test(image_channels=3, num_classes=10)
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

    plotName = "task3_model1_best  " + datetime.now().strftime("%a_%H_%M")
    header = "Task 3, Model 1 best"

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

