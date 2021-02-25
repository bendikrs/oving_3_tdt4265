from collections import OrderedDict
import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy


class ExampleModel(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super(ExampleModel, self).__init__() # usikker på om exampleModel og self egentlig skal inn her
        # TODO: Implement this function (Task  2a)
        num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes
        # Define the convolutional layers
        # self.feature_extractor = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=image_channels,
        #         out_channels=num_filters,
        #         kernel_size=5,
        #         stride=1,
        #         padding=2
        #     )

        self.modelList = OrderedDict([
            ('conv1', nn.Conv2d(
                in_channels=image_channels, 
                out_channels=num_filters, 
                kernel_size=5, stride=1, padding=2
            )),
            ('relu1', nn.ReLU()),
            ('maxPool1', nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )),
            ('conv2', nn.Conv2d(
                in_channels=num_filters, 
                out_channels=num_filters*2, 
                kernel_size=5, stride=1, padding=2
            )),
            ('relu2', nn.ReLU()),
            ('maxPool2', nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )),
            ('conv3', nn.Conv2d(
                in_channels=num_filters*2, 
                out_channels=num_filters*4, 
                kernel_size=5, stride=1, padding=2
            )),
            ('relu3', nn.ReLU()),
            ('maxPool3', nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )),
            ('flattern', nn.Flatten(start_dim=1 # kanskje?
            )),
            ('fc1', nn.Linear(
                in_features=4*4*128,
                out_features=64
            )),
            ('relu4', nn.ReLU()),
            ('out', nn.Linear(
                in_features=64,
                out_features=10
            ))            
        ])
        self.model = nn.Sequential(self.modelList)
        
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 32*32*32
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        
        ''' # koffor er denne her? er ikkje siste laget 64 -> 10 units?
        self.classifier = nn.Sequential( 
            nn.Linear(self.num_output_features, num_classes),
        )
        '''

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        batch_size = x.shape[0]
        # out = self.model._modules['conv1']
        out = self.model(x)
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
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
    dataloaders = load_cifar10(batch_size)
    model = ExampleModel(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    # code for calculating all the accuracies in task 2b)
    final_test_acc = list(trainer.test_history["accuracy"].values())[-1]
    final_val_acc = list(trainer.validation_history["accuracy"].values())[-1]
    final_train_acc = list(trainer.train_history["accuracy"].values())[-1]
    print(f'Final validation accuracy {final_val_acc}\nFinal train accuracy {final_train_acc}\nFinal test accuracy {final_test_acc}\n')
    create_plots(trainer, "task2")
