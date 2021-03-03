
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import numpy as np
from torchvision.transforms.transforms import Grayscale
image = Image.open("images/zebra.jpg")
print("Image shape:", image.size)

model = torchvision.models.resnet18(pretrained=True)
print(model)
first_conv_layer = model.conv1
print("First conv layer weight shape:", first_conv_layer.weight.shape)
print("First conv layer:", first_conv_layer)

# Resize, and normalize the image with the mean and standard deviation
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = image_transform(image)[None]
print("Image shape:", image.shape)

activation = first_conv_layer(image)
print("Activation shape:", activation.shape)

def torch_image_to_numpy(image: torch.Tensor):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        iamge: shape=[height, width, 3] in the range [0, 1]
    """
    # Normalize to [0 - 1.0]
    image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2: # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
    image = np.moveaxis(image, 0, 2)
    return image
indices = [14, 26, 32, 49, 52]

# Task 4b -------------------------
'''
filters = [torch_image_to_numpy(first_conv_layer.weight[i]) for i in range(first_conv_layer.weight.shape[0])]
axs = []
fig = plt.figure()
plt.suptitle("Task 4b - weights (top row) and activation (bottom row)", fontsize = 14)
for itt, index in enumerate(indices):
    axs.append(fig.add_subplot(2,5, itt+1))
    axs[-1].set_title("filter "+str(index)) 
    plt.imshow(filters[index], cmap='gray')
    axs.append(fig.add_subplot(2,5, itt+6))
    axs[-1].set_title("activ. "+str(index))
    plt.imshow(activation[0][index].detach().numpy(), cmap='gray')

fig.tight_layout()
plt.savefig("task4b.png")
plt.show()
'''
# Task 4c -------------------------
activation = model.conv1(image)
activation = model.bn1(activation)
activation = model.relu(activation)
activation = model.maxpool(activation)
activation = model.layer1(activation)
activation = model.layer2(activation)
activation = model.layer3(activation)
activation = model.layer4(activation)

print(activation.shape)

#filters = [torch_image_to_numpy(model.layer4.weight[i]) for i in range(model.layer4.weight.shape[0])]
axs = []
fig = plt.figure()
plt.suptitle("Task 4c - activations of ten first filters", fontsize = 14)
for itt in range(1,6):
    axs.append(fig.add_subplot(2,5, itt))
    axs[-1].set_title("activ. "+str(itt))
    plt.imshow(activation[0][itt].detach().numpy(), cmap='gray')
    axs.append(fig.add_subplot(2,5, itt+5))
    axs[-1].set_title("activ. "+str(itt+5))
    plt.imshow(activation[0][itt+5].detach().numpy(), cmap='gray')

fig.tight_layout()
plt.savefig("task4c.png")
plt.show()
