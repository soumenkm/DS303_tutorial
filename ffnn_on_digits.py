#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:57:46 2024

@author: soumensmacbookair
"""

#%% Imports
import torch, time
import torchinfo
import torchvision

#%% Set the device
if torch.cuda.is_available():
    device_type = "cuda"
    print("Using GPU...")
    print(f"Total # of GPU: {torch.cuda.device_count()}")
    print(f"GPU Details: {torch.cuda.get_device_properties(device=torch.device(device_type))}")
else:
     device_type = "cpu"
     print("Using CPU...")

device = torch.device(device_type)

#%% Build the model
BATCH_SIZE = 128
class FFNN(torch.nn.Module):

    def __init__(self):

        super(FFNN, self).__init__()
        self.layer1 = torch.nn.Linear(in_features=784, out_features=64, bias=True)
        self.act1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(in_features=64, out_features=32, bias=True)
        self.act2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Linear(in_features=32, out_features=16, bias=True)
        self.act3 = torch.nn.ReLU()
        self.layer4 = torch.nn.Linear(in_features=16, out_features=10, bias=True)
        self.act4 = torch.nn.Softmax(dim=1)

    def forward(self, input_x):

        assert input_x.shape[1] == 784, "input_x must have 784 features"
        assert input_x.shape.__len__() == 2, "input_x must be of rank 2"

        layers = [self.layer1, self.act1, self.layer2, self.act2,
            self.layer3, self.act3, self.layer4, self.act4]
        output_y = input_x
        for layer in layers:
            output_y = layer(output_y)

        return output_y

model = FFNN()
model = model.to(device=device)
print(torchinfo.summary(model, (BATCH_SIZE, 784)))

#%% Create the data pipeline
def transform_x(x):
    x = torchvision.transforms.functional.pil_to_tensor(x)
    x = x/255.0
    x = torch.reshape(x, (-1,))
    return x

def transform_y(y):
    y = torch.nn.functional.one_hot(torch.tensor(y), num_classes=10)
    return y

training_data = torchvision.datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.Lambda(transform_x),
    target_transform=torchvision.transforms.Lambda(transform_y)
)

test_data = torchvision.datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.Lambda(transform_x),
    target_transform=torchvision.transforms.Lambda(transform_y)
)

train_dl = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

#%% Compile the model
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0)
loss_fn = torch.nn.CrossEntropyLoss()

def calc_accuracy(true_y, pred_y, is_count = False):
    pred_y = pred_y.argmax(dim=1)
    true_y = true_y.argmax(dim=1)
    if is_count:
        return (true_y == pred_y).to(torch.float64).sum().item()
    else:
        return (true_y == pred_y).to(torch.float64).mean().item()

#%% Create the training loop
t1 = time.time()
epoch = 10
for ep in range(epoch):
    for i, (input_x, true_y) in enumerate(train_dl):
        input_x, true_y = input_x.to(device=device), true_y.to(device=device)
        pred_y = model(input_x)
        loss = loss_fn(pred_y, true_y.argmax(dim=1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        acc = calc_accuracy(true_y, pred_y)
    print(f"Epoch: {ep+1}, Last Batch Loss: {loss:.2f}, Last Batch Acc: {acc:.2f}")

t2 = time.time()
print(f"Time taken on {device_type}: {(t2-t1):.5f} sec")

#%% Test accuracy
def calculate_accuracy(model, test_loader):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device=device), labels.to(device=device)
            outputs = model(inputs)
            correct += calc_accuracy(true_y=labels, pred_y=outputs, is_count=True)
            total += len(labels)

    # Calculate accuracy
    accuracy = correct / total
    return accuracy

test_accuracy = calculate_accuracy(model, test_dl)
print("Test Accuracy:", test_accuracy)



