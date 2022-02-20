"""
The template for the students to upload their code for evaluation.
"""

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

class LeNet_Mnist(nn.Module):
   def __init__(self) -> None:
       super(LeNet_Mnist, self).__init__()
       self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
       self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
       self.conv2_drop = nn.Dropout2d()
       self.fc1 = nn.Linear(320, 50)
       self.fc2 = nn.Linear(50, 10)

   def forward(self, x) -> torch.Tensor:
       x = F.relu(F.max_pool2d(self.conv1(x), 2))
       x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
       x = x.view(-1, 320)
       x = F.relu(self.fc1(x))
       x = F.dropout(x, training=self.training)
       x = self.fc2(x)
       return F.log_softmax(x, dim=1)


class Adv_Training():
    def __init__(self, device, file_path, epsilon=0.2, min_val=0, max_val=1):
        self.epsilon = epsilon
        self.min_val = min_val
        self.max_val = max_val
        self.device = device
        self.model = self.constructor(file_path).to(device)

    def constructor(self, file_path=None):
        model = LeNet_Mnist()
        if file_path != None:
            model.load_state_dict(torch.load(file_path+'/lenet_mnist_model.pth', map_location=self.device)) # lenet_mnist_model.pth, defense_project-model.pth
        return model


    def perturb(self, original_images, labels):
        original_images.requires_grad = True
        self.model.eval()
        outputs = self.model(original_images)
        loss = F.nll_loss(outputs, labels)
        self.model.zero_grad()
        loss.backward()
        data_grad = original_images.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_image = original_images + self.epsilon*sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, self.min_val, self.max_val)
        return perturbed_image

    def train(self, model, trainset, device, epoches=10):
        model.to(device)
        model.train()
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=10)
        dataset_size = len(trainset)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        for epoch in range(epoches):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs = inputs.to(device)
                labels = labels.to(device)
            # --------------TODO--------------\
                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            # --------------End TODO--------------\
                running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / dataset_size))
            running_loss = 0.0
        return model


