#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:57:46 2024

@author: soumensmacbookair
"""

#%% Imports
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from PIL import Image
import copy

#%% Display the image
def display_image(
    csv_path: str,
    image_path: str
) -> None:

    # Read the .csv file to know annotations
    landmark_df = pd.read_csv(csv_path)
    image_name = image_path.split("/")[-1]
    landmarks = landmark_df[landmark_df.iloc[:,0] == image_name].iloc[:,1:].to_numpy().astype(np.float64).reshape((-1,2))

    # Show the image and landmarks
    image = Image.open("/".join(csv_path.split("/")[:-1]) + "/" + image_name)

    plt.imshow(image)
    plt.scatter(landmarks[:,0], landmarks[:,1], s=10, marker=".", c="red")
    plt.title(image_name)
    plt.show()

display_image("./faces/face_landmarks.csv", "./faces/2956581526_cd803f2daa.jpg")

#%% Implement the custom Dataset class
class FaceDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        transforms: torchvision.transforms = None
    ) -> None:

        super(FaceDataset, self).__init__()
        self.landmark_df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(
        self
    ) -> int:

        return self.landmark_df.shape[0]

    @staticmethod
    def transform_features(
        data_x: Image.Image
    ) -> torch.tensor:

        x = torchvision.transforms.PILToTensor()(data_x)
        x = x.to(dtype=torch.float64)

        return x

    def __getitem__(
        self,
        index: int,
        is_transform: bool = True
    ) -> Tuple[Image, np.array]:

        image_name = self.landmark_df.iloc[index,0]
        landmarks = self.landmark_df.iloc[index,1:].to_numpy().astype(np.float64).reshape((-1,2))

        image = Image.open(self.root_dir + "/" + image_name)
        orig_image = copy.deepcopy(image)

        if self.transforms:
            image = self.transforms(image)

        image = FaceDataset.transform_features(image)
        orig_image = FaceDataset.transform_features(orig_image)

        if is_transform:
            return image, landmarks
        else:
            return orig_image, landmarks

    def display_samples(
        self
    ) -> None:

        random_indices = np.random.choice(np.arange(self.__len__()), size=12, replace=False)
        fig, axes = plt.subplots(3, 4)

        for i, ax in enumerate(axes.flat):
            index = random_indices[i]
            image, landmarks = self.__getitem__(index, is_transform=False)
            image_name = self.landmark_df.iloc[index,0]

            # image = torchvision.transforms.ToPILImage()(image)
            image = image.transpose(0, 2).transpose(0, 1)

            ax.imshow(image.to(torch.int64))
            ax.scatter(landmarks[:, 0], landmarks[:, 1], s=5, marker=".", c="red")
            ax.set_title(image_name)
            ax.axis('off')

        plt.tight_layout()
        plt.show()

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256))
])

face_dataset = FaceDataset(csv_path="./faces/face_landmarks.csv",
                           root_dir="./faces",
                           transforms=transforms)

# Plot few samples of the face dataset
face_dataset.display_samples()

# Create a batch
dataloader = torch.utils.data.DataLoader(face_dataset,
                                         batch_size=4,
                                         shuffle=True,
                                         drop_last=True)

print("END")
