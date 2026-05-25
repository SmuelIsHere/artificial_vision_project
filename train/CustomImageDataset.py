import torch
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict

class CustomImageDataset(Dataset):

    def __init__(self, image_paths, labels, device, transform=None):
        self.image_paths = image_paths
        self.labels = labels    
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]
        label = self.labels[idx]
        label = torch.tensor([label[3], label[4], label[5]]).to(self.device)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

    @staticmethod
    def make_weights(dataset, device):
        from collections import defaultdict
        import torch

        gender_dict = defaultdict(int)
        bag_dict = defaultdict(int)
        hat_dict = defaultdict(int)


        for label in dataset.labels:

            gender = label[3]
            bag = label[4]
            hat = label[5]
            
            if gender != -1:
                gender_dict[gender] += 1

            if bag != -1:
                bag_dict[bag] += 1

            if hat != -1:
                hat_dict[hat] += 1

        key_gender = sorted(gender_dict.keys())
        key_bag = sorted(bag_dict.keys())
        key_hat = sorted(hat_dict.keys())

        total_gender = sum(gender_dict.values())
        gender_weights = torch.tensor([
            (total_gender / (len(gender_dict) * gender_dict[key])) for key in key_gender
        ]).to(device)

        total_bag = sum(bag_dict.values())
        bag_weights = torch.tensor([
            (total_bag / (len(bag_dict) * bag_dict[key])) for key in key_bag
        ]).to(device)

        total_hat = sum(hat_dict.values())
        hat_weights = torch.tensor([
            (total_hat / (len(hat_dict) * hat_dict[key])) for key in key_hat
        ]).to(device)
        
        return gender_weights, bag_weights, hat_weights

