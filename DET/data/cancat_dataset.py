from torchvision.datasets import CocoDetection
from torch.utils.data import Dataset

class ConcatCocoDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, index):
        for dataset in self.datasets:
            if index < len(dataset):
                return dataset[index]
            index -= len(dataset)

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)