import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class HeartDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        split: str= "train",
        seed: int= 42,
        train_ratio: float= 0.8,
        val_ratio: float= 0.1,
    ):
        data = pd.read_csv(csv_file)

        #handling all the zeros in Cholesterol replacing them with the median
        data["Cholesterol"] = data["Cholesterol"].replace(0, np.nan)
        data["Cholesterol"] = data["Cholesterol"].fillna(data["Cholesterol"].median())
        #encode the catgeorys
        categorys = ["Sex","ChestPainType","RestingECG","ExerciseAngina","ST_Slope"]
        data= pd.get_dummies(data, columns=categorys)
        #drop label
        X = data.drop(columns=["HeartDisease"]).to_numpy(dtype=np.float32)
        self.feature_names = data.drop(columns=["HeartDisease"]).columns.tolist()
        y = data["HeartDisease"].to_numpy()

        # Reproducible shuffle
        n = len(data)
        idx = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

        train_end = int(train_ratio * n)
        val_end = int((train_ratio + val_ratio) * n)

        train_idx = idx[:train_end]
        val_idx = idx[train_end:val_end]
        test_idx  = idx[val_end:]

        if split == "train":
            split_idx = train_idx
        elif split == "validate":
            split_idx = val_idx
        elif split == "test":
            split_idx = test_idx
        else:
            raise ValueError("split must be 'train', 'validate', or 'test'")

        #standardize train stats only
        train_x= X[train_idx]
        mean= train_x.mean(axis=0)
        std= train_x.std(axis=0)
        std= np.where(std == 0, 1.0, std)

        self.features= ((X[split_idx] - mean)/std)
        self.labels= y[split_idx]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label



