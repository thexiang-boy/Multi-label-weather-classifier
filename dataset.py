import csv
import os.path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

mean = [0.4490, 0.4553, 0.4453]
std = [0.1925, 0.1844, 0.1811]


class AttributesDataset():
    def __init__(self, annotation_path):
        period_labels = []
        road_labels = []
        weather_labels = []

        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                period_labels.append(row['period'])
                weather_labels.append(row['weather'])
                road_labels.append(row['road'])

        self.period_labels = np.unique(period_labels)
        self.weather_labels = np.unique(weather_labels)
        self.road_labels = np.unique(road_labels)

        self.num_periods = len(self.period_labels)
        self.num_weathers = len(self.weather_labels)
        self.num_roads = len(self.road_labels)

        self.period_id_to_name = dict(zip(range(len(self.period_labels)), self.period_labels))
        self.period_name_to_id = dict(zip(self.period_labels, range(len(self.period_labels))))

        self.weather_id_to_name = dict(zip(range(len(self.weather_labels)), self.weather_labels))
        self.weather_name_to_id = dict(zip(self.weather_labels, range(len(self.weather_labels))))

        self.road_id_to_name = dict(zip(range(len(self.road_labels)), self.road_labels))
        self.road_name_to_id = dict(zip(self.road_labels, range(len(self.road_labels))))


class WeatherDataset(Dataset):
    def __init__(self, annotation_path, attributes, transform):
        super().__init__()

        self.transform = transform
        self.attr = attributes

        # Initializes the array to store the real label and image path
        self.data = []
        self.period_labels = []
        self.weather_labels = []
        self.road_labels = []

        # read annotation from csv
        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row['image_path'])
                self.period_labels.append(self.attr.period_name_to_id[row['period']])
                self.weather_labels.append(self.attr.weather_name_to_id[row['weather']])
                self.road_labels.append(self.attr.road_name_to_id[row['road']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Sample data by index
        img_path = self.data[idx]

        # Read image
        img = Image.open(img_path)

        # transform if necessary
        if self.transform:
            img = self.transform(img)

        # return images and labels
        dict_data = {
            'img': img,
            'labels': {
                'period_labels': self.period_labels[idx],
                'road_labels': self.road_labels[idx],
                'weather_labels': self.weather_labels[idx]
            }
        }
        return dict_data
