import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter


#

class WeatherModel(nn.Module):
    def __init__(self, n_period_classes, n_road_classes, n_weather_classes):
        super().__init__()

        # model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.base_model = models.mobilenet_v2().features # no classifier
        last_channel = models.mobilenet_v2().last_channel
        # The input to the classifier should be two-dimensional, but we're going to have
        # [batch_size, channels, width, height]
        # so do space averaging: reduce the width and height to 1
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        # Create a separate classifier for multi-label output
        self.period = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_period_classes)
        )
        self.road = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_road_classes)
        )
        self.weather = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_weather_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)

        # turn [batch, channels, 1, 1] -> [batch, channels], put into classifier
        x = torch.flatten(x, 1)

        return {
            'period':self.period(x),
            'road':self.road(x),
            'weather':self.weather(x)
        }

    def get_loss(self, net_output, ground_truth):
        period_loss = F.cross_entropy(net_output['period'], ground_truth['period_labels'])
        road_loss = F.cross_entropy(net_output['road'], ground_truth['road_labels'])
        weather_loss = F.cross_entropy(net_output['weather'], ground_truth['weather_labels'])
        loss = period_loss + road_loss + weather_loss
        return loss, {'period': period_loss, 'road': road_loss, 'weather': weather_loss}

if __name__ == '__main__':
    weathercls = WeatherModel(14,15,16)
    print(weathercls)
