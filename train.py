import argparse
import os
from datetime import datetime

import torch
import torchvision.transforms as transforms
from WeatherClassifier.dataset import WeatherDataset, AttributesDataset, mean, std
from WeatherClassifier.test import calculate_metrics, validate
from WeatherClassifier.model import WeatherModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

def get_cur_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')


def checkpoint_save(model, name, epoch):
    f = os.path.join(name, 'checkpoint-{:06d}.pth'.format(epoch))
    torch.save(model.state_dict(), f)
    print('Saved checkpoint', f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training pipeline')
    parser.add_argument('--attributes_file', type=str, default='./train.csv',
                          help="Path to the file with attributes")
    parser.add_argument('--device', type=str, default='cuda', help="Device: 'cuda' or 'cpu")
    args = parser.parse_args()

    start_epoch = 1
    N_epochs = 5
    batch_size = 5
    num_workers = 2 # gpu number
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")

    # The property variable contains the labels for the categories
    # in the dataset and the mapping between string names and ids
    attributes = AttributesDataset(args.attributes_file)

    # Specify image transformations for enhancement during training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2),
                                shear=None, resample=False, fillcolor=(255,255,255)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # During verification, tensors and normalized transforms are used
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = WeatherDataset('./train.csv', attributes, train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataset = WeatherDataset('./val.csv', attributes, val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = WeatherModel(n_period_classes=attributes.num_periods,
                         n_road_classes=attributes.num_roads,
                         n_weather_classes=attributes.num_weathers).to(device)

    optimizer = torch.optim.Adam(model.parameters())

    logdir = os.path.join('./logs/', get_cur_time())
    savedir = os.path.join('./checkpoints/', get_cur_time())
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    logger = SummaryWriter(logdir)

    n_train_samples = len(train_dataloader)
    # Uncomment the following line to see a sample image with real tags and all tags in the Val dataset:

    # visualize_grid(model, val_dataloader, attributes, device, show_cn_matrices=False, show_images=True,
    #                checkpoint=None, show_gt=True)
    # print("\nAll period labels:\n", attributes.period_labels)
    # print("\nAll road labels:\n", attributes.road_labels)
    # print("\nAll weather labels:\n", attributes.weather_labels)

    print("Starting traning ...")

    for epoch in range(start_epoch, N_epochs + 1):
        total_loss = 0
        accuracy_period = 0
        accuracy_road = 0
        accuracy_weather = 0

        for batch in train_dataloader:
            optimizer.zero_grad()

            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            loss_train, losses_train = model.get_loss(output, target_labels)
            total_loss += loss_train.item()
            batch_accuracy_period,  batch_accuracy_road, batch_accuracy_weather = \
                calculate_metrics(output, target_labels)

            accuracy_period += batch_accuracy_period
            accuracy_road += batch_accuracy_road
            accuracy_weather += batch_accuracy_weather

            loss_train.backward()
            optimizer.step()

        print("epoch {:4d}, loss: {:.4f}, period: {:.4f}, road: {:.4f}, weather: {:.4f}".format(
            epoch,
            total_loss / n_train_samples,
            accuracy_period / n_train_samples,
            accuracy_road / n_train_samples,
            accuracy_weather / n_train_samples
        ))

        logger.add_scalar('train_loss', total_loss / n_train_samples, epoch)

        if epoch % 5 == 0:
            validate(model, val_dataloader, logger, epoch, device)

        if epoch % 25 == 0:
            checkpoint_save(model, savedir, epoch)


