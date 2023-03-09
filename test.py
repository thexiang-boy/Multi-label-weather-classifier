import argparse
import os
import warnings
import numpy as np
import torch
import torchvision.transforms as transforms
from WeatherClassifier.dataset import WeatherDataset, AttributesDataset, mean, std
from WeatherClassifier.model import WeatherModel
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader


def checkpoint_load(model, name):
    print('Restoring check: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch


def validate(model, dataloader, logger, iteration, device, checkpoint=None):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)

    model.eval()
    with torch.no_grad():
        avg_loss = 0
        accuracy_period = 0
        accuracy_road = 0
        accuracy_weather = 0

        for batch in dataloader:

            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            val_train, val_train_losses = model.get_loss(output, target_labels)
            avg_loss += val_train.item()
            batch_accuracy_period,  batch_accuracy_road, batch_accuracy_weather = \
                calculate_metrics(output, target_labels)

            accuracy_period += batch_accuracy_period

            accuracy_road += batch_accuracy_road
            accuracy_weather += batch_accuracy_weather

        n_samples = len(dataloader)
        avg_loss /= n_samples
        accuracy_period /= n_samples
        accuracy_road /= n_samples
        accuracy_weather /= n_samples
        print('-' * 72)
        print("Validation loss: {:.4f}, period: {:.4f}, road: {:.4f}, weather: {:.4f}\n".format(
            avg_loss, accuracy_period,  accuracy_road, accuracy_weather))

        logger.add_scalar('val_loss', avg_loss, iteration)
        logger.add_scalar('val_accuracy_period', accuracy_period, iteration)
        logger.add_scalar('val_accuracy_road', accuracy_road, iteration)
        logger.add_scalar('val_accuracy_weather', accuracy_weather, iteration)
        model.train()


def visualize_grid(model, dataloader, attributes, device, show_cn_matrices=True,
                   show_images=True, checkpoint=None,show_gt=False):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)
    model.eval()

    imgs = []
    labels = []
    gt_labels = []
    gt_period_all = []
    gt_road_all = []
    gt_weather_all = []
    predicted_period_all = []
    predicted_road_all = []
    predicted_weather_all = []

    accuracy_period = 0
    accuracy_road = 0
    accuracy_weather = 0

    with torch.no_grad():
        for batch in dataloader:
            img = batch['img']
            gt_periods = batch['labels']['period_labels']
            gt_roads = batch['labels']['road_labels']
            gt_weathers = batch['labels']['weather_labels']
            output = model(img.to(device))

            batch_accuracy_period,  batch_accuracy_road, batch_accuracy_weather = \
                calculate_metrics(output, batch['labels'])
            accuracy_period += batch_accuracy_period
            accuracy_road += batch_accuracy_road
            accuracy_weather += batch_accuracy_weather

            # get the most confident prediction for each image
            _, predicted_periods = output['period'].cpu().max(1)
            _, predicted_roads = output['road'].cpu().max(1)
            _, predicted_weathers = output['weather'].cpu().max(1)

            for i in range(img.shape[0]):
                image = np.clip(img[i].permute(1, 2, 0).numpy() * std + mean, 0, 1)

                predicted_period = attributes.period_id_to_name[predicted_periods[i].item()]
                predicted_road = attributes.road_id_to_name[predicted_roads[i].item()]
                predicted_weather = attributes.weather_id_to_name[predicted_weathers[i].item()]

                gt_period = attributes.period_id_to_name[gt_periods[i].item()]
                gt_road = attributes.road_id_to_name[gt_roads[i].item()]
                gt_weather = attributes.weather_id_to_name[gt_weathers[i].item()]

                gt_period_all.append(gt_period)
                gt_road_all.append(gt_road)
                gt_weather_all.append(gt_weather)

                predicted_period_all.append(predicted_period)
                predicted_road_all.append(predicted_road)
                predicted_weather_all.append(predicted_weather)

                imgs.append(image)
                labels.append("{}\n{}\n{}".format( predicted_road, predicted_period, predicted_weather))
                gt_labels.append("{}\n{}\n{}".format( gt_road, gt_period, gt_weather))

        if not show_gt:
            n_samples = len(dataloader)
            print("\nAccuracy:\nperiod: {:.4f}, road: {:.4f}, weather: {:.4f}".format(
                accuracy_period / n_samples,
                accuracy_road / n_samples,
                accuracy_weather / n_samples))
        # confusion matrix
        model.train()


def calculate_metrics(output, target):
    _, predicted_period = output['period'].cpu().max(1)
    gt_period = target['period_labels'].cpu()

    _, predicted_road = output['road'].cpu().max(1)
    gt_road = target['roda_labels'].cpu()

    _, predicted_weather = output['weather'].cpu().max(1)
    gt_weather = target['weather_labels'].cpu()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        accuracy_period = balanced_accuracy_score(y_true=gt_period.numpy(), y_pred=predicted_period.numpy())
        accuracy_road = balanced_accuracy_score(y_true=gt_road.numpy(), y_pred=predicted_road.numpy())
        accuracy_weather = balanced_accuracy_score(y_true=gt_weather.numpy(), y_pred=predicted_weather.numpy())

    return accuracy_period, accuracy_road, accuracy_weather


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference pipeline')
    parser.add_argument('--checkpoint', type=str, default=r'checkpoints\2023-02-23\checkpoint-000050.pth',help="Path to the checkpoint")
    parser.add_argument('--attributes_file', type=str, default='./weather-product-images/weathers.csv',
                        help="Path to the file with attributes")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    # Attribute variables contain the label of the category
    # in the dataset and the mapping between the string name and ID
    attributes = AttributesDataset(args.attributes_file)
    attributes = AttributesDataset(args.attributes_file)
    # During verification, tensors and normalization transformations are used
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_dataset = WeatherDataset('val.csv', attributes, val_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    model = WeatherModel(n_period_classes=attributes.num_periods,n_road_classes=attributes.num_roads,
                         n_weather_classes=attributes.num_weathers).to(device)
    # visualize
    visualize_grid(model, test_dataloader, attributes, device, checkpoint=args.checkpoint)










