import argparse
import torch
import torchvision.transforms
from torch.utils.data import DataLoader
from WeatherClassifier.dataset import WeatherDataset, AttributesDataset


def get_mean_std_value(loader):
    '''
    Calculate mean and std of data
    :param loader:
    :return:
    '''
    data_sum, data_squared_sum, num_batches = 0, 0, 0

    for data in loader:

        data = data['img']
        # data: [batch_size,channels,height,width]
        # calculate the mean sum of dim=0，2，3。dim=1 is the channels number
        # does not need to be involved in the calculation
        data_sum += torch.mean(data, dim=[0,2,3])    # [batch_size,channels,height,width]
        # calculate the std sum of dim=0，2，3。dim=1 is the channels number
        data_squared_sum += torch.mean(data**2,dim=[0,2,3])  # [batch_size,channels,height,width]
        # 统计batch的数量
        num_batches += 1
    # 计算均值
    mean = data_sum/num_batches
    # 计算标准差
    std = (data_squared_sum/num_batches - mean**2)**0.5
    return mean,std


if __name__ == '__main__':
    # Training set(CIFAR-10)
    # train_dataset = datasets.CIFAR10(root='G:/datasets/cifar10',train=True,download=False,transform=transforms.ToTensor())
    # train_loader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size)

    parser = argparse.ArgumentParser(description='Training pipeline')
    parser.add_argument('--attributes_file', type=str, default='./train.csv',
                        help="Path to the file with attributes")
    parser.add_argument('--device', type=str, default='cuda', help="Device: 'cuda' or 'cpu")
    args = parser.parse_args()

    batch_size = 20

    attributes = AttributesDataset(args.attributes_file)
    train_transform = torchvision.transforms.ToTensor()# enhancement
    # Training set(Baidu)
    train_dataset = WeatherDataset('./train.csv', attributes, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, )

    mean, std = get_mean_std_value(train_dataloader)
    print(f'mean = {mean},std = {std}')
