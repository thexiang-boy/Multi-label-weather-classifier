# split_data.py
import argparse
import csv
import os
import time

import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


def save_csv(data, path, fieldnames=['image_path', 'period', 'weather', 'road']):
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(dict(zip(fieldnames, row)))


def download_image(url, path, period_label, weather_label, road_label):
    req = requests.get(url)
    filename = url.split('_')[-4]
    if req.status_code != 200:
        print('下载异常')
        return
    try:
        if not os.path.exists(path):
            os.mkdir(path)
        # img_name = filename
        filename = filename + period_label + weather_label + road_label + '.jpeg'
        filepath = os.path.join(path, filename)
        with open(filepath, 'wb') as f:
            f.write(req.content)
            # print(f"下载成功 {filename}")
    except Exception as e:
        print(e)
    return filepath


def process_image(pd, num=100):
    # download images and append annotations
    datas = []
    for i in tqdm(range(0, num), desc='正在下载图片'):
        if i == num:
            return
        time.sleep(0.2)
        url = pd.iloc[i]['url']
        # only 3 category
        period = pd.iloc[i]['period']
        weather = pd.iloc[i]['weather']
        road = pd.iloc[i]['road']
        time.sleep(0.5)
        img_name = download_image(url, os.path.join(input_folder, 'category'), period, weather, road)
        datas.append([img_name, period, weather, road])
    return datas


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split data for the dataset')
    parser.add_argument('--input', type=str, default="./data", help="Path to the dataset")
    parser.add_argument('--output', type=str, default="", help="Path to the working folder")

    args = parser.parse_args()
    input_folder = args.input
    output_folder = args.output
    annotation = os.path.join(input_folder, 'category.txt')
    # open annotation
    anno_pd = pd.read_csv(annotation, sep='\t')
    all_data = process_image(anno_pd, 200)
    print(f"下载完成数据{len(all_data)}张")
    # Set the seed of the random number generator so that we can reproduce the result later
    np.random.seed(42)
    # Construct a Numpy array from the list
    all_data = np.asarray(all_data)
    # 200 samples were randomly selected
    inds = np.random.choice(200, 200, replace=False)
    # split data to train/val
    # save as .csv
    save_csv(all_data[inds][:180], os.path.join(output_folder, 'train.csv'))
    save_csv(all_data[inds][180:200], os.path.join(output_folder, 'val.csv'))
