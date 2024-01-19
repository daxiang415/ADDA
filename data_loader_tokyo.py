import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

import random
import numpy as np
import torch

def set_all_seeds(seed=42):
    random.seed(seed)         # 设置Python内建的随机种子
    np.random.seed(seed)     # 设置NumPy的随机种子
    torch.manual_seed(seed)  # 设置PyTorch的随机种子

    if torch.cuda.is_available():   # 如果你使用CUDA
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_all_seeds(42)

def scale(x, y, trans):
    x = trans.transform(x).astype(np.float32) ####transform 将训练数据转换成正态分布
    y = np.log10(y.to_numpy().astype(np.float32))
    #y=y.to_numpy().astype(np.float32)

    return x, y


def load_data():
    # 读取数据
    laixi_data = pd.read_excel('tokyo_data.xlsx')

    props = ['day_of_year',
             'month',
             'hour',
             'wind_speed',
             'radiation_time',
             'sun_rise',
             'rainfall',
             'temperature',
             'humidity',
             'water_vapor_pressure',
             'dew_temperature',
             ]

    target = ['solar_radiation_modification']

    # 时间划分
    laixi_data['time'] = pd.to_datetime(laixi_data['time'], dayfirst=True)  # 改time为时间戳格式
    s_date = '2020-01-01 00:00'
    e_date = '2021-12-31 23:00'
    s_date_test = '2021-12-31 00:00'
    e_date_test = '2022-12-31 23:00'

    # 划分数据集
    train_data = laixi_data[(s_date <= laixi_data['time']) & (laixi_data['time'] <= e_date)]
    test_data = laixi_data[(s_date_test <= laixi_data['time']) & (laixi_data['time'] <= e_date_test)]

    train_x, train_y = train_data[props], train_data[target]
    test_x, test_y = test_data[props], test_data[target]

    # 归一化
    trans = StandardScaler().fit(train_x)  ####StandardScaler().fit()用于计算 train_x的均值和方差
    train_x, train_y = scale(train_x, train_y, trans)
    test_x, test_y = scale(test_x, test_y, trans)





    print('训练集: ', train_x.shape, train_y.shape)
    print('测试集: ', test_x.shape, test_y.shape)
    return (train_x, train_y), (test_x, test_y)



class TrainData_tokyo(Dataset):
    def __init__(self):
        super(TrainData_tokyo, self).__init__()
        (self.train_x, self.train_y), _ = load_data()
        self.encode_x = np.concatenate([self.train_x, self.train_y], axis=-1) ####编码意味着在最后的维度上将 train_x和train_y 进行拼接
        self.step = 24  ####时间周期
        self.prum = 1  ####单步预测

    def __len__(self):   ####下划线的定义： _ 隐藏函数, __实例化后自动调用此函数
        return self.train_x.shape[0] - self.step

    def __getitem__(self, item):   ###item: 在len的范围内随机取一个数字
        en = self.encode_x[item:item + self.step, :]
        de = self.train_x[item + self.step:item + self.step + self.prum, :3]
        tg = self.train_y[item + self.step:item + self.step + self.prum].reshape(-1)
        return en, de, tg


class TestData_tokyo(Dataset):
    def __init__(self):
        super(TestData_tokyo, self).__init__()
        _, (self.test_x, self.test_y) = load_data()
        self.encode_x = np.concatenate([self.test_x, self.test_y], axis=-1)
        self.step = 24
        self.prum = 1

    def __len__(self):  ####下划线的定义： _ 隐藏函数, __实例化后自动调用此函数
        return self.test_x.shape[0] - self.step

    def __getitem__(self, item):  ###item: 在len的范围内随机取一个数字
        en = self.encode_x[item:item + self.step, :]
        de = self.test_x[item + self.step:item + self.step + self.prum, :3]
        tg = self.test_y[item + self.step:item + self.step + self.prum].reshape(-1)
        return en, de, tg


if __name__ == "__main__":
    trainData = TestData_tokyo()
    genData = DataLoader(trainData, batch_size=16, shuffle=True)
    for i, (en, de, tg) in enumerate(genData):
        print(i, en.shape, de.shape, tg.shape)