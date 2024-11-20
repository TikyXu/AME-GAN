import random
import pandas as pd
import torch.nn
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.preprocessing import MinMaxScaler

CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD  = [0.2023, 0.1994, 0.2010]

def get_loaders(train_portion, batch_size, path_to_save_data):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    # train_data = datasets.CIFAR10(root=path_to_save_data, train=True, 
    #                               download=True, transform=train_transform)
    train_data = datasets.CIFAR10(root=path_to_save_data, train=True, 
                                  download=True, transform=train_transform)

    num_train = len(train_data)                        # 50k
    indices = list(range(num_train))                   # 
    split = int(np.floor(train_portion * num_train))   # 40k
    
    train_idx, valid_idx = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_idx)
    
    train_loader = DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler,
        pin_memory=True, num_workers=32)
    
    if train_portion == 1:
        return train_loader
    
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    val_loader = DataLoader(
        train_data, batch_size=batch_size, sampler=valid_sampler,
        pin_memory=True, num_workers=16)
    
    print("train loader ",train_loader.dataset.data.shape[1:]," length:",len(train_idx))    
    print("validation loader ",val_loader.dataset.data.shape[1:]," length:",len(valid_idx))
    return train_loader, val_loader
    
def get_test_loader(batch_size, path_to_save_data):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    # test_data = datasets.CIFAR10(root=path_to_save_data, train=False,
    #                              download=True, transform=test_transform)
    test_data = datasets.CIFAR10(root=path_to_save_data, train=False,
                                 download=True, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                                              shuffle=False, num_workers=16)
    return test_loader

def list_to_index(data, column):
    
    normal = ['normal']
    
    dos    = ['back', 'land', 'neptune', 'pod', 'smurf', 
              'teardrop', 'apache2', 'mailbomb', 'processtable', 'udpstorm']
    
    probe  = ['ipsweep', 'nmap', 'portsweep', 'satan', 'mscan', 'saint']
    
    r2l    = ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 
              'spy', 'warezclient', 'warezmaster', 'sendmail', 'named', 
              'snmpgetattack', 'snmpguess', 'xlock', 'xsnoop', 'worm']
    
    u2l    = ['buffer_overflow', 'loadmodule', 'perl', 'rootkit', 'httptunnel', 
              'ps', 'sqlattack', 'xterm']
    
    pdlist_class_dict = {}
    for index, data_class in enumerate([normal, dos, probe, r2l, u2l]):
        for item in data_class:
            pdlist_class_dict[item] = index
    
    # 给表格数据赋值
    data_new = pd.DataFrame([pdlist_class_dict[row] for row in data[column]])
    
    # # count=0
    # # for origin, new in zip(data[column], data_new):
    # #     count += 1
    # #     if 77 <= count <= 137:
    # #         print(f'{origin}:{new}')
    # value_count = data_new.value_counts(ascending=False)
    # value_count_ratio = [count/len(data_new) for count in value_count.values]
    
    # for line, ratio in zip(value_count, value_count_ratio):
    #     print(f'{line} {(ratio*100):4f}%')

    return data_new

def column_norm(data, col_names):
    # 计算每一列的最大最小值，并进行归一化
    for column in col_names:
        max_value = max(data[column]) # 每一列最大值
        min_value = min(data[column]) # 每一列最小值
        difference = max_value - min_value # 每一列最大最小差值

        # 如果最大值不为1，进行归一化
        if max_value != 1 :
            # 排除最大最小值相等情况
            if difference != 0:
                data[column] = [((row - min_value) /difference) for row in data[column]]
            # 排除最大最小值相等且为0的情况
            elif max_value != 0:
                data[column] = [(row / max_value) for row in data[column]]

def column_max(data, col_names):
    column_max_values = {}
    # 计算Train和Test的每一列的最大值
    for column in col_names:
        column_max_values[column] = max(data[column]) # 每一列最大值
        # print(f'{column} -- Max:{train_max_value}')
    return column_max_values
  
# 进行归一化
def column_max_norm(data, col_names, col_max_val):
    # 进行归一化
    for column in col_names:
        max_value = col_max_val[column]
        
        # 如果最大值不为1，进行归一化
        if max_value != 1 :
            data[column] = [(row / max_value) for row in data[column]]
            
def encode_text_dummy(df, name):
    # 使用 pd.get_dummies 获取 one-hot 编码的 DataFrame
    dummies = pd.get_dummies(df[name], prefix=name)
    # 使用 pd.concat 一次性添加所有新列
    df = pd.concat([df, dummies], axis=1)
    
    # 删除原始列
    df.drop(name, axis=1, inplace=True)
    
    return df
            
class NSL_KDD_Dataset(Dataset):
    def __init__(self, file_path, col_name, binary_or_multi, train_or_valid):
        NSL_KDD = pd.read_csv(file_path, names=col_name) # 加载原始数据
        data = NSL_KDD.copy() # 拷贝原始数据

        # 测试集增加特殊行(马上会删除特殊行)，以使得train和test中所有str列,拥有相同的value_count
        if train_or_valid == 'valid':            
            insert_test = [[0,'tcp','urh_i','SF',491,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0.00,0.00,0.00,0.00,1.00,0.00,0.00,150,25,0.17,0.03,0.17,0.00,0.00,0.00,0.05,0.00,'normal',20],
                           [0,'tcp','red_i','SF',491,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0.00,0.00,0.00,0.00,1.00,0.00,0.00,150,25,0.17,0.03,0.17,0.00,0.00,0.00,0.05,0.00,'normal',20],
                           [0,'tcp','http_8001','SF',491,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0.00,0.00,0.00,0.00,1.00,0.00,0.00,150,25,0.17,0.03,0.17,0.00,0.00,0.00,0.05,0.00,'normal',20],
                           [0,'tcp','aol','SF',491,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0.00,0.00,0.00,0.00,1.00,0.00,0.00,150,25,0.17,0.03,0.17,0.00,0.00,0.00,0.05,0.00,'normal',20],
                           [0,'tcp','harvest','SF',491,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0.00,0.00,0.00,0.00,1.00,0.00,0.00,150,25,0.17,0.03,0.17,0.00,0.00,0.00,0.05,0.00,'normal',20],
                           [0,'tcp','http_2784','SF',491,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0.00,0.00,0.00,0.00,1.00,0.00,0.00,150,25,0.17,0.03,0.17,0.00,0.00,0.00,0.05,0.00,'normal',20]
                          ]
            insert_test_df = pd.DataFrame(data=insert_test, columns=col_name)
            data = pd.concat([data, insert_test_df])
        
        text_category = ['protocol_type', 'service', 'flag']            
        for col in text_category:
            data = encode_text_dummy(data,col)
        
        col_name = data.columns.values.tolist() # 更新col_name
        
        # 删除测试集中多增加的特殊行
        if train_or_valid == 'valid':
            data = data[:len(data)-6]
            
            
        col_name.remove("label") # label列不作数据标准化，后续直接转换为[0, 1]
        column_norm(data, col_name) # 数据标准化
        # print(f'Dataset class count:({len(data["label"].value_counts())})\n{data["label"].value_counts()}')
        if binary_or_multi=='binary':
            # 把label数据转换为[0,1]
            label = data["label"].map(lambda x: 0 if x == "normal" else 1)
        else:
            label = list_to_index(data, "label")
            
        del data['label'] # 删除label列
        
        # 凑形状，增加5列
        addition_data = np.zeros((len(data), 5))
        addition_col = ['zero_1', 'zero_2', 'zero_3', 'zero_4', 'zero_5']
        addition = pd.DataFrame(data=addition_data, columns=addition_col)
        data = pd.concat([data, addition], axis=1)     
            
        data = data.to_numpy().astype(np.float32)
        label = label.to_numpy().reshape(label.size).astype(np.float32)

        if train_or_valid == 'train':
            train_num = 125952  # 原有长度为125973
            data = data[:train_num]
            label = label[:train_num]
        else:
            test_num = 22528  # 原有长度为22544
            data = data[:test_num]
            label = label[:test_num]

        print("NSL-KDD dataset[data:%s,label:%s]"%(data.shape, label.shape))
        
        self.len = data.shape[0]
        self.data = torch.from_numpy(data)
        self.label = torch.from_numpy(label)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.len

def get_nslkdd_train_loader(train_portion, batch_size, path_to_save_data, binary_or_multi):
    print(f'Loading Train Dataset ...')
    # file_path = "data/NSL-KDD/KDDTrain+.txt"
    col_name = ["duration", "protocol_type", "service", "flag", "src_bytes",
                "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                "is_hot_login", "is_guest_login", "count", "srv_count", "serror_rate",
                "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "score"]
    # print('Col_name:',len(col_name))
    
    train_transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    # train_data = datasets.CIFAR10(root=path_to_save_data, train=True, 
    #                               download=True, transform=train_transform)
    

    train_data =  NSL_KDD_Dataset(file_path=path_to_save_data, 
                                  col_name=col_name,
                                  binary_or_multi=binary_or_multi,
                                  train_or_valid='train')
    # train_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
    
    num_train = len(train_data)
    indices = list(range(num_train))
    random.shuffle(indices)
    split = int(np.floor(train_portion * num_train))
    
    train_idx, valid_idx = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_idx)
    
    train_loader = DataLoader(dataset=train_data, 
                              batch_size=batch_size, 
                              sampler=train_sampler, 
                              pin_memory=True, 
                              num_workers=32)
    
    if train_portion == 1.0:
        return train_loader, None
    
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    val_loader = DataLoader(dataset=train_data, 
                            batch_size=batch_size, 
                            sampler=valid_sampler, 
                            pin_memory=True, 
                            num_workers=16)
    
    return train_loader, val_loader

def get_nslkdd_test_loader(batch_size, path_to_save_data, binary_or_multi):
    print(f'Loading Test Dataset ...')
    
    col_name = ["duration", "protocol_type", "service", "flag", "src_bytes",
                "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                "is_hot_login", "is_guest_login", "count", "srv_count", "serror_rate",
                "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "score"]
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    
    # test_data = datasets.CIFAR10(root=path_to_save_data, train=False,
    #                              download=True, transform=test_transform)
    
    test_data = NSL_KDD_Dataset(file_path=path_to_save_data, 
                                col_name=col_name,
                                binary_or_multi=binary_or_multi,
                                train_or_valid='valid')
    
    test_loader = DataLoader(dataset=test_data, 
                             batch_size=batch_size,
                             shuffle=True, 
                             num_workers=16)
    return test_loader

class UNSW_NB15_Dataset(Dataset):
    def __init__(self, train_dataset_path, test_dataset_path, binary_or_multi, train_or_test):
        categorical_columns = ['proto', 'service', 'state']
        
        classification = ['Normal', 'Fuzzers', 'Analysis', 'Backdoor', 'DoS', 
                        'Exploits', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms']
        # 加载数据 train_num:125973, test_num:22544, total_data:148517
        data_train = pd.read_csv(train_dataset_path).copy()
        data_test = pd.read_csv(test_dataset_path).copy()
        total_data = pd.concat([data_train, data_test], axis=0) # 合并train和test
        total_data = total_data.drop(['id'], axis=1)
        train_num = len(data_train)
        test_num = len(data_test)

        # 特征
        features = total_data.iloc[:, :-2]     
        
        # 标签（以Binary/Multi形式加载Y的值）
        if binary_or_multi=='binary':    
            # 删除attack_cat列
            total_data = total_data.drop('attack_cat', axis=1)
            # 把labels转换为binary[0,1] 
            labels = total_data.iloc[:, -1]
        elif binary_or_multi=='multi':
            # 删除label列
            total_data = total_data.drop('label', axis=1)
            labels_class = total_data.iloc[:, -1]
            
            pdlist_class_dict = {}
            for index, data_class in enumerate(classification):
                pdlist_class_dict[data_class] = index
                    
            labels = np.array([pdlist_class_dict[row] for row in np.array(labels_class)])
            
        # One-hot编码数据
        features = pd.get_dummies(features, columns=categorical_columns)
        
        # Min-Max标准化
        scaler = MinMaxScaler().fit(features)
        features = scaler.transform(features)

        # 凑形状，增加60列
        addition_number = 60
        addition_data = np.zeros((len(total_data), addition_number))
        features = np.concatenate((features, addition_data), axis=1)
        
        # X_train = features[:train_num][:, :, np.newaxis]
        # X_test = features[train_num:][:, :, np.newaxis]
        X_train = features[:train_num]
        X_test = features[train_num:]
        Y_train = labels[:train_num]
        Y_test = labels[train_num:]
        
        if train_or_test == 'train':
            train_num = 175232
            self.len = train_num
            self.data = torch.from_numpy(X_train.astype(np.float32))[:train_num]
            self.label = torch.from_numpy(Y_train.astype(np.float32))[:train_num]
        elif train_or_test == 'test':
            test_num = 82304
            self.len = test_num
            self.data = torch.from_numpy(X_test.astype(np.float32))[:test_num]
            self.label = torch.from_numpy(Y_test.astype(np.float32))[:test_num]

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.len

def get_unsw_nb15_train_loader(train_file_path, test_file_path, batch_size, binary_or_multi):
    print(f'Loading Train Dataset ...')
    train_data = UNSW_NB15_Dataset(train_dataset_path=train_file_path, 
                                  test_dataset_path=test_file_path, 
                                  binary_or_multi=binary_or_multi, 
                                  train_or_test='train')
    
    train_loader = DataLoader(dataset=train_data, 
                             batch_size=batch_size,
                             shuffle=True, 
                             num_workers=16)
    print(f'[{train_data.data.shape}]')
    return train_loader

def get_unsw_nb15_test_loader(train_file_path, test_file_path, batch_size, binary_or_multi):
    print(f'Loading Test Dataset ...')
    test_data = UNSW_NB15_Dataset(train_dataset_path=train_file_path, 
                                  test_dataset_path=test_file_path, 
                                  binary_or_multi=binary_or_multi, 
                                  train_or_test='test')
    
    test_loader = DataLoader(dataset=test_data, 
                             batch_size=batch_size,
                             shuffle=True, 
                             num_workers=16)
    print(f'[{test_data.data.shape}]')
    return test_loader