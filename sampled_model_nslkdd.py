import os
import datetime
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from datetime import datetime



os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device_dis = torch.device('cuda:0')
# device_gen = torch.device('cuda:0')

# 获取当前日期和时间
run_time = datetime.now()
model_kind = f'NSL-KDD_Multi'

save_path = os.path.join(f'sampled_model_results', model_kind)
save_path = os.path.join(save_path, str(run_time))

os.makedirs(save_path, exist_ok=True)
os.makedirs(f'{save_path}', exist_ok=True)
os.makedirs(f'{save_path}/models', exist_ok=True)
os.makedirs(f'{save_path}/results', exist_ok=True)



batch_size = 128
train_epoches = 60
class_num = 5
last_epoch = -1
print_interval = 100



def Load_NSLKDD(path_train_data, path_test_data, batch_size, binary_or_multi):
    # 列名，根据NSL-KDD数据集文档定义
    column_names = [
        "duration", "protocol_type", "service", "flag", "src_bytes",
        "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
        "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
        "is_hot_login", "is_guest_login", "count", "srv_count", "serror_rate",
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "score"
        # "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
    ]
    normal = ['normal']    
    dos    = ['back', 'land', 'neptune', 'pod', 'smurf', 
              'teardrop', 'apache2', 'mailbomb', 'processtable', 'udpstorm']
    probe  = ['ipsweep', 'nmap', 'portsweep', 'satan', 'mscan', 'saint']
    r2l    = ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 
              'spy', 'warezclient', 'warezmaster', 'sendmail', 'named', 
              'snmpgetattack', 'snmpguess', 'xlock', 'xsnoop', 'worm']
    u2l    = ['buffer_overflow', 'loadmodule', 'perl', 'rootkit', 'httptunnel', 
              'ps', 'sqlattack', 'xterm']
    
    categorical_columns = ['protocol_type', 'service', 'flag']
        
    # 加载数据 train_num:125973, test_num:22544, total_data:148517
    data_train = pd.read_csv(path_train_data, header=None, names=column_names)
    data_test = pd.read_csv(path_test_data, header=None, names=column_names)
    total_data = pd.concat([data_train, data_test], axis=0) # 合并train和test
    train_num = len(data_train)
    # 删除Score列
    total_data = total_data.drop('score', axis=1)

    # 特征、标签
    features = total_data.iloc[:, :-1] 
    labels = total_data.iloc[:, -1]
    
    # One-hot编码数据
    features = pd.get_dummies(features, columns=categorical_columns)
    
    # Min-Max标准化
    scaler = MinMaxScaler().fit(features)
    features = scaler.transform(features)
    
    # 凑形状，增加6列
    addition_number = 6
    addition_data = np.zeros((len(total_data), addition_number))
    features = np.concatenate((features, addition_data), axis=1)

    pdlist_class_dict = {}
    for index, data_class in enumerate([normal, dos, probe, r2l, u2l]):
        for item in data_class:
            pdlist_class_dict[item] = index

    # 给表格数据赋值
    if binary_or_multi == 'multi':
        labels = np.array([pdlist_class_dict[row] for row in labels])
    elif binary_or_multi == 'binary':
        labels = np.array([0 if row=='normal' else 1 for row in labels])
    
    X_train = np.array(features[:train_num]).astype(np.float32)
    X_test = np.array(features[train_num:]).astype(np.float32)
    Y_train = np.array(labels[:train_num]).astype(np.longlong)
    Y_test = np.array(labels[train_num:]).astype(np.longlong)

    # 创建tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

    # 创建DataLoader
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader



sampled_path =  f'models/AMEGAN_NSLKDD_Multi-Class 2024-10-08 03:12:41.647242/Discriminator/Epoch14/Iter891/Wasserstein/[11, 5, 5, 15, 14, 6]-Accuracy:0.00035635086291362837 Latency:0.7035398230088495 .pt'
dis = torch.load(sampled_path).cuda()

dis = nn.DataParallel(dis)
dis.train()



criterion = nn.CrossEntropyLoss()

params_dis = [param for param in dis.parameters()]
optimizer_dis = Adam(params=params_dis, lr=5e-4, weight_decay=1e-3)
scheduler_dis = CosineAnnealingLR(optimizer_dis, T_max=train_epoches, last_epoch=last_epoch)

# 加载数据
path_train_data='datasets/NSL-KDD/KDDTrain+.txt'
path_test_data='datasets/NSL-KDD/KDDTest+.txt'
train_loader, test_loader = Load_NSLKDD(path_train_data=path_train_data, 
                                        path_test_data=path_test_data, 
                                        batch_size=batch_size,
                                        binary_or_multi='multi')



best_accu = 0
best_epoch = 0
for epo in range(train_epoches):
    print(f'Training Epoch{epo+1}')
    
    dis.train()
    for index, (train_image, train_label) in enumerate(train_loader):
        train_image, train_label = train_image.view(train_image.shape[0], -1, train_image.shape[1]).cuda(non_blocking=True), train_label.to(torch.int64).cuda(non_blocking=True)
        train_label = torch.nn.functional.one_hot(train_label, class_num).to(torch.float32)
        optimizer_dis.zero_grad()
        
        outs_real, _ = dis(image=train_image, 
                           temperature=5.0, 
                           latency_to_accumulate=Variable(torch.Tensor([[0.0]]), requires_grad=True), 
                           supernet_or_sample=False)
        
        dis_loss = criterion(outs_real, train_label)
        
        dis_loss.backward()
        optimizer_dis.step()
        scheduler_dis.step()
        
        correct = (torch.argmax(outs_real, dim=1) == torch.argmax(train_label, dim=1)).sum().item()
        if (index % print_interval) == 0 and index != 0:
            print(f'Epo{epo+1}/Iter{index+1}:Accu {correct/len(train_label)}')

    dis.eval()
        
    Y_pred = np.array([])
    Y_test = np.array([])
    for image, label in test_loader:
        Y_test = np.append(Y_test, label, axis=None)
        image, label = image.view(image.shape[0], -1, image.shape[1]).cuda(non_blocking=True), label.cuda(non_blocking=True)

        predicted, _ = dis(image=image, 
                           temperature=5.0, 
                           latency_to_accumulate=Variable(torch.Tensor([[0.0]]), requires_grad=True), 
                           supernet_or_sample=False)
        predicted = predicted.view(label.shape[0], -1)
        predicted = torch.argmax(predicted, dim=1)
        
        Y_pred = np.append(Y_pred, predicted.cpu().numpy(), axis=None)
            
    accuracy_test = accuracy_score(Y_test, Y_pred)
    f1_test=f1_score(Y_test, Y_pred, average='weighted')
    prs_test=precision_score(Y_test, Y_pred, average='weighted')
    recall_test = recall_score(Y_test, Y_pred, average='weighted')
    
    if accuracy_test > best_accu:
        best_accu = accuracy_test
        best_epoch = epo
        best_pred = Y_pred
        ground_truth = Y_test
        #保存整个模型
        torch.save(dis, f'{save_path}/models/{model_kind}_Discriminator_{epo}_accu_{accuracy_test:.4f}.pth')
        np.save(f'{save_path}/models/best_pred_accu_{accuracy_test:.4f}.npy', best_pred)
        np.save(f'{save_path}/models/ground_truth_accu_{accuracy_test:.4f}.npy', ground_truth)
        
    print_lines = f'Epoch {epo+1}/{train_epoches}:\nAccuracy:{accuracy_test}\nF1-score:{f1_test}\nPrecision:{prs_test}\nRecall:{recall_test}\n'
    print_lines += f'Best Accuracy:{best_accu} - Epoch:{best_epoch+1}\n'
    print(print_lines)
    
    with open(f'{save_path}/results/{model_kind}_Multi.txt', 'a', encoding='utf-8') as file:
        # 将输出内容写入文件
        file.write(print_lines+'\n')