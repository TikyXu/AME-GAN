import os
import math
import numpy as np
import pandas as pd

import torch

from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.compose import ColumnTransformer

from einops import repeat, pack
from einops.layers.torch import Rearrange

from datetime import datetime
from general_functions.dataloaders import get_nslkdd_train_loader, get_nslkdd_test_loader
from fbnet_building_blocks.fbnet_builder import PositionalEmbedding, MultiLayerPerceptron
from fbnet_building_blocks.fbnet_builder import PRIMITIVES
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
from supernet_functions.training_functions_supernet import CANDIDATE_BLOCKS_TRANSFORMER



os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device_dis = torch.device('cuda:0')
# device_gen = torch.device('cuda:0')

# 获取当前日期和时间
run_time = datetime.now()

save_path = os.path.join(f'ablation_study_results', str(run_time))
os.makedirs(save_path, exist_ok=True)
os.makedirs(f'{save_path}/BiLSTM', exist_ok=True)
os.makedirs(f'{save_path}/BiLSTM/models', exist_ok=True)
os.makedirs(f'{save_path}/BiLSTM/results', exist_ok=True)



def Load_NSLKDD_Old(path_train_data, path_test_data, batch_size):
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
        
    # 加载数据 train_num:125973, test_num:22544, total_data:148517
    data_train = pd.read_csv(path_train_data, header=None, names=column_names)
    data_test = pd.read_csv(path_test_data, header=None, names=column_names)
    total_data = pd.concat([data_train, data_test], axis=0) # 合并train和test
    train_num = len(data_train)
    # 删除Score列
    total_data = total_data.drop('score', axis=1)

    # 标签
    Y = total_data.iloc[:, -1]

    pdlist_class_dict = {}
    for index, data_class in enumerate([normal, dos, probe, r2l, u2l]):
        for item in data_class:
            pdlist_class_dict[item] = index

    # 给表格数据赋值
    Y = np.array([pdlist_class_dict[row] for row in Y])
    Y_train = Y[:train_num]
    Y_test = Y[train_num:]

    # 特征
    X = total_data.iloc[:, :-1]
    X_colums = list(X.columns)
    # 凑128形状，增加6列
    add_number = 6
    add_data = np.zeros((len(X), add_number))
    add_columns = [f'zero_{i}' for i in range(add_number)]
    new_columns = X_colums + add_columns
    X = pd.DataFrame(np.concatenate((np.array(X), add_data), axis=1), columns = new_columns)

    X_train = X.iloc[:train_num]
    X_test = X.iloc[train_num:]

    # 识别数值和分类特征
    categorical_features = ["protocol_type", "service", "flag"]
    numerical_features = list(set(X_train.columns) - set(categorical_features))

    # 构建预处理流水线
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    # 将特征转换应用到训练和测试集
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

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

def Load_NSLKDD(path_train_data, path_test_data, binary_or_multi='multi'):
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
    
    add_column_num = 6
    add_column = np.zeros(shape=(features.shape[0], add_column_num))
    features = np.concatenate((features, add_column), axis=1)

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
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_train = torch.LongTensor(Y_train)
    Y_test = torch.LongTensor(Y_test)
    
    return X_train, Y_train, X_test, Y_test

class MultiLayerPerceptron(nn.Module):
    def __init__(self, num_classes, dim, dropout):
        super().__init__()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Linear(dim, num_classes),
            nn.Dropout(dropout),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp_head(x)
    
class PositionalEmbedding(nn.Module):
    def __init__(self, *, seq_len, patch_size, dim, channels, emb_dropout = 0.):
        super().__init__()
        assert (seq_len % patch_size) == 0

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size
        # patch_dim = patch_size
        self.patch_dim = [patch_size, channels, patch_dim]
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size),
            # batch_size channels (patch_number * patch_size) -> batch_size patch_number (patch_size * channels)
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, series):
        x = self.to_patch_embedding(series)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)
        
        x, ps = pack([cls_tokens, x], 'b * d')

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        return x, ps



# 定义模型
class BiLSTM(nn.Module):
    def __init__(self, input_dim, num_classes, lstm_hidden_dim=64, lstm_layers=2, dropout=0.1):
        super(BiLSTM, self).__init__()
        
        self.bilstm = nn.LSTM(input_dim, lstm_hidden_dim, num_layers=lstm_layers, batch_first=True, bidirectional=True)

        self.embedding = nn.Linear(lstm_hidden_dim*2, input_dim)
        
        self.mlp = MultiLayerPerceptron(dim=input_dim,
                                        num_classes=num_classes,
                                        dropout=dropout)
    
    def forward(self, x):
        # LSTM部分
        x, _ = self.bilstm(x)  # x: (batch_size, seq_length, input_dim)
        x = x[:, -1, :]  # 取最后一个时间步的输出作为特征

        x = self.embedding(x)

        y = self.mlp(x)
        return y

class BiLSTMTransformer(nn.Module):    
    def __init__(self, input_dim, num_classes, lstm_hidden_dim=64, lstm_layers=2, nhead=4, dim_feedforward=128, num_layers=6, dropout=0.1):
        super(BiLSTMTransformer, self).__init__()
        
        self.bilstm = nn.LSTM(input_dim, lstm_hidden_dim, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.embedding = nn.Linear(lstm_hidden_dim*2, input_dim)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.mlp = MultiLayerPerceptron(dim=input_dim,
                                        num_classes=num_classes,
                                        dropout=dropout)
    
    def forward(self, x):
        # LSTM部分
        x, _ = self.bilstm(x)  # x: (batch_size, seq_length, input_dim)
        x = x[:, -1, :]  # 取最后一个时间步的输出作为特征
        x = self.embedding(x)

        x = self.transformer_encoder(x).squeeze(1)

        y = self.mlp(x)
        return y

train_epoches = 30
batch_size=256
transformer_block_dim = 128
last_epoch = -1
print_interval = 100

input_dim = 128
class_num = 5
lstm_layers = 2
nhead = 4
tb_layers = 6
dropout = 0.1

# model = BiLSTM(input_dim=input_dim, 
#                num_classes=class_num, 
#                lstm_hidden_dim=input_dim, 
#                lstm_layers=lstm_layers,
#                dropout=dropout).to(torch.device("cuda"))

model = BiLSTMTransformer(input_dim=input_dim, 
                          num_classes=class_num, 
                          lstm_hidden_dim=input_dim, 
                          lstm_layers=lstm_layers, 
                          nhead=nhead, 
                          dim_feedforward=input_dim, 
                          num_layers=tb_layers, 
                          dropout=dropout).to(torch.device("cuda"))
# model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer_model = Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler_model = CosineAnnealingLR(optimizer_model, T_max=train_epoches, last_epoch=last_epoch)



# 加载数据
path_train_data='datasets/NSL-KDD/KDDTrain+.txt'
path_test_data='datasets/NSL-KDD/KDDTest+.txt'

X_train, Y_train, X_test, Y_test = Load_NSLKDD(path_train_data=path_train_data, 
                                               path_test_data=path_test_data, 
                                               binary_or_multi='multi')# 装载数据到loader里面

train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)



best_accu = 0
best_epoch = 0
for epo in range(train_epoches):
    print(f'Training Epoch{epo+1}')
    
    model.train()
    for index, (train_image, train_label) in enumerate(train_loader):
        train_image, train_label = train_image.view(train_image.shape[0], -1, train_image.shape[1]).cuda(non_blocking=True), train_label.to(torch.int64).cuda(non_blocking=True)
        train_label = torch.nn.functional.one_hot(train_label, class_num).to(torch.float32)
        optimizer_model.zero_grad()
        
        outs_real = model(train_image).view(train_label.shape)
        
        dis_loss = criterion(outs_real, train_label)
        
        dis_loss.backward()
        optimizer_model.step()
        scheduler_model.step()
        
        correct = (torch.argmax(outs_real, dim=1) == torch.argmax(train_label, dim=1)).sum().item()
        if (index % print_interval) == 0 and index != 0:
            print(f'Epo{epo}/Iter{index}:Accu {correct/len(train_label)}')

    model.eval()    
    
    Y_pred = np.array([])
    Y_test = np.array([])
    for image, label in test_loader:
        Y_test = np.append(Y_test, label, axis=None)
        image, label = image.view(image.shape[0], -1, image.shape[1]).cuda(non_blocking=True), label.cuda(non_blocking=True)

        predicted = model(image).view(label.shape[0], -1)
        predicted = torch.argmax(predicted, dim=1)
        
        Y_pred = np.append(Y_pred, predicted.cpu().numpy(), axis=None)
            
    accuracy_test = accuracy_score(Y_test, Y_pred)
    
    if accuracy_test > best_accu:
        best_accu = accuracy_test
        best_epoch = epo
        best_pred = Y_pred
        
        #保存整个模型
        torch.save(model, f'{save_path}/BiLSTM/models/BiLSTM_{epo}_accu_{accuracy_test:.4f}.pth')
        
    print_lines = f'Epoch {epo+1}/{train_epoches}:\nAccuracy:{accuracy_test}\n'
    print_lines += f'Best Accuracy:{best_accu} - Epoch:{best_epoch+1}\n'
    print(print_lines)
    
    with open(f'{save_path}/BiLSTM/results/BiLSTM_Multi.txt', 'a', encoding='utf-8') as file:
        # 将输出内容写入文件
        file.write(print_lines+'\n')