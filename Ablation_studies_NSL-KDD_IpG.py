import os
import math
import copy
import random
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

from einops import rearrange, repeat, pack, unpack

from datetime import datetime



os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device_dis = torch.device('cuda:0')
# device_gen = torch.device('cuda:0')

# 获取当前日期和时间
run_time = datetime.now()

save_path = os.path.join(f'ablation_study_results', str(run_time))
os.makedirs(save_path, exist_ok=True)
os.makedirs(f'{save_path}/Inversely-proportional_Generation', exist_ok=True)
os.makedirs(f'{save_path}/Inversely-proportional_Generation/models', exist_ok=True)
os.makedirs(f'{save_path}/Inversely-proportional_Generation/results', exist_ok=True)



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
    
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
                
        self.to_qkv1 = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_qkv2 = nn.Linear(dim, inner_dim * 3, bias = False)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, y):
        qkv1 = self.to_qkv1(x).chunk(3, dim = -1)
        _, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv1)

        qkv2 = self.to_qkv2(y).chunk(3, dim = -1)
        q2, _, _ = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv2)
        
        dots = torch.matmul(q2, k1.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    


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
    


class BiLSTMTransformerGenerator(nn.Module):    
    def __init__(self, z_dim, input_dim, num_classes, lstm_hidden_dim=64, lstm_layers=2, nhead=4, dim_feedforward=128, num_layers=6, dropout=0.1):
        super(BiLSTMTransformerGenerator, self).__init__()
        # Generator
        self.input_noise = nn.Sequential(nn.Linear(z_dim, input_dim),
                                         nn.LayerNorm(input_dim))
        
        self.input_label = nn.Sequential(nn.Linear(num_classes, input_dim),            
                                         nn.LayerNorm(input_dim))
        
        self.multiheadcrossattention = MultiHeadCrossAttention(dim=input_dim, 
                                                               heads=nhead, 
                                                               dim_head = int(input_dim/nhead), 
                                                               dropout = dropout)

        self.bilstm = nn.LSTM(input_dim, lstm_hidden_dim, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.embedding = nn.Linear(lstm_hidden_dim*2, input_dim)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.mlp = MultiLayerPerceptron(dim=input_dim,
                                        num_classes=input_dim,
                                        dropout=dropout)
    
    def forward(self, x, y):
        noise = self.input_noise(x)
        label = self.input_label(y)

        mhca = self.multiheadcrossattention(noise, label)

        # LSTM部分
        x, _ = self.bilstm(mhca)  # x: (batch_size, seq_length, input_dim)
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
z_dim = 100
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

dis = BiLSTMTransformer(
                          input_dim=input_dim, 
                          num_classes=class_num, 
                          lstm_hidden_dim=input_dim, 
                          lstm_layers=lstm_layers, 
                          nhead=nhead, 
                          dim_feedforward=input_dim, 
                          num_layers=tb_layers, 
                          dropout=dropout
                        ).to(torch.device("cuda"))

gen = BiLSTMTransformerGenerator(
                          z_dim=z_dim,
                          input_dim=input_dim, 
                          num_classes=class_num, 
                          lstm_hidden_dim=input_dim, 
                          lstm_layers=lstm_layers, 
                          nhead=nhead, 
                          dim_feedforward=input_dim, 
                          num_layers=tb_layers, 
                          dropout=dropout
                        ).to(torch.device("cuda"))

criterion = nn.CrossEntropyLoss()

optimizer_dis = Adam(params=dis.parameters(), lr=1e-4, weight_decay=1e-4)
optimizer_gen = Adam(params=gen.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler_dis = CosineAnnealingLR(optimizer_dis, T_max=train_epoches, last_epoch=last_epoch)
scheduler_gen = CosineAnnealingLR(optimizer_gen, T_max=train_epoches, last_epoch=last_epoch)



path_train_data='datasets/NSL-KDD/KDDTrain+.txt'
path_test_data='datasets/NSL-KDD/KDDTest+.txt'

X_train, Y_train, X_test, Y_test = Load_NSLKDD(path_train_data=path_train_data, 
                                               path_test_data=path_test_data, 
                                               binary_or_multi='multi')# 装载数据到loader里面

train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)



class_number_count = pd.DataFrame(Y_train).value_counts().values # 训练集数据Value Counts
class_ratio = [item/sum(class_number_count) for item in class_number_count] # 训练集数据各类占比
minus_log = [-math.log(item) for item in class_ratio] # 计算各类的倒数自然对数
invert_ratio = [log/sum(minus_log) for log in minus_log] # 计算各类的反比

def _weighted_random_int():
    total = sum(invert_ratio)
    r = random.uniform(0, total)
    s = 0
    for i, w in enumerate(invert_ratio):
        s += w
        if r < s:
            return i



best_accu = 0
best_epoch = 0
for epo in range(train_epoches):
    print(f'Training Epoch{epo+1}')    
    
    for index, (train_image, train_label) in enumerate(train_loader):
        train_image, train_label = train_image.view(train_image.shape[0], -1, train_image.shape[1]).cuda(non_blocking=True), train_label.to(torch.int64).cuda(non_blocking=True)
        train_label = torch.nn.functional.one_hot(train_label, class_num).to(torch.float32).cuda(non_blocking=True)
        optimizer_dis.zero_grad()
        #############
        # 训练判别器
        #############
        dis.train()
        outs_real = dis(train_image).view(train_label.shape)        
        dis_loss = criterion(outs_real, train_label)

        dis_loss.backward()
        optimizer_dis.step()
        scheduler_dis.step()

        #############
        # 训练生成器
        #############
        gen.train()
        # dis.eval()
        # 生成噪声
        noise = torch.randn(train_image.shape[0], 1, z_dim).cuda(non_blocking=True)
        
        # 生成反比例标签
        inversely_label = np.array([_weighted_random_int() for i in range(train_image.shape[0])])
        fake_label = torch.from_numpy(inversely_label).to(int).cuda(non_blocking=True)
        fake_label_one_hot = nn.functional.one_hot(fake_label, num_classes=class_num).to(torch.float32).view(train_image.shape[0], 1, -1).cuda(non_blocking=True)
            
        fake_sample = gen(noise, fake_label_one_hot).unsqueeze(1)
        # with torch.no_grad():
        #     generate_outs = dis(fake_sample).view(train_label.shape)
        generate_outs = dis(fake_sample).view(train_label.shape)
        gen_loss = criterion(generate_outs, fake_label_one_hot.squeeze(1))
        gen_loss.backward()
        optimizer_gen.step()
        scheduler_gen.step()
        
        
        accuracy = accuracy_score(torch.argmax(outs_real, dim=1).cpu(), torch.argmax(train_label, dim=1).cpu())
        if (index % print_interval) == 0 and index != 0:
            print(f'Epo{epo}/Iter{index} - Accu:{accuracy}')

    dis.eval()    
    Y_pred = np.array([])
    Y_test = np.array([])
    for image, label in test_loader:
        Y_test = np.append(Y_test, label, axis=None)
        image, label = image.view(image.shape[0], -1, image.shape[1]).cuda(non_blocking=True), label.cuda(non_blocking=True)

        predicted = dis(image).view(label.shape[0], -1)
        predicted = torch.argmax(predicted, dim=1)
        
        Y_pred = np.append(Y_pred, predicted.cpu().numpy(), axis=None)
            
    accuracy_test = accuracy_score(Y_test, Y_pred)
    
    if accuracy_test > best_accu:
        best_accu = accuracy_test
        best_epoch = epo
        best_pred = Y_pred
        
        #保存整个模型
        torch.save(dis, f'{save_path}/Inversely-proportional_Generation/models/BiLSTMTransformer_{epo}_accu_{accuracy_test:.4f}.pth')
        
    print_lines = f'Epoch {epo+1}/{train_epoches}:\nAccuracy:{accuracy_test}\n'
    print_lines += f'Best Accuracy:{best_accu} - Epoch:{best_epoch+1}\n'
    print(print_lines)
    
    with open(f'{save_path}/Inversely-proportional_Generation/results/Inversely-proportional_Generation_Multi.txt', 'a', encoding='utf-8') as file:
        # 将输出内容写入文件
        file.write(print_lines+'\n')