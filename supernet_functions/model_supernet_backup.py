import torch
import einops
from torch import nn
from collections import OrderedDict

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from fbnet_building_blocks.fbnet_builder import PositionalEmbedding, TransformerBlockWithCrossAttention, MultiLayerPerceptron
from supernet_functions.config_for_supernet import CONFIG_SUPERNET

# class MixedOperation(nn.Module):
    
#     # Arguments:
#     # proposed_operations is a dictionary {operation_name : op_constructor}
#     # latency is a dictionary {operation_name : latency}
#     def __init__(self, layer_parameters, proposed_operations, latency):
#         super(MixedOperation, self).__init__()
#         ops_names = [op_name for op_name in proposed_operations]
        
#         self.ops = nn.ModuleList([proposed_operations[op_name](*layer_parameters)
#                                   for op_name in ops_names])
#         self.latency = [latency[op_name] for op_name in ops_names]
#         self.thetas = nn.Parameter(torch.Tensor([1.0 / len(ops_names) for i in range(len(ops_names))]))
    
#     def forward(self, x, temperature, latency_to_accumulate):
#         soft_mask_variables = nn.functional.gumbel_softmax(self.thetas, temperature)
#         output  = sum(m * op(x) for m, op in zip(soft_mask_variables, self.ops))
#         latency = sum(m * lat for m, lat in zip(soft_mask_variables, self.latency))
#         latency_to_accumulate = latency_to_accumulate + latency
#         return output, latency_to_accumulate

    
class MixedOperationTransformer(nn.Module):
    
    # Arguments:
    # proposed_operations is a dictionary {operation_name : op_constructor}
    # latency is a dictionary {operation_name : latency}
    def __init__(self, layer_parameters, proposed_operations, latency):
        super(MixedOperationTransformer, self).__init__()
        ops_names = [op_name for op_name in proposed_operations]
        
        self.ops = nn.ModuleList([proposed_operations[op_name](*layer_parameters)
                                  for op_name in ops_names])
        self.latency = [latency[op_name] for op_name in ops_names]
        self.thetas = nn.Parameter(torch.Tensor([1.0 / len(ops_names) for i in range(len(ops_names))]))
    
    def forward(self, x, temperature, latency_to_accumulate):
        soft_mask_variables = nn.functional.gumbel_softmax(self.thetas, temperature)
        output  = sum(m * op(x) for m, op in zip(soft_mask_variables, self.ops))
        latency = sum(m * lat for m, lat in zip(soft_mask_variables, self.latency))
        latency_to_accumulate = latency_to_accumulate + latency
        return output, latency_to_accumulate
    
    def set_theta(self):
        self.thetas = nn.Parameter(torch.ones(1))
        

class SuperNet_Generator(nn.Module):
    def __init__(self, lookup_table, device, cnt_classes=2):
        super(SuperNet_Generator, self).__init__()
        
        self.seq_len = CONFIG_SUPERNET['train_settings']['transformer_block_dim']
        self.patch_size = int(CONFIG_SUPERNET['train_settings']['transformer_block_dim']/8)
        self.dim = CONFIG_SUPERNET['train_settings']['transformer_block_dim']
        self.channels = 1
        self.emb_dropout = 0.1
        self.cnt_classes = cnt_classes
        self.z_dim = 100
        self.patch_number = int(self.dim / self.patch_size)
        self.search_space_number = len(lookup_table.lookup_table_operations)
        self.device = device
        
        self.epoch  = None
        self.iter = None
        self.code = None
        
        # Generator
        self.gen_first_1 = nn.Sequential( #[batch_size, patch_number+1, z_dim]
                                            nn.LayerNorm(self.z_dim),
                                            nn.Linear(self.z_dim, int(self.patch_number+1) * self.dim),            
                                            Rearrange('b c (n d) -> b n (d c)', d = self.dim),
                                            nn.LayerNorm(self.dim))
        
        self.gen_first_2 = nn.Sequential(
                                            nn.LayerNorm(self.cnt_classes),
                                            nn.Linear(self.cnt_classes, int(self.patch_number+1)*self.dim),            
                                            Rearrange('b c (n d) -> b n (d c)', d = self.dim),
                                            nn.LayerNorm(self.dim))

        self.cross_attention = TransformerBlockWithCrossAttention(dim=self.dim, heads=8, dim_head=16, mlp_dim=128, dropout=0.1)
        
        self.stages_to_search = nn.ModuleList([MixedOperationTransformer(
                                                   lookup_table.layers_parameters[layer_id],
                                                   lookup_table.lookup_table_operations,
                                                   lookup_table.lookup_table_latency[layer_id])
                                               for layer_id in range(lookup_table.cnt_layers)])
        
    def forward(self, noise, label, temperature, latency_to_accumulate, supernet_or_sample):
        # Generator SuperNet
        noise = noise.to(self.device)
        label = label.to(self.device)
        latency_to_accumulate = latency_to_accumulate
           
        noise = self.gen_first_1(noise)
        label = self.gen_first_2(label)
        
        y = self.cross_attention(noise, label)
        
        for mixed_op in self.stages_to_search:
            if supernet_or_sample:
                y, latency_to_accumulate = mixed_op(y, temperature, latency_to_accumulate)
            else:
                y = mixed_op(y)
        
        ps = [torch.Size([]), torch.Size([self.patch_number])]
        y, _ = unpack(y, ps, 'b * d')
        
        return (y, latency_to_accumulate) if supernet_or_sample else y
    

class SuperNet_Discriminator(nn.Module):
    def __init__(self, lookup_table, device, cnt_classes=2):
        super(SuperNet_Discriminator, self).__init__()
        
        self.seq_len = CONFIG_SUPERNET['train_settings']['transformer_block_dim']
        self.patch_size = int(CONFIG_SUPERNET['train_settings']['transformer_block_dim']/8)
        self.dim = CONFIG_SUPERNET['train_settings']['transformer_block_dim']
        self.channels = 1
        self.emb_dropout = 0.1
        self.cnt_classes = cnt_classes
        self.device = device
        
        self.epoch  = None
        self.iter = None
        self.code = None

        # Discriminator
        self.dis_first = PositionalEmbedding(seq_len=self.seq_len, 
                                             patch_size=self.patch_size, 
                                             dim=self.dim, 
                                             channels=self.channels, 
                                             emb_dropout=self.emb_dropout)
        self.stages_to_search = nn.ModuleList([MixedOperationTransformer(
                                                   lookup_table.layers_parameters[layer_id],
                                                   lookup_table.lookup_table_operations,
                                                   lookup_table.lookup_table_latency[layer_id])
                                               for layer_id in range(lookup_table.cnt_layers)])
        self.featurewise_embedding = nn.Sequential(nn.Flatten(),
                                                   nn.Linear((self.seq_len//self.patch_size+1)*self.dim, self.dim))
        self.dis_last_stages = MultiLayerPerceptron(num_classes=self.cnt_classes,
                                                    dim=self.dim,
                                                    dropout=0.1)
        
    def forward(self, image, temperature, latency_to_accumulate, supernet_or_sample): # supernet_or_sample: True=Supernet,False=Sample
        # Discriminator SuperNet
        image = image.to(self.device)
        latency_to_accumulate = latency_to_accumulate
        
        y, ps = self.dis_first(image)
        positional_embedding_code = y.clone()
        for mixed_op in self.stages_to_search:
            if supernet_or_sample:
                y, latency_to_accumulate = mixed_op(y, temperature, latency_to_accumulate)
            else:
                y = mixed_op(y)
        
        # cls_tokens, _ = einops.unpack(y, ps, 'b * d')
        # cls_tokens = y[:,0]
        cls_tokens = self.featurewise_embedding(y)
        y = self.dis_last_stages(cls_tokens)
        
        return y, (latency_to_accumulate if supernet_or_sample else positional_embedding_code)
    
    
class SupernetLoss(nn.Module):
    def __init__(self):
        super(SupernetLoss, self).__init__()
        self.alpha = CONFIG_SUPERNET['loss']['alpha']
        self.beta = CONFIG_SUPERNET['loss']['beta']
        self.weight_criterion = nn.CrossEntropyLoss()
    
    def forward(self, outs, targets, latency, losses_ce, losses_lat, N):
        
        ce = self.weight_criterion(outs, targets.long())
        lat = torch.log(latency ** self.beta)
        
        losses_ce.update(ce.item(), N)
        losses_lat.update(lat.item(), N)
        
        loss = self.alpha * ce * lat
        return loss #.unsqueeze(0)

