import os
import torch
from torch import nn
import numpy as np
from tensorboardX import SummaryWriter
import argparse

from torchinfo import summary

from general_functions.dataloaders import get_nslkdd_train_loader, get_nslkdd_test_loader, get_loaders, get_test_loader
from general_functions.utils import get_logger, weights_init, create_directories_from_list
import fbnet_building_blocks.fbnet_builder as fbnet_builder
from architecture_functions.training_functions import TrainerArch
from architecture_functions.config_for_arch import CONFIG_ARCH

parser = argparse.ArgumentParser("architecture")

parser.add_argument('--architecture_name', type=str, default='fbnet_transformer_binary_tb8_3', help='You can choose architecture from the fbnet_building_blocks/fbnet_modeldef.py')
args = parser.parse_args()

def main():
    manual_seed = 1
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.backends.cudnn.benchmark = True
    
    # Making Sure the Folders Existed
    for path in [CONFIG_ARCH['logging']['path_to_tensorboard_logs'],]:    
        print(path)    
        if not os.path.exists(path):
            # pass
            os.makedirs(path)
    
    create_directories_from_list([CONFIG_ARCH['logging']['path_to_tensorboard_logs']])
    
    logger = get_logger(CONFIG_ARCH['logging']['path_to_log_file'])
    writer = SummaryWriter(log_dir=CONFIG_ARCH['logging']['path_to_tensorboard_logs'])

    #### DataLoading
    # train_loader = get_loaders(1.0, CONFIG_ARCH['dataloading']['batch_size'],
    #                            CONFIG_ARCH['dataloading']['path_to_save_data'],
    #                           )
    # valid_loader = get_test_loader(CONFIG_ARCH['dataloading']['batch_size'],
    #                                CONFIG_ARCH['dataloading']['path_to_save_data'])
    
    train_loader, _ = get_nslkdd_train_loader(train_portion=CONFIG_ARCH['dataloading']['train_portion'],
                                              batch_size=CONFIG_ARCH['dataloading']['batch_size'],
                                              path_to_save_data=CONFIG_ARCH['dataloading']['path_to_save_data']+'/NSL-KDD/KDDTrain+.txt',
                                              binary_or_multi='binary'
                                             )
    valid_loader = get_nslkdd_test_loader(batch_size=CONFIG_ARCH['dataloading']['batch_size'],
                                          path_to_save_data=CONFIG_ARCH['dataloading']['path_to_save_data']+'/NSL-KDD/KDDTest+.txt',
                                          binary_or_multi='binary'
                                         )
    
    #### Model
    arch = args.architecture_name
    model = fbnet_builder.get_model(arch, cnt_classes=2).cuda()
    model = model.apply(weights_init)
    model = nn.DataParallel(model, [0])
    # summary(model, input_size=(1, 3, 32, 32))

    #### Loss and Optimizer
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=CONFIG_ARCH['optimizer']['lr'],
                                momentum=CONFIG_ARCH['optimizer']['momentum'],
                                weight_decay=CONFIG_ARCH['optimizer']['weight_decay'])
    criterion = nn.CrossEntropyLoss().cuda()
    
    #### Scheduler
    if CONFIG_ARCH['train_settings']['scheduler'] == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=CONFIG_ARCH['train_settings']['milestones'],
                                                    gamma=CONFIG_ARCH['train_settings']['lr_decay'])  
    elif CONFIG_ARCH['train_settings']['scheduler'] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=CONFIG_ARCH['train_settings']['cnt_epochs'],
                                                               eta_min=0.001, last_epoch=-1)
    else:
        logger.info("Please, specify scheduler in architecture_functions/config_for_arch")
        
    
    #### Training Loop
    trainer = TrainerArch(criterion, optimizer, scheduler, logger, writer)
    trainer.train_loop(train_loader, valid_loader, model) 
    
if __name__ == "__main__":
    main()