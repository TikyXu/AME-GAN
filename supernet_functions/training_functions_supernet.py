import os
import csv
import copy
import math
import time
import random
import datetime
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from scipy.special import softmax
from torch.autograd import Variable

from general_functions.utils import AverageMeter, save, accuracy, check_tensor_in_list, writh_new_ARCH_to_fbnet_transformer_modeldef
from general_functions.nsga import Individual, Population, fast_nondominated_sort
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
from supernet_functions.lookup_table_builder import CANDIDATE_BLOCKS_TRANSFORMER, SEARCH_SPACE_TRANSFORMER
from fbnet_building_blocks.fbnet_modeldef import MODEL_ARCH

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore", category=UserWarning, message="RNN module weights are not part of single contiguous chunk of memory")

class TrainerSupernet:
    
    def __init__(self, device_dis, device_gen, logger, writer, run_time, lookup_table):
        self.top1       = AverageMeter()
        self.top3       = AverageMeter()
        self.losses     = AverageMeter()
        self.losses_lat = AverageMeter()
        self.losses_ce  = AverageMeter()
        self.device_dis = device_dis
        self.device_gen = device_gen       
        self.logger       = logger
        self.writer       = writer
        self.run_time     = run_time
        self.lookup_table = lookup_table
        
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.sample_number = 6        
        self.sample_number_dis = 0
        self.sample_number_gen = 0
        self.individuals_dis = []
        self.individuals_gen = []
        
        self.best_discriminator = None
        self.best_discriminator_sample = None
        self.best_generator = None
        self.best_generator_sample = None
        
        self.temperature                 = CONFIG_SUPERNET['train_settings']['init_temperature']
        self.exp_anneal_rate             = CONFIG_SUPERNET['train_settings']['exp_anneal_rate'] # apply it every epoch        
        self.cnt_epochs                  = CONFIG_SUPERNET['train_settings']['cnt_epochs']
        self.train_thetas_from_the_epoch = CONFIG_SUPERNET['train_settings']['train_thetas_from_the_epoch']
        self.print_freq                  = CONFIG_SUPERNET['train_settings']['print_freq']
        self.path_to_save_model          = CONFIG_SUPERNET['train_settings']['path_to_save_model']        
        self.class_num                   = CONFIG_SUPERNET['train_settings']['class_num']

        self.save_model_path = os.path.join(self.path_to_save_model, str(self.run_time))
        
        self.class_number_count          = [67343, 45927, 11656, 995, 52]
        class_ratio = [item/sum(self.class_number_count) for item in self.class_number_count]
        minus_log = [-math.log(item) for item in class_ratio]
        self.invert_ratio = [log/sum(minus_log) for log in minus_log]

    def train_loop(self, train_loader, test_loader, generator, discriminator):        
        # firstly, train weights only
        self.logger.info("\nTrain only Weights from epochs 1 ~ %d\n" % (self.train_thetas_from_the_epoch))
        for epoch in range(self.train_thetas_from_the_epoch):
            # self.writer.add_scalar('learning_rate/weights', self.w_optimizer_gen.param_groups[0]['lr'], epoch)
            # self.writer.add_scalar('learning_rate/weights', self.w_optimizer_dis.param_groups[0]['lr'], epoch)
           
            self.logger.info("Weights training epoch %d" % (epoch+1))
            self._training_step(w_or_theta='w',
                                generator=generator, 
                                discriminator=discriminator, 
                                loader=train_loader,
                                test_loader=test_loader,
                                epoch=epoch, 
                                info_for_logger="_w_step_")
            # self.w_scheduler_gen.step()
            # self.w_scheduler_dis.step()
            generator = self.best_generator        
            discriminator = self.best_discriminator
            
            self.logger.info('')
            
        self.logger.info("Train Weights & Theta from epochs %d ~ %d\n" % (self.train_thetas_from_the_epoch+1, self.cnt_epochs))
        for epoch in range(self.train_thetas_from_the_epoch, self.cnt_epochs):
            # self.writer.add_scalar('learning_rate/weights', self.w_optimizer_gen.param_groups[0]['lr'], epoch)
            # self.writer.add_scalar('learning_rate/weights', self.w_optimizer_dis.param_groups[0]['lr'], epoch)
            
            # self.writer.add_scalar('learning_rate/theta', self.theta_optimizer_gen.param_groups[0]['lr'], epoch)
            # self.writer.add_scalar('learning_rate/theta', self.theta_optimizer_dis.param_groups[0]['lr'], epoch)
            
            self.logger.info("Weights & Theta training epoch %d" % (epoch+1))
            self._training_step(w_or_theta='theta',
                                generator=generator, 
                                discriminator=discriminator, 
                                loader=train_loader,
                                test_loader=test_loader,
                                epoch=epoch, 
                                info_for_logger="_w_step_")
            
            generator = self.best_generator
            discriminator = self.best_discriminator
            
            self.temperature = self.temperature * self.exp_anneal_rate            
            self.logger.info('')
            
        self.logger.info("Final Model\n")  
        for dis_individual in self.individuals_dis:
            mutation_dis, epoch_dis, iteration_dis, code_dis, objectives_dis = dis_individual.get_model_info()
            self.logger.info(f'Discriminator-{mutation_dis},Epo:{epoch_dis+1},Iter:{iteration_dis+1},Code:{code_dis},Obj:{objectives_dis}')
        
        for gen_individual in self.individuals_gen:
            mutation_gen, epoch_gen, iteration_gen, code_gen, objectives_gen = gen_individual.get_model_info()
            self.logger.info(f'Generator-{mutation_gen},Epo:{epoch_gen},Iter:{iteration_gen},Code:{code_gen},Obj:{objectives_gen}')
    
    def _training_step(self, w_or_theta, generator, discriminator, loader, test_loader, epoch, info_for_logger=""):
        generator = generator.train()                     
        last_epoch = -1
        
        # mutation_list = ['Minimax','Least-Squares','Hinge','Wasserstein']
        mutation_list = ['Minimax','Least-Squares','Hinge','Wasserstein']
        discriminator_list = {}
        thetas_params_dis_list = {}
        params_except_thetas_dis_list = {}
        w_optimizer_dis_list = {}
        w_scheduler_dis_list = {}
        theta_optimizer_dis_list = {}
        theta_scheduler_dis_list = {}        
        
        generator_list = {}
        thetas_params_gen_list = {}
        params_except_thetas_gen_list = {}
        w_optimizer_gen_list = {}
        w_scheduler_gen_list = {}
        theta_optimizer_gen_list = {}
        theta_scheduler_gen_list = {}
        
        # Sample得到的判别器的验证数据
        # random_index = random.randint(0, int(len(test_loader) / CONFIG_SUPERNET['dataloading']['batch_size']))

        # 生成一个随机的 random_state
        # np.random.seed(1337)
        random_state = np.random.randint(0, 1000)
        # print(f'RANDOM STATE -- {random_state}')
        test_image, test_label = test_loader.dataset[:]
        _, validate_image, _, validate_label = train_test_split(test_image, test_label, test_size=CONFIG_SUPERNET['train_settings']['test_size'], stratify=test_label, random_state=random_state)        
        validate_image = validate_image.view(validate_image.shape[0], 1, -1).cuda(non_blocking=True)
        validate_label = validate_label.type(torch.float32).cuda(non_blocking=True)
        # validate_label_one_hot = nn.functional.one_hot(validate_label, num_classes=self.class_num).to(torch.float32).view(validate_label.shape[0], 1, -1).cuda(non_blocking=True)

        iteration = 0
        if self.best_generator == None:
            self.best_generator = generator
        if self.best_discriminator_sample == None:
            self.best_generator_sample, _, _ = self._sample(generator, mode='Random', unique_name_of_arch='')
            
        for image, label in loader:
            start_time = time.time()
            batch_size = image.shape[0]
            
            image, label = image.view(batch_size, -1, image.shape[1]).cuda(non_blocking=True), label.cuda(non_blocking=True)
            real_label_one_hot = nn.functional.one_hot(label.to(torch.int64), num_classes=self.class_num).to(torch.float32).view(batch_size, 1, -1).cuda(non_blocking=True)
            
            fake_label = np.array([self._weighted_random_int() for i in range(batch_size)])
            fake_label = torch.from_numpy(fake_label).to(int).cuda(non_blocking=True)
            fake_label_one_hot = nn.functional.one_hot(fake_label, num_classes=self.class_num).to(torch.float32).view(batch_size, 1, -1).cuda(non_blocking=True)
            
            noise = torch.tensor(np.random.normal(0, 1, (batch_size, 1, 100)), dtype=torch.float32, requires_grad=False).cuda(non_blocking=True) 
            latency_to_accumulate_dis = Variable(torch.Tensor([[0.0]]), requires_grad=True).cuda(non_blocking=True)
            latency_to_accumulate_gen = Variable(torch.Tensor([[0.0]]), requires_grad=True).cuda(non_blocking=True)
            
            # 训练theta的时候采用sample的generator
            if w_or_theta=='theta':
                self.best_generator_sample.eval()
                with torch.no_grad():
                    fake_invaild_image = self.best_generator_sample(noise, fake_label_one_hot, self.temperature, latency_to_accumulate_gen, False)                
            # 训练w的时候采用Supernet的generator
            else:
                self.best_generator.eval()
                with torch.no_grad():
                    fake_invaild_image, _ = self.best_generator(noise, fake_label_one_hot, self.temperature, latency_to_accumulate_gen, True)
            
            fake_invaild_image = fake_invaild_image.view(batch_size, 1, -1).cuda(non_blocking=True)
            dis_accuracy_list = {}
            gen_accuracy_list = {}
            
            discriminator_population = []
            discriminator_population_latency = []
            discriminator_population_code = []
            
            generator_population = []
            generator_population_latency = []
            generator_population_code = []
            
            # for index, layer in enumerate(discriminator.module.stages_to_search): 
            #     print(f'--{index}-{layer.thetas.detach().cpu().numpy()} - SUM:{sum(layer.thetas.detach().cpu().numpy())}')
            # -----------------
            # Discriminator
            # -----------------  
            dis_max_sample_index = [] 
            for mutation in mutation_list:
                mutate_discriminator = copy.deepcopy(discriminator)
                
                thetas_params_dis_list[mutation] = [param for name, param in mutate_discriminator.named_parameters() if 'thetas' in name]
                params_except_thetas_dis_list[mutation] = [param for param in mutate_discriminator.parameters() if not check_tensor_in_list(param, thetas_params_dis_list[mutation])]

                w_optimizer_dis_list[mutation] = torch.optim.Adam(params=params_except_thetas_dis_list[mutation], 
                                                                  lr=CONFIG_SUPERNET['optimizer']['w_lr'],
                                                                  weight_decay=CONFIG_SUPERNET['optimizer']['w_weight_decay'])
                # Automatically optimizes learning rate
                w_scheduler_dis_list[mutation] = torch.optim.lr_scheduler.CosineAnnealingLR(w_optimizer_dis_list[mutation],
                                                                                            T_max=CONFIG_SUPERNET['train_settings']['cnt_epochs'],
                                                                                            last_epoch=last_epoch)
                theta_optimizer_dis_list[mutation] = torch.optim.Adam(params=thetas_params_dis_list[mutation], 
                                                                      lr=CONFIG_SUPERNET['optimizer']['thetas_lr'],
                                                                      weight_decay=CONFIG_SUPERNET['optimizer']['thetas_weight_decay'])
                # Automatically optimizes learning rate
                theta_scheduler_dis_list[mutation] = torch.optim.lr_scheduler.ExponentialLR(theta_optimizer_dis_list[mutation],
                                                                                            gamma=0.1,
                                                                                            last_epoch=last_epoch)
                
                mutate_discriminator.train()
                    
                optimizer_dis = w_optimizer_dis_list[mutation]
                # Automatically optimizes learning rate
                scheduler_dis = w_scheduler_dis_list[mutation]
                    
                if w_or_theta=='theta':
                    optimizer_dis_theta = theta_optimizer_dis_list[mutation]
                    # Automatically optimizes learning rate
                    scheduler_dis_theta = theta_scheduler_dis_list[mutation]
                
                optimizer_dis.zero_grad()                
                if w_or_theta=='theta':
                    optimizer_dis_theta.zero_grad()
                    
                outs_real, _ = mutate_discriminator(image, self.temperature, latency_to_accumulate_dis, True)
                outs_fake_invaild, _ = mutate_discriminator(fake_invaild_image, self.temperature, latency_to_accumulate_dis, True)

                if mutation=='Minimax':
                    dis_loss_real = torch.mean(torch.nn.CrossEntropyLoss()(outs_real, label.long())) 
                    dis_loss_fake = torch.mean(torch.nn.CrossEntropyLoss()(outs_fake_invaild, fake_label.long()))
                    dis_loss = (dis_loss_real + dis_loss_fake) / 2
                elif mutation=='Least-Squares':
                    dis_loss_real = torch.mean((outs_real - real_label_one_hot) ** 2)
                    dis_loss_fake = torch.mean(outs_fake_invaild ** 2)
                    dis_loss = dis_loss_real + dis_loss_fake
                elif mutation=='Hinge':
                    loss_real = F.relu(1 - outs_real).mean()
                    loss_fake = F.relu(1 + outs_fake_invaild).mean()
                    dis_loss = loss_real + loss_fake
                elif mutation=='Wasserstein':
                    loss_real =  -torch.mean(outs_real)
                    loss_fake = torch.mean(outs_fake_invaild)
                    gradient_penalty = self._compute_gradient_penalty(discriminator=mutate_discriminator, 
                                                                      real_samples=image,
                                                                      fake_samples=fake_invaild_image,
                                                                      latency_to_accumulate_dis=latency_to_accumulate_dis,
                                                                      supernet_or_sample=True)
                    lambda_gp = 10
                    dis_loss = loss_real + loss_fake + lambda_gp * gradient_penalty
                    
                
                dis_loss.backward(retain_graph=True)
                optimizer_dis.step() # 更新模型参数
                scheduler_dis.step() # 更新学习率
                
                if w_or_theta=='theta':
                    optimizer_dis_theta.step() # 更新模型参数
                    scheduler_dis_theta.step() # 更新学习率
                    
                discriminator_list[mutation] = mutate_discriminator
                 
                # 计算supernet的accuracy
                pred_multi_class = torch.cat((outs_real, outs_fake_invaild), dim=0)
                pred = torch.argmax(pred_multi_class, dim=1)
                target = torch.cat((label.to(torch.int64), fake_label), dim=0)

                dis_accuracy_list[mutation] = self._evaluation(pred=pred, target=target)
                if w_or_theta == 'w':
                    self.logger.info(f'Dis weights Iter:{iteration+1} {mutation}, Accu:{dis_accuracy_list[mutation]:.4f}')
                # ---------------------------
                # 评估采样的discriminator的性能
                # ---------------------------
                # Sample
                if w_or_theta=='theta':
                    
                    # 开始权重训练的第1个Epoch只有Random采样
                    if epoch == CONFIG_SUPERNET['train_settings']['train_thetas_from_the_epoch'] and iteration == 0:
                        for i in range(self.sample_number):                        
                            random_discriminator_individual, random_discriminator_individual_latency, random_discriminator_individual_code = self._sample(mutate_discriminator, mode='Random', unique_name_of_arch='')                     
                            discriminator_population.append(random_discriminator_individual)                    
                            discriminator_population_latency.append(random_discriminator_individual_latency)
                            discriminator_population_code.append(random_discriminator_individual_code)
                    else:
                        # Max采样
                        discriminator_individual, discriminator_individual_latency, discriminator_individual_code = self._sample(mutate_discriminator, mode='Max', unique_name_of_arch='')
                        discriminator_population.append(discriminator_individual)                    
                        discriminator_population_latency.append(discriminator_individual_latency)                                               
                        discriminator_population_code.append(discriminator_individual_code)

                        # print(f'Discriminator Max Sampling:{discriminator_individual_code} - [{len(discriminator_population)-1}]')
                        dis_max_sample_index.append((len(discriminator_population)-1))

                        # 交叉+变异+Random采样
                        cross_or_mutate = random.uniform(0, 1)
                        dis_a, dis_b = 0, 0
                        if len(self.individuals_dis) != 1:
                            # 随机选择交叉点a,b
                            while True:
                                dis_a, dis_b = sorted([random.randint(0,len(self.individuals_dis)-1), random.randint(0,len(self.individuals_dis)-1)])
                                if dis_a != dis_b:
                                    break
                        else:
                            cross_or_mutate = 1
                        # 交叉采样
                        if cross_or_mutate < 0.5:                            
                            cross_code1, cross_code2 = self._crossover(self.individuals_dis[dis_a],self.individuals_dis[dis_b])
                            
                            cross_dis1, cross_dis1_latency = self._sample_from_code(model=mutate_discriminator, code=cross_code1, unique_name_of_arch='')
                            discriminator_population.append(cross_dis1)                    
                            discriminator_population_latency.append(cross_dis1_latency)
                            discriminator_population_code.append(cross_code1)
                            
                            cross_dis2, cross_dis2_latency = self._sample_from_code(model=mutate_discriminator, code=cross_code2, unique_name_of_arch='')
                            discriminator_population.append(cross_dis2)                    
                            discriminator_population_latency.append(cross_dis2_latency)
                            discriminator_population_code.append(cross_code2)
                        # 变异采样
                        else:
                            # print(f'self.individuals_dis:{len(self.individuals_dis)},a={dis_a},b={dis_b}')
                            mutate_code = self._mutate(self.individuals_dis[dis_a])
                            mutate_dis, mudate_dis_latency = self._sample_from_code(model=mutate_discriminator, code=mutate_code, unique_name_of_arch='')
                            
                            discriminator_population.append(mutate_dis)                    
                            discriminator_population_latency.append(mudate_dis_latency)
                            discriminator_population_code.append(mutate_code)
                        # Random采样
                        for i in range(self.sample_number_dis):                        
                            random_discriminator_individual, random_discriminator_individual_latency, random_discriminator_individual_code = self._sample(mutate_discriminator, mode='Random', unique_name_of_arch='')
                            discriminator_population.append(random_discriminator_individual)                    
                            discriminator_population_latency.append(random_discriminator_individual_latency)
                            discriminator_population_code.append(random_discriminator_individual_code)
                    
            # 选择最佳sample_number_dis / sample_number
            if w_or_theta=='theta':
                
                # print(f'dis_max_sample_index:{dis_max_sample_index}')
                
                last_iteration_sampled_dis_number = len(self.individuals_dis)

                for i, individual in enumerate(discriminator_population):    
                    individual.eval()
                    # Accuracy
                    with torch.no_grad():
                        val_output, _ = individual(validate_image, self.temperature, 0, False)
                    accuracy_dis = self._accuracy(val_output, validate_label)
                    
                    self.individuals_dis.append(Individual(mutation=mutation, 
                                                            epoch=epoch, 
                                                            iteration=iteration,
                                                            code=discriminator_population_code[i],
                                                            objectives=[pow(discriminator_population_latency[i], -1), accuracy_dis]))
                                        
                    if i in dis_max_sample_index:
                        with open(f'logs/UNSW-NB15 Discriminator_MaxSample_Objectives {str(self.run_time)}.csv', 'a', newline='') as f:
                            writer = csv.writer(f)
    
                            # 如果是第一次写入文件，可以写入表头
                            # writer.writerow(['dis_accuracy', 'dis_inverse_latency'])  # 表头，只需写一次
                            # print(f'Max Sampling Discriminator: index[{i}],code:({discriminator_population_code[i]}):obj:[{pow(discriminator_population_latency[i], -1)}, {accuracy_dis}]')
                            
                            data = [pow(discriminator_population_latency[i], -1), accuracy_dis]  # 每次要写入的数据
                            writer.writerow(data)  # 写入数据

                population_dis = Population(individuals=self.individuals_dis)
                front_index_dis = fast_nondominated_sort(model_kind='Discriminator',
                                                         population=population_dis) # 快速非支配排序
                
                # 计算包含fronts最多的变异Discriminator
                mutation_dis_count = {}
                for mutation in mutation_list:
                    mutation_dis_count[mutation] = 0                   
                for front in population_dis.fronts:                        
                    mutation_dis_count[front.mutation] += 1
                
                for mutation in mutation_list:
                    self.logger.info(f'Dis theta Iter:{iteration+1} {mutation} Count:{mutation_dis_count[mutation]}')
                # 使用max()函数获取最大值的key
                mutation_kind = max(mutation_dis_count, key=lambda x: mutation_dis_count[x])                
                self.logger.info(f'Choose: {mutation_kind}')
                discriminator = discriminator_list[mutation_kind]
                
                code_count_dis = torch.zeros((len(SEARCH_SPACE_TRANSFORMER["input_shape"]), len(CANDIDATE_BLOCKS_TRANSFORMER)))

                for index, individual in zip(front_index_dis, population_dis.fronts):
                    if index >= last_iteration_sampled_dis_number:
                        self._model_save(model=discriminator_population[index-last_iteration_sampled_dis_number], 
                                         gen_or_dis='Discriminator', epoch=individual.epoch, 
                                         iteration=individual.iteration, mutation=individual.mutation, 
                                         code=individual.code, objectives=individual.objectives, 
                                         save_path=self.save_model_path)
                        if individual.mutation == mutation_kind:
                            for column, value in enumerate(individual.code):
                                code_count_dis[column, value] += 1.
                            
                # print(f'Code Count:\n{code_count_dis}')                
                
                # 获取 best_discriminator_sample
                individuals_accuracy_list = torch.tensor([individual.objectives[0] for individual in population_dis.fronts], dtype=torch.float32)
                dis_mutation, dis_epoch, dis_iteration, dis_code, dis_objectives = population_dis.fronts[torch.argmax(individuals_accuracy_list)].get_model_info()               
                self.best_discriminator_sample = self._model_load(gen_or_dis='Discriminator', epoch=dis_epoch, 
                                                                  iteration=dis_iteration, mutation=dis_mutation, 
                                                                  code=dis_code, objectives=dis_objectives, 
                                                                  save_path=self.save_model_path).to(self.device_dis)
                
                self.sample_number_dis = max((self.sample_number - math.ceil(len(population_dis.fronts) * 1.5 / len(mutation_list))), 0)                
                self.individuals_dis = population_dis.fronts

                # for i, indv_dis in enumerate(self.individuals_dis):
                #     print(f'Final Dis Front[{i}]:[{indv_dis.code}] - ({indv_dis.objectives})')
                
                self._theta_reward_update(model=discriminator,
                                          code_count=code_count_dis,
                                          lr=CONFIG_SUPERNET['optimizer']['thetas_lr'],
                                          device=self.device_dis)
                
                
                if (iteration + 1) % self.print_freq == 0 and iteration != 0 or (iteration + 1) == len(loader):
                    vaild_start = time.time()
                    with torch.no_grad():
                        output_vaild = torch.empty(0).cuda(non_blocking=True)
                        labels_vaild = torch.empty(0).cuda(non_blocking=True)
                        for image, label in test_loader:                            
                            image, label = image.view(image.shape[0], -1, image.shape[1]).cuda(non_blocking=True), label.cuda(non_blocking=True)
                            out, _ = self.best_discriminator_sample(image, self.temperature, 0, False)
                            output_vaild = torch.cat((output_vaild, out), dim=0)
                            label_one_hot = nn.functional.one_hot(label.type(torch.int64), num_classes=self.class_num).to(torch.float32).view(image.shape[0], 1, -1).cuda(non_blocking=True)
                            labels_vaild = torch.cat((labels_vaild, label_one_hot), dim=0)
                        
                        vaild_accu = self._accuracy(output_vaild, labels_vaild)
                    vaild_end = time.time()
                    
                    # 每print_freq输出一次_intermediate_stats_logging   
                    self._intermediate_stats_logging(model_name='Discriminator',
                                                    train_kind='theta',
                                                    accuracy=vaild_accu,
                                                    step=iteration,
                                                    epoch=epoch,
                                                    len_loader=len(loader),
                                                    chosen_mutation=mutation_kind,
                                                    time_consumption=vaild_end-vaild_start)
                
            else:
                mutation_kind = max(dis_accuracy_list, key=lambda x: dis_accuracy_list[x])                
                self.logger.info(f'Choose: {mutation_kind}')
                discriminator = discriminator_list[mutation_kind]
                self.best_discriminator = discriminator
            
            
                if (iteration + 1) % self.print_freq == 0 and iteration != 0 or (iteration + 1) == len(loader):
                    vaild_start = time.time()
                    with torch.no_grad():
                        output_vaild = torch.empty(0).cuda(non_blocking=True)
                        labels_vaild = torch.empty(0).cuda(non_blocking=True)
                        for image, label in test_loader:                            
                            image, label = image.view(image.shape[0], -1, image.shape[1]).cuda(non_blocking=True), label.cuda(non_blocking=True)
                            out, _ = self.best_discriminator(image, self.temperature, 0, True)
                            output_vaild = torch.cat((output_vaild, out), dim=0)
                            label_one_hot = nn.functional.one_hot(label.type(torch.int64), num_classes=self.class_num).to(torch.float32).view(image.shape[0], 1, -1).cuda(non_blocking=True)
                            labels_vaild = torch.cat((labels_vaild, label_one_hot), dim=0)
                        
                        vaild_accu = self._accuracy(output_vaild, labels_vaild)
                    vaild_end = time.time()
                    
                    # 每print_freq输出一次_intermediate_stats_logging
                    self._intermediate_stats_logging(model_name='Discriminator',
                                                    train_kind='w',
                                                    accuracy=vaild_accu,
                                                    step=iteration,
                                                    epoch=epoch,
                                                    len_loader=len(loader),
                                                    chosen_mutation=mutation_kind,
                                                    time_consumption=vaild_end-vaild_start)
            
            # -----------------
            # Generator
            # -----------------
            # for index, layer in enumerate(generator.module.stages_to_search): 
            #     print(f'-- Generator:[{index}]-{layer.thetas.detach().cpu().numpy()} - SUM:{sum(layer.thetas.detach().cpu().numpy())}')
            if w_or_theta=='theta':
                self.best_discriminator_sample.eval()
            else:
                self.best_discriminator.eval()
            
            gen_max_sample_index = []
            for mutation in mutation_list:
                mutate_generator = copy.deepcopy(generator)
                
                thetas_params_gen_list[mutation] = [param for name, param in mutate_generator.named_parameters() if 'thetas' in name]
                params_except_thetas_gen_list[mutation] = [param for param in mutate_generator.parameters() if not check_tensor_in_list(param, thetas_params_gen_list[mutation])]

                w_optimizer_gen_list[mutation] = torch.optim.Adam(params=params_except_thetas_gen_list[mutation], 
                                                                  lr=CONFIG_SUPERNET['optimizer']['w_lr'], 
                                                                  weight_decay=CONFIG_SUPERNET['optimizer']['w_weight_decay'])
                # Automatically optimizes learning rate
                w_scheduler_gen_list[mutation] = torch.optim.lr_scheduler.CosineAnnealingLR(w_optimizer_gen_list[mutation],
                                                                                            T_max=CONFIG_SUPERNET['train_settings']['cnt_epochs'],
                                                                                            last_epoch=last_epoch)
                theta_optimizer_gen_list[mutation] = torch.optim.Adam(params=thetas_params_gen_list[mutation],
                                                    lr=CONFIG_SUPERNET['optimizer']['thetas_lr'],
                                                    weight_decay=CONFIG_SUPERNET['optimizer']['thetas_weight_decay'])
                # Automatically optimizes learning rate
                theta_scheduler_gen_list[mutation] = torch.optim.lr_scheduler.ExponentialLR(theta_optimizer_gen_list[mutation],
                                                                                            gamma=0.1,
                                                                                            last_epoch=last_epoch)
                
                mutate_generator.train()
                
                optimizer_gen = w_optimizer_gen_list[mutation]
                # Automatically optimizes learning rate
                scheduler_gen = w_scheduler_gen_list[mutation]
                    
                if w_or_theta=='theta':
                    optimizer_gen_theta = theta_optimizer_gen_list[mutation]
                    # Automatically optimizes learning rate
                    scheduler_gen_theta = theta_scheduler_gen_list[mutation]
                
                optimizer_gen.zero_grad()
                
                if w_or_theta=='theta':
                    optimizer_gen_theta.zero_grad()
                
                generate_fake_invaild_image, latency_to_accumulate_gen = mutate_generator(noise, fake_label_one_hot, self.temperature, latency_to_accumulate_gen, True)
                generate_fake_invaild_image = generate_fake_invaild_image.view(batch_size, 1, -1).cuda(non_blocking=True)
                
                # 训练theta的时候采用sample的discriminator
                if w_or_theta=='theta':
                    self.best_discriminator_sample.train()
                    fake_invaild_image_label, _ = self.best_discriminator_sample(generate_fake_invaild_image, self.temperature, Variable(torch.Tensor([[0.0]]), requires_grad=True), False)
                # 训练w的时候采用Supernet的discriminator
                else:
                    self.best_discriminator.train()
                    fake_invaild_image_label, _ = self.best_discriminator(generate_fake_invaild_image, self.temperature, Variable(torch.Tensor([[0.0]]), requires_grad=True), True)
                
                if mutation=='Minimax':
                    gen_loss = torch.mean(torch.nn.CrossEntropyLoss()(fake_invaild_image_label, fake_label.long()))
                elif mutation=='Least-Squares':
                    gen_loss = torch.mean((fake_invaild_image_label - fake_label_one_hot) ** 2)
                elif mutation=='Hinge':
                    gen_loss = -fake_invaild_image_label.mean()
                elif mutation=='Wasserstein':
                    gen_loss = -torch.mean(fake_invaild_image_label)
                
                gen_loss.backward(retain_graph=True)
                optimizer_gen.step() # 更新模型参数
                scheduler_gen # 更新学习率
                if w_or_theta=='theta':
                    optimizer_gen_theta.step()
                    scheduler_gen_theta.step()
                    
                generator_list[mutation] = mutate_generator
                
                # 计算supernet的accuracy                
                pred = torch.argmax(fake_invaild_image_label, dim=1)
                target = fake_label
                
                gen_accuracy_list[mutation] = self._evaluation(pred=pred, target=target)
                if w_or_theta == 'w':
                    self.logger.info(f'Gen weights Iter:{iteration+1} {mutation}, Accu:{gen_accuracy_list[mutation]:.4f}')
                # ---------------------------
                # 评估采样的generator的性能
                # ---------------------------

                if w_or_theta=='theta':
                    # 开始权重训练的第1个Epoch只有Random采样
                    if epoch == CONFIG_SUPERNET['train_settings']['train_thetas_from_the_epoch'] and iteration == 0:
                        for i in range(self.sample_number-1):                        
                            random_generator_individual, random_generator_individual_latency, random_generator_individual_code = self._sample(mutate_generator, mode='Random', unique_name_of_arch='')                     
                            generator_population.append(random_generator_individual)                    
                            generator_population_latency.append(random_generator_individual_latency)
                            # print(f'Gen Random Code:{generator_individual_code}')
                            generator_population_code.append(random_generator_individual_code)
                    
                    else:
                        # Max采样
                        generator_individual, generator_individual_latency, generator_individual_code = self._sample(mutate_generator, mode='Max', unique_name_of_arch='')
                        generator_population.append(generator_individual)                    
                        generator_population_latency.append(generator_individual_latency)
                        generator_population_code.append(generator_individual_code)

                        # print(f'Generator Max Sampling:{generator_individual_code} - [{len(generator_population)-1}]')
                        gen_max_sample_index.append((len(generator_population)-1))

                        # 交叉+变异+Random采样
                        cross_or_mutate = random.uniform(0, 1)
                        gen_a, gen_b = 0, 0
                        if len(self.individuals_gen) != 1:
                            # 随机选择交叉点a,b
                            while True:
                                gen_a, gen_b = sorted([random.randint(0,len(self.individuals_gen)-1), random.randint(0,len(self.individuals_gen)-1)])
                                if gen_a != gen_b:
                                    break
                        else:
                            cross_or_mutate = 1
                        
                        # 交叉采样
                        if cross_or_mutate < 0.5:                            
                            cross_code1, cross_code2 = self._crossover(self.individuals_gen[gen_a],self.individuals_gen[gen_b])
                            
                            cross_gen1, cross_gen1_latency = self._sample_from_code(model=mutate_generator, code=cross_code1, unique_name_of_arch='')
                            generator_population.append(cross_gen1)                    
                            generator_population_latency.append(cross_gen1_latency)
                            generator_population_code.append(cross_code1)
                            
                            cross_gen2, cross_gen2_latency = self._sample_from_code(model=mutate_generator, code=cross_code2, unique_name_of_arch='')
                            generator_population.append(cross_gen2)                    
                            generator_population_latency.append(cross_gen2_latency)
                            generator_population_code.append(cross_code2)
                        # 变异采样
                        else:
                            # print(f'self.individuals_gen:{len(self.individuals_gen)},a={gen_a},b={gen_b}')
                            mutate_code = self._mutate(self.individuals_gen[gen_a])
                            mutate_gen, mudate_gen_latency = self._sample_from_code(model=mutate_generator, code=mutate_code, unique_name_of_arch='')
                            
                            generator_population.append(mutate_gen)                    
                            generator_population_latency.append(mudate_gen_latency)
                            generator_population_code.append(mutate_code)
                        # Random采样
                        for i in range(self.sample_number_gen):                        
                            random_generator_individual, random_generator_individual_latency, random_generator_individual_code = self._sample(mutate_generator, mode='Random', unique_name_of_arch='')
                            generator_population.append(random_generator_individual)                    
                            generator_population_latency.append(random_generator_individual_latency)
                            generator_population_code.append(random_generator_individual_code)            
            
            if w_or_theta=='theta':

                # print(f'gen_max_sample_index:{gen_max_sample_index}')
                
                last_iteration_sampled_gen_number = len(self.individuals_gen)
                
                for i, individual in enumerate(generator_population):    
                    individual.eval()
                    with torch.no_grad():
                        sampled_gen_images = individual(noise, fake_label_one_hot, self.temperature, 0, False)

                    sampled_gen_images = sampled_gen_images.view(batch_size, 1, -1).cuda(non_blocking=True)
                    
                    sampled_gen_images_output, pe_code_fake = self.best_discriminator_sample(sampled_gen_images, self.temperature, Variable(torch.Tensor([[0.0]]), requires_grad=True), False)
                    # Accuracy
                    accuracy_gen = self._accuracy(sampled_gen_images_output, fake_label)
                    # Diversity
                    inception_score = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(val_output, dim=0), fake_label_one_hot.view(fake_label_one_hot.shape[0], -1), reduction='batchmean') / fake_label_one_hot.shape[0]
                    _, pe_code_validate = self.best_discriminator_sample(validate_image, self.temperature, Variable(torch.Tensor([[0.0]]), requires_grad=True), False)
                    diversity = self._diversity(inception_score.cuda(non_blocking=True), pe_code_fake, pe_code_validate)                     

                    self.individuals_gen.append(Individual(mutation=mutation, 
                                                           epoch=epoch, 
                                                           iteration=iteration,
                                                           code=generator_population_code[i],
                                                           objectives=[diversity, accuracy_gen]))

                    if i in gen_max_sample_index:
                        with open(f'logs/UNSW-NB15 Generator_MaxSample_Objectives {str(self.run_time)}.csv', 'a', newline='') as f:
                    
                            writer = csv.writer(f)
    
                            # # 如果是第一次写入文件，可以写入表头
                            # writer.writerow(['gen_accuracy', 'gen_inverse_latency', 'gen_diversity'])  # 表头，只需写一次
                            # print(f'Max Sampling Generator: index[{i}],code:({generator_population_code[i]}):obj:[{pow(generator_population_latency[i], -1)}, {accuracy_gen}]')

                            # 这里可以是你的数据生成或获取逻辑
                            data = [diversity, accuracy_gen]  # 每次要写入的数据
                            writer.writerow(data)  # 写入数据

                population_gen = Population(individuals=self.individuals_gen)
                front_index_gen  = fast_nondominated_sort(model_kind='Generator',
                                                          population=population_gen)
                
                # 计算包含fronts最多的变异Generator
                mutation_gen_count = {}
                for mutation in mutation_list:
                    mutation_gen_count[mutation] = 0                   
                for front in population_gen.fronts:                        
                    mutation_gen_count[front.mutation] += 1
                
                for mutation in mutation_list:
                    self.logger.info(f'Gen theta Iter:{iteration+1} {mutation} Count:{mutation_gen_count[mutation]}')
                # 使用max()函数获取最大值的 key
                mutation_kind = max(mutation_gen_count, key=lambda x: mutation_gen_count[x])
                self.logger.info(f'Choose: {mutation_kind}')
                generator = generator_list[mutation_kind]
                
                code_count_gen = torch.zeros((len(SEARCH_SPACE_TRANSFORMER["input_shape"]), len(CANDIDATE_BLOCKS_TRANSFORMER)))
                for index, individual in zip(front_index_gen, population_gen.fronts):
                    if index >= last_iteration_sampled_gen_number:
                        self._model_save(model=generator_population[index-last_iteration_sampled_gen_number], 
                                         gen_or_dis='Generator', epoch=individual.epoch, 
                                         iteration=individual.iteration, mutation=individual.mutation, 
                                         code=individual.code, objectives=individual.objectives, 
                                         save_path=self.save_model_path)                        
                        if individual.mutation == mutation_kind:
                            for column, value in enumerate(individual.code):
                                code_count_gen[column, value] += 1.
                            
                # print(f'Code Count:\n{code_count_gen}')
                                                    
                individuals_accuracy_list = torch.tensor([individual.objectives[0] for individual in population_gen.fronts], dtype=torch.float32)                
                gen_mutation, gen_epoch, gen_iteration, gen_code, gen_objectives = population_gen.fronts[torch.argmax(individuals_accuracy_list)].get_model_info()
                self.best_generator_sample = self._model_load(gen_or_dis='Generator', epoch=gen_epoch, 
                                                              iteration=gen_iteration, mutation=gen_mutation, 
                                                              code=gen_code, objectives=gen_objectives, 
                                                              save_path=self.save_model_path).to(self.device_gen)
                
                self.sample_number_gen = max((self.sample_number - math.ceil(len(population_gen.fronts) * 1.5 / len(mutation_list))), 0)                
                self.individuals_gen = population_gen.fronts
                
                # for i, indv_gen in enumerate(self.individuals_gen):
                #     print(f'Final Gen Front[{i}]:[{indv_gen.code}] - ({indv_gen.objectives})')
                
                self._theta_reward_update(model=generator,
                                          code_count=code_count_gen,
                                          lr=CONFIG_SUPERNET['optimizer']['thetas_lr'],
                                          device=self.device_gen)
                
                # 每print_freq输出一次_intermediate_stats_logging
                self._intermediate_stats_logging(model_name='Generator',
                                                 train_kind='theta',
                                                 accuracy=torch.max(individuals_accuracy_list).item(),
                                                 step=iteration,
                                                 epoch=epoch,
                                                 len_loader=len(loader),
                                                 chosen_mutation=mutation_kind,
                                                 time_consumption=0)
            else:
                mutation_kind = max(gen_accuracy_list, key=lambda x: gen_accuracy_list[x])
                self.logger.info(f'Choose: {mutation_kind}')
                generator = generator_list[mutation_kind]
                self.best_generator = generator
            
                # 每print_freq输出一次_intermediate_stats_logging
                self._intermediate_stats_logging(model_name='Generator',
                                                 train_kind='w',
                                                 accuracy=dis_accuracy_list[mutation_kind],
                                                 step=iteration,
                                                 epoch=epoch,
                                                 len_loader=len(loader),
                                                 chosen_mutation=mutation_kind,
                                                 time_consumption=0)

            iteration += 1
            self.logger.info('')
            
            # if w_or_theta == 'w':
            #     if iteration == CONFIG_SUPERNET['train_settings']['print_freq']:
            #         break
            # else:
            #     if iteration == CONFIG_SUPERNET['train_settings']['print_freq']*2:
            #         break
    
    def _weighted_random_int(self):
        total = sum(self.invert_ratio)
        r = random.uniform(0, total)
        s = 0
        for i, w in enumerate(self.invert_ratio):
            s += w
            if r < s:
                return i
    
    def _compute_gradient_penalty(self, discriminator, real_samples, fake_samples, latency_to_accumulate_dis, supernet_or_sample=True):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        real_samples = real_samples.to(self.device_dis)
        fake_samples = fake_samples.to(self.device_dis)
        latency_to_accumulate_dis = latency_to_accumulate_dis.to(self.device_dis)
        
        cuda = True if torch.cuda.is_available() else False
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
        # Get random c between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates, _ = discriminator(interpolates, self.temperature, latency_to_accumulate_dis, True)
        # fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = torch.tensor(np.ones(real_samples.shape[0])).to(int).cuda(non_blocking=True)
        fake = nn.functional.one_hot(fake, num_classes=self.class_num).to(torch.float32).cuda(non_blocking=True)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return float(gradient_penalty)
    
    def _accuracy(self, output, target):
        # device = {'Discriminator':self.device_dis, 'Generator':self.device_gen}
        if len(target.shape) >= 2:
            target = target.reshape(target.shape[0], -1)
            target = torch.argmax(target, axis=1)

        output = torch.argmax(output, axis=1)
        # output = output.to(device[model_kind])
        # target = target.to(device[model_kind])
        
        with torch.no_grad():
            # _, pred = output.topk(k=1, dim=1)
            # _, real = target.view(target.shape[0], -1).topk(k=1, dim=1)
            # correct = pred.eq(real).sum().item()
            # accuracy = correct / target.size(0)

            accuracy = accuracy_score(output.cpu(), target.cpu())
        return accuracy
    
    def _calculate_confusion_matrix(self, num_class, pred, target):
        
        confusion_matrix = torch.zeros(size=(num_class, num_class)).cuda()
        for i in range(len(pred)):
            confusion_matrix[pred[i], target[i]] += 1        
        
        return confusion_matrix

    def _evaluation(self, pred, target):
        
        # device = {'Discriminator':self.device_dis, 'Generator':self.device_gen}
        # pred = pred.to(device[model_kind])
        # target = target.to(device[model_kind])
        
        # confusion_matrix = self._calculate_confusion_matrix(num_class, pred, target)
        # accuracy = sum(confusion_matrix.diag()) / len(pred)
        # precision = confusion_matrix.diag() / (confusion_matrix.sum(1) + 1e-8)
        # recall = confusion_matrix.diag() / (confusion_matrix.sum(0) + 1e-8)
        # f1_score = 2 * precision * recall / (precision + recall + 1e-8)

        pred = pred.cpu()
        target = target.cpu()

        accuracy = accuracy_score(pred, target)

        return accuracy
    
    def _diversity(self, inception_score, pe_code_fake, pe_code_validate):
        inception_score = inception_score.to(self.device_gen)
        pe_code_fake = pe_code_fake.to(self.device_gen)
        pe_code_validate = pe_code_validate.to(self.device_gen)
        # validate_label = validate_label.to(self.device_gen)
        
        average_pe_fake = torch.mean(pe_code_fake, axis=0)
        average_pe_validate = torch.mean(pe_code_validate, axis=0)
                
        # average_pe_validate = torch.zeros(pe_code_validate[0].shape, dtype=torch.float32).cuda(non_blocking=True).to(self.device_gen)
        # for code, label in zip(pe_code_validate, validate_label):
        #     if not label:
        #         average_pe_validate += code
        # average_pe_validate = average_pe_validate / sum(validate_label)
        
        # diversity = (torch.sum(torch.exp(inception_score) / pow(abs(average_pe_validate - average_pe_fake), 2))/np.prod(average_pe_validate.shape)).item()

        diversity = torch.exp(inception_score).item() / torch.sum(pow(average_pe_validate - average_pe_fake, 2)).item()

        return diversity
      
    def _intermediate_stats_logging(self, model_name, train_kind, accuracy, step, epoch, len_loader, chosen_mutation, time_consumption):        
        if (step > 0) and ((step + 1) % self.print_freq == 0) or step == len_loader - 1:
            self.logger.info(
                model_name +' '+ train_kind +
                ": Epo:[{}/{}], Iter:[{}/{}], Accuracy:{:.4f}, Mutation:{}, Time:{:.4f}".format(
                epoch + 1, self.cnt_epochs, step + 1, len_loader, accuracy, chosen_mutation, time_consumption))
    
    def _sample(self, model, mode, unique_name_of_arch):
        # print(f'{model.module.__class__.__name__}/{mode}')
        sampled_model = copy.deepcopy(model)
        # print(f'---------------------Original Model:\n{sampled_model}\n')
        ops_names = [op_name for op_name in self.lookup_table.lookup_table_operations]
        # cnt_ops = len(ops_names)

        sampled_latency = 0
        arch_operations=[]
        chosen_code = []
        
        for index, layer in enumerate(sampled_model.module.stages_to_search): 
            if mode == 'Max':
                optimal_ops_index = np.argmax(layer.thetas.detach().cpu().numpy())
            elif mode == 'Random':
                optimal_ops_index = np.random.randint(0, len(self.lookup_table.lookup_table_operations))
            
            chosen_code.append(optimal_ops_index)
            # Latency Calculation
            sampled_latency += self.lookup_table.lookup_table_latency[index][ops_names[optimal_ops_index]]
            # Operation Chosen
            arch_operations.append(layer.ops[optimal_ops_index])
            
        sampled_model.module.stages_to_search = nn.Sequential(*arch_operations)
        
        return sampled_model, sampled_latency, chosen_code
    
    def _theta_reward_update(self, model, code_count, lr, device):
        
        for layer, count in zip(model.module.stages_to_search, code_count):
            # print(f'-- Before:{sum(layer.thetas.detach().cpu().numpy())}\n{layer.thetas.detach().cpu().numpy()}')
            new_theta = np.array(layer.thetas.detach().cpu().numpy() + np.array(count * lr))
            # new_theta = self._norm(new_theta)        
            layer.thetas.data = torch.nn.Parameter(torch.from_numpy(new_theta).type(torch.float)).to(device)
            # print(f'--  After:{sum(layer.thetas.detach().cpu().numpy())}\n{layer.thetas.detach().cpu().numpy()}')
            
    def _norm(self, value):
        # print(f'-- Before:{sum(value)}\n{value}')
        row_sum = sum(value)
        for indice, item in enumerate(value):
            value[indice] = item/row_sum
        # print(f'--  After:\n{value}')
        return value
    
    def _sample_from_code(self, model, code, unique_name_of_arch):

        sampled_model = copy.deepcopy(model)
        # print(f'---------------------Original Model:\n{sampled_model}\n')
        ops_names = [op_name for op_name in self.lookup_table.lookup_table_operations]

        sampled_latency = 0
        arch_operations=[]
        
        for i, (index, layer) in enumerate(zip(code, sampled_model.module.stages_to_search)): 
            # Latency Calculation
            sampled_latency += self.lookup_table.lookup_table_latency[i][ops_names[index]]
            # Operation Chosen
            arch_operations.append(layer.ops[index])
            
        sampled_model.module.stages_to_search = nn.Sequential(*arch_operations)
        
        return sampled_model, sampled_latency
    
    def _crossover(self, individual1, individual2):
        code1 = copy.deepcopy(individual1.code)
        code2 = copy.deepcopy(individual2.code)
        
        # 随机取start和end，且两数不相等
        start, end = 0, 0
        while start == end:
            start, end = sorted([random.randint(0,len(code1)-1), random.randint(0,len(code1)-1)])
            
        for i in range(start, end+1):
            code1[i], code2[i] = code2[i], code1[i]
        
        return code1, code2
    
    def _mutate(self, individual):
        code = copy.deepcopy(individual.code)
        
        mutate_index = random.randint(0,len(code)-1)
        # 当前第mutate_index上的值为current_num
        current_num = code[mutate_index]
        while current_num == code[mutate_index]:
            code[mutate_index] = random.randint(0, len(self.lookup_table.lookup_table_operations)-1)
        
        return code
               
    def _model_save(self, model, gen_or_dis, epoch, iteration, mutation, code, objectives, save_path):
        save_path = save_path+'/'+gen_or_dis+'/'+f'Epoch{epoch+1}'+'/'+f'Iter{iteration+1}'+'/'+f'{mutation}'
        os.makedirs(save_path, exist_ok=True) # Create directories if needed, ignore if already exist
        
        objective = ''
        if gen_or_dis == 'Discriminator':
            objectives_name = ['Accuracy', 'Latency']
        elif gen_or_dis == 'Generator':
            objectives_name = ['Accuracy', 'Latency', 'Diversity']
        for obj, name in zip(objectives, objectives_name):
            objective += (name + ':' + str(obj) + ' ')
            
        model_name = str(code) + '-' + objective + '.pt'
        model_path = os.path.join(save_path, model_name)
        
        # print(f'-- Saving {gen_or_dis} - {model_path}')
            
        torch.save(model, model_path)
        
    def _model_load(self, gen_or_dis, epoch, iteration, mutation, code, objectives, save_path):
        save_path = save_path+'/'+gen_or_dis+'/'+f'Epoch{epoch+1}'+'/'+f'Iter{iteration+1}'+'/'+f'{mutation}'
        
        objective = ''
        if gen_or_dis == 'Discriminator':
            objectives_name = ['Accuracy', 'Latency']
        elif gen_or_dis == 'Generator':
            objectives_name = ['Accuracy', 'Latency', 'Diversity']
        for obj, name in zip(objectives, objectives_name):
            objective += (name + ':' + str(obj) + ' ')
            
        model_name = str(code) + '-' + objective + '.pt'
        model_path = os.path.join(save_path, model_name)

        model = torch.load(model_path)
        
        return model        