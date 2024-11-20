import os
import copy
import math
import time
import random
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from torch.autograd import Variable
from general_functions.utils import check_tensor_in_list
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
from fbnet_building_blocks.fbnet_modeldef import MODEL_ARCH
from general_functions.nsga import Individual, Population, fast_nondominated_sort

class TrainerSupernet:
    def __init__(self, logger, writer, run_time, lookup_table):        
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

        self.class_number_count          = [67343, 45927, 11656, 995, 52]
        class_ratio = [item/sum(self.class_number_count) for item in self.class_number_count]
        minus_log = [-math.log(item) for item in class_ratio]
        self.invert_ratio = [log/sum(minus_log) for log in minus_log]
        
    def train_loop(self, train_w_loader, train_thetas_loader, test_loader, generator, discriminator):
        
        best_top1 = 0.0
        
        # firstly, train weights only
        print("\nTrain only Weights from epochs 1 ~ %d\n" % (self.train_thetas_from_the_epoch))
        for epoch in range(self.train_thetas_from_the_epoch):
            # self.writer.add_scalar('learning_rate/weights', self.w_optimizer_gen.param_groups[0]['lr'], epoch)
            # self.writer.add_scalar('learning_rate/weights', self.w_optimizer_dis.param_groups[0]['lr'], epoch)
           
            self.logger.info("Weights training epoch %d" % (epoch+1))
            self._training_step(w_or_theta='w',
                                generator=generator, 
                                discriminator=discriminator, 
                                loader=train_w_loader,
                                test_loader=test_loader,
                                epoch=epoch, 
                                info_for_logger="_w_step_")
            # self.w_scheduler_gen.step()
            # self.w_scheduler_dis.step() 
            print()
            
        print("Train Weights & Theta from epochs %d ~ %d\n" % (self.train_thetas_from_the_epoch+1, self.cnt_epochs))
        for epoch in range(self.train_thetas_from_the_epoch, self.cnt_epochs):
            # self.writer.add_scalar('learning_rate/weights', self.w_optimizer_gen.param_groups[0]['lr'], epoch)
            # self.writer.add_scalar('learning_rate/weights', self.w_optimizer_dis.param_groups[0]['lr'], epoch)
            
            # self.writer.add_scalar('learning_rate/theta', self.theta_optimizer_gen.param_groups[0]['lr'], epoch)
            # self.writer.add_scalar('learning_rate/theta', self.theta_optimizer_dis.param_groups[0]['lr'], epoch)
            
            self.logger.info("Weights & Theta training epoch %d" % (epoch+1))
            self._training_step(w_or_theta='theta',
                                generator=generator, 
                                discriminator=discriminator, 
                                loader=train_w_loader,
                                test_loader=test_loader,
                                epoch=epoch, 
                                info_for_logger="_w_step_")
            # self.w_scheduler_gen.step()
            # self.w_scheduler_dis.step()
            
            # ---------------------------------------------------------
            # top1_avg = self._validate(model, test_loader, epoch)
            # ---------------------------------------------------------
            # top1_avg = self._validate(discriminator, test_loader, epoch)
            
            # if best_top1 < top1_avg:
            #     best_top1 = top1_avg
            #     self.logger.info("Best top1 acc by now. Save model")
            #     save(generator, self.path_to_save_model_gen)
            #     save(discriminator, self.path_to_save_model_dis)
            
            self.temperature = self.temperature * self.exp_anneal_rate
            print()
            
        save_path = os.path.join(self.path_to_save_model, str(self.run_time))
        for dis_individual in self.individuals_dis:
            model_dis, _, epoch_dis, iteration_dis, code_dis, objectives_dis = dis_individual.get_model_info()
            self._model_save(model_dis, 'Discriminator', epoch_dis, iteration_dis, code_dis, objectives_dis, save_path)
        
        for gen_individual in self.individuals_gen:
            model_gen, _, epoch_gen, iteration_gen, code_gen, objectives_gen = gen_individual.get_model_info()
            self._model_save(model_gen, 'Generator', epoch_gen, iteration_gen, code_gen, objectives_gen, save_path)
    
    def _training_step(self, w_or_theta, generator, discriminator, loader, test_loader, epoch, info_for_logger=""):
        generator = generator.train()                     
        last_epoch = -1
        
        # mutation_list = ['Minimax','Least-Squares','Hinge','Wasserstein']
        mutation_list = ['Minimax','Least-Squares']
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
        random_index = random.randint(0, int(len(test_loader) / CONFIG_SUPERNET['dataloading']['batch_size']))
        validate_image, validate_label = test_loader.dataset[random_index*CONFIG_SUPERNET['dataloading']['batch_size']: (random_index+1)*CONFIG_SUPERNET['dataloading']['batch_size']]
        validate_image = validate_image.view(CONFIG_SUPERNET['dataloading']['batch_size'], 1, -1).cuda(non_blocking=True)
        validate_label = validate_label.type(torch.int64).cuda(non_blocking=True)
        validate_label_one_hot = nn.functional.one_hot(validate_label, num_classes=self.class_num).to(torch.float32).cuda(non_blocking=True)

        iteration = 0
        if self.best_generator == None:
            self.best_generator = generator
        if self.best_discriminator_sample == None:
            self.best_generator_sample, _, _ = self._sample(generator, mode='Random', unique_name_of_arch='')
            
        for image, label in loader:
            batch_size = image.shape[0]
            
            image, label = image.view(batch_size, -1, image.shape[1]).cuda(non_blocking=True), label.cuda(non_blocking=True)
            real_label_one_hot = nn.functional.one_hot(label.to(torch.int64), num_classes=self.class_num).to(torch.float32).cuda(non_blocking=True)

            # fake_invaild_label = torch.tensor(np.zeros(batch_size)).to(int).cuda(non_blocking=True)
            fake_invaild_label = np.array([self._weighted_random_int() for i in range(batch_size)])
            fake_invaild_label = torch.from_numpy(fake_invaild_label).to(int).cuda(non_blocking=True)            
            fake_invaild_label_one_hot = nn.functional.one_hot(fake_invaild_label, num_classes=self.class_num).to(torch.float32).cuda(non_blocking=True)
            # fake_vaild_label = torch.tensor(np.ones(batch_size)).to(int).cuda(non_blocking=True)
            # fake_vaild_label_one_hot = nn.functional.one_hot(fake_vaild_label, num_classes=self.class_num).to(torch.float32).cuda(non_blocking=True)
            
            noise = torch.tensor(np.random.normal(0, 1, (batch_size, 1, 100)), dtype=torch.float32, requires_grad=False).cuda(non_blocking=True) 
            latency_to_accumulate_dis = Variable(torch.Tensor([[0.0]]), requires_grad=True).cuda(non_blocking=True)
            latency_to_accumulate_gen = Variable(torch.Tensor([[0.0]]), requires_grad=True).cuda(non_blocking=True)
            
            # 训练theta的时候采用sample的generator
            if w_or_theta=='theta':
                self.best_generator_sample.eval()
                with torch.no_grad():
                    fake_invaild_image = self.best_generator_sample(noise, fake_invaild_label_one_hot.view(batch_size, 1, -1), self.temperature, latency_to_accumulate_gen, False)                
            # 训练w的时候采用Supernet的generator
            else:
                self.best_generator.eval()
                with torch.no_grad():
                    fake_invaild_image, _ = self.best_generator(noise, fake_invaild_label_one_hot.view(batch_size, 1, -1), self.temperature, latency_to_accumulate_gen, True)
            
            fake_invaild_image = fake_invaild_image.view(batch_size, 1, -1).cuda(non_blocking=True)
            dis_accuracy_list = {}
            gen_accuracy_list = {}
            
            discriminator_population = []
            discriminator_population_latency = []
            discriminator_population_code = []
            
            generator_population = []
            generator_population_latency = []
            generator_population_code = []
                      
            # -----------------
            # Discriminator
            # -----------------   
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
                    
                outs_real_multi, _ = mutate_discriminator(image, self.temperature, latency_to_accumulate_dis, True)
                # outs_real = torch.argmax(outs_real_multi, dim=1)
                # outs_real = outs_real.type(torch.float32)
                # outs_real = outs_real.requires_grad_()
                outs_fake_invaild_multi, _ = mutate_discriminator(fake_invaild_image, self.temperature, latency_to_accumulate_dis, True)
                # outs_fake_invaild = torch.argmax(outs_fake_invaild_multi, dim=1)
                # outs_fake_invaild = outs_fake_invaild.type(torch.float32)
                # outs_fake_invaild = outs_fake_invaild.requires_grad_()
                if mutation=='Minimax':
                    dis_loss_real = torch.mean(torch.nn.CrossEntropyLoss()(torch.argmax(outs_real_multi, dim=1).type(torch.float32).requires_grad_(), 
                                                                           label.type(torch.float32))) 
                    dis_loss_fake = torch.mean(torch.nn.CrossEntropyLoss()(torch.argmax(outs_fake_invaild_multi, dim=1).type(torch.float32).requires_grad_(), 
                                                                           fake_invaild_label.type(torch.float32)))
                    dis_loss = (dis_loss_real + dis_loss_fake) / 2
                elif mutation=='Least-Squares':
                    dis_loss_real = torch.mean((outs_real_multi - real_label_one_hot) ** 2)
                    dis_loss_fake = torch.mean(outs_fake_invaild_multi ** 2)
                    dis_loss = dis_loss_real + dis_loss_fake
                elif mutation=='Hinge':
                    loss_real = F.relu(1 - outs_real_multi).mean()
                    loss_fake = F.relu(1 + outs_fake_invaild_multi).mean()
                    dis_loss = loss_real + loss_fake
                elif mutation=='Wasserstein':
                    loss_real =  -torch.mean(outs_real_multi)
                    loss_fake = torch.mean(outs_fake_invaild_multi)
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
                    optimizer_dis_theta.step()
                    scheduler_dis_theta.step()
                    
                discriminator_list[mutation] = mutate_discriminator
                 
                # 计算supernet的accuracy4), num_classes=2).to(torch.float32).view(n, 1, -1).cuda(non_blocking=True)
                # dis_accuracy_list[mutation] = self._accuracy(out, real)
                dis_accuracy_list[mutation] = self._accuracy(torch.cat((outs_real_multi, outs_fake_invaild_multi), dim=0), 
                                                             torch.cat((real_label_one_hot, fake_invaild_label_one_hot), dim=0))
                
                self.logger.info(f'Dis Iter:{iteration+1} {mutation}, Accu:{dis_accuracy_list[mutation]:.4f}')
                
                # ---------------------------
                # 评估采样的discriminator的性能
                # ---------------------------
                
                # Sample
                if w_or_theta=='theta':
                    # Max采样
                    discriminator_individual, discriminator_individual_latency, discriminator_individual_code = self._sample(mutate_discriminator, 
                                                                                                                             mode='Max', 
                                                                                                                             unique_name_of_arch='')
                    discriminator_population.append(discriminator_individual)                    
                    discriminator_population_latency.append(discriminator_individual_latency)
                    discriminator_population_code.append(discriminator_individual_code)
                    
                    # 只有Random采样
                    if epoch == CONFIG_SUPERNET['train_settings']['train_thetas_from_the_epoch'] and iteration == 0:
                        for i in range(self.sample_number-1):                        
                            random_discriminator_individual, random_discriminator_individual_latency, random_discriminator_individual_code = self._sample(mutate_discriminator, 
                                                                                                                                                          mode='Random', 
                                                                                                                                                          unique_name_of_arch='')                     
                            discriminator_population.append(random_discriminator_individual)                    
                            discriminator_population_latency.append(random_discriminator_individual_latency)
                            discriminator_population_code.append(random_discriminator_individual_code)
                    # 交叉+变异+Random采样
                    else:
                        cross_or_mutate = random.uniform(0, 1)
                        a, b = 0, 0
                        if len(self.individuals_dis) != 1:
                            # 随机选择交叉点a,b
                            while True:
                                a, b = sorted([random.randint(0,len(self.individuals_dis)-1), random.randint(0,len(self.individuals_dis)-1)])
                                if a != b:
                                    break
                        else:
                            cross_or_mutate = 1
                        # 交叉采样
                        if cross_or_mutate < 0.5:                            
                            cross_code1, cross_code2 = self._cross(self.individuals_dis[a],self.individuals_dis[b])
                            
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
                            mutate_code = self._mutate(self.individuals_dis[a])
                            mutate_dis, mudate_dis_latency = self._sample_from_code(model=mutate_discriminator, code=mutate_code, unique_name_of_arch='')
                            
                            discriminator_population.append(mutate_dis)                    
                            discriminator_population_latency.append(mudate_dis_latency)
                            discriminator_population_code.append(mutate_code)
                        # Random采样
                        for i in range(self.sample_number_dis):                        
                            random_discriminator_individual, random_discriminator_individual_latency, random_discriminator_individual_code = self._sample(mutate_discriminator, 
                                                                                                                                                          mode='Random', 
                                                                                                                                                          unique_name_of_arch='')
                            discriminator_population.append(random_discriminator_individual)                    
                            discriminator_population_latency.append(random_discriminator_individual_latency)
                            discriminator_population_code.append(random_discriminator_individual_code)
                    
                    for i, individual in enumerate(discriminator_population):    
                        individual.eval()
                        # Accuracy
                        with torch.no_grad():
                            val_output, _ = individual(validate_image, self.temperature, 0, False)
                        accuracy = self._accuracy(val_output, validate_label_one_hot)                        
                    
                        self.individuals_dis.append(Individual(model=individual, 
                                                      mutation=mutation, 
                                                      epoch=epoch, 
                                                      iteration=iteration,
                                                      code=discriminator_population_code[i],
                                                      objectives=[accuracy, pow(discriminator_population_latency[i], -1)]))
            if w_or_theta=='theta':
                
                population_dis = Population(individuals=self.individuals_dis)
                fast_nondominated_sort(population=population_dis) # 快速非支配排序
                                
                mutation_dis_count = {}
                for mutation in mutation_list:
                    mutation_dis_count[mutation] = 0                   
                for front in population_dis.fronts:                        
                    mutation_dis_count[front.mutation] += 1
                # 使用max()函数获取最大值的 key
                mutation_kind = max(mutation_dis_count, key=lambda x: mutation_dis_count[x])
                discriminator = discriminator_list[mutation_kind]
                
                individuals_accuracy_list = torch.tensor([individual.objectives[0] for individual in self.individuals_dis], dtype=torch.float32)
                self.best_discriminator_sample = self.individuals_dis[torch.argmax(individuals_accuracy_list)].get_model()
                
                self.sample_number_dis = max((self.sample_number - math.ceil(len(population_dis.fronts) * 1.5 / len(mutation_list))), 0)                
                self.individuals_dis = population_dis.fronts
                print(f'Discriminator Fronts Number:{len(self.individuals_dis)}')
                
            else:
                mutation_kind = max(dis_accuracy_list, key=lambda x: dis_accuracy_list[x])
                discriminator = discriminator_list[mutation_kind]
                self.best_discriminator = discriminator
            # ---------------------------------------------------
            # 需要完善Discriminator的演化：
            # 3、通过精英策略选择最佳discriminator --------- Done
            # 4、最佳discriminator的Supernet权重赋值给其余各Supernet，且保留最佳discriminator
            # ---------------------------------------------------            
            if w_or_theta=='theta':
                if (iteration + 1) % self.print_freq == 0 and iteration != 0:
                    vaild_start = time.time()
                    with torch.no_grad():
                        output_vaild = torch.empty(0).cuda(non_blocking=True)
                        labels_vaild = torch.empty(0).cuda(non_blocking=True)
                        for image, label in test_loader:                            
                            image, label = image.view(image.shape[0], -1, image.shape[1]).cuda(non_blocking=True), label.cuda(non_blocking=True)
                            out, _ = self.best_discriminator_sample(image, self.temperature, 0, False)
                            output_vaild = torch.cat((output_vaild, out), dim=0)
                            label_one_hot = nn.functional.one_hot(label.type(torch.int64), num_classes=self.class_num).to(torch.float32).cuda(non_blocking=True)
                            labels_vaild = torch.cat((labels_vaild, label_one_hot), dim=0)
                        
                        vaild_accu = self._accuracy(output_vaild, labels_vaild)
                    vaild_end = time.time()
                        
                    self._intermediate_stats_logging(model_name='Discriminator',
                                                    train_kind='theta',
                                                    accuracy=vaild_accu,
                                                    step=iteration,
                                                    epoch=epoch,
                                                    len_loader=len(loader),
                                                    chosen_mutation=mutation_kind,
                                                    time_consumption=vaild_end-vaild_start)
            else:
                if (iteration + 1) % self.print_freq == 0 and iteration != 0:
                    vaild_start = time.time()
                    with torch.no_grad():
                        output_vaild = torch.empty(0).cuda(non_blocking=True)
                        labels_vaild = torch.empty(0).cuda(non_blocking=True)
                        for image, label in test_loader:                            
                            image, label = image.view(image.shape[0], -1, image.shape[1]).cuda(non_blocking=True), label.cuda(non_blocking=True)
                            out, _ = self.best_discriminator(image, self.temperature, 0, True)
                            output_vaild = torch.cat((output_vaild, out), dim=0)
                            label_one_hot = nn.functional.one_hot(label.type(torch.int64), num_classes=self.class_num).to(torch.float32).cuda(non_blocking=True)
                            labels_vaild = torch.cat((labels_vaild, label_one_hot), dim=0)
                        
                        vaild_accu = self._accuracy(output_vaild, labels_vaild)
                    vaild_end = time.time()
                    
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
            if w_or_theta=='theta':
                self.best_discriminator_sample.eval()
            else:
                self.best_discriminator.eval()
            
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
                
                generate_fake_invaild_image, latency_to_accumulate_gen = mutate_generator(noise, fake_invaild_label_one_hot.view(batch_size, 1, -1), self.temperature, latency_to_accumulate_gen, True)
                generate_fake_invaild_image = generate_fake_invaild_image.view(batch_size, 1, -1).cuda(non_blocking=True)
                # generate_fake_vaild_image, latency_to_accumulate_gen = mutate_generator(noise, fake_vaild_label_one_hot.view(batch_size, 1, -1), self.temperature, latency_to_accumulate_gen, True)
                # generate_fake_vaild_image = generate_fake_vaild_image.view(batch_size, 1, -1).cuda(non_blocking=True)
                
                # 训练theta的时候采用sample的discriminator
                if w_or_theta=='theta':
                    fake_invaild_image_label, _ = self.best_discriminator_sample(generate_fake_invaild_image, 
                                                                                 self.temperature, 
                                                                                 Variable(torch.Tensor([[0.0]]), requires_grad=True).cuda(non_blocking=True), 
                                                                                 False)
                    # fake_vaild_image_label, _ = self.best_discriminator_sample(generate_fake_vaild_image, 
                    #                                                            self.temperature, 
                    #                                                            Variable(torch.Tensor([[0.0]]), requires_grad=True).cuda(non_blocking=True), 
                    #                                                            False)
                # 训练w的时候采用Supernet的discriminator
                else:
                    fake_invaild_image_label, _ = self.best_discriminator(generate_fake_invaild_image, 
                                                                          self.temperature, 
                                                                          Variable(torch.Tensor([[0.0]]), requires_grad=True).cuda(non_blocking=True), 
                                                                          True)
                    # fake_vaild_image_label, _ = self.best_discriminator(generate_fake_vaild_image, 
                    #                                                     self.temperature, 
                    #                                                     Variable(torch.Tensor([[0.0]]), requires_grad=True).cuda(non_blocking=True), 
                    #                                                     True)
                # fake_image_label, _ = discriminator(generate_fake_image, self.temperature, 0, True)
                
                if mutation=='Minimax':
                    gen_loss = torch.mean(torch.nn.CrossEntropyLoss()(fake_invaild_image_label, fake_invaild_label.long()))
                    # gen_fake_vaild_loss = torch.mean(torch.nn.CrossEntropyLoss()(fake_vaild_image_label, fake_vaild_label.long()))
                    # gen_loss = (gen_fake_invaild_loss + gen_fake_vaild_loss) / 2
                elif mutation=='Least-Squares':
                    gen_loss = torch.mean((fake_invaild_image_label - fake_invaild_label_one_hot) ** 2)
                    # gen_fake_vaild_loss = torch.mean((fake_vaild_image_label - fake_vaild_label_one_hot) ** 2)
                    # gen_loss = (gen_fake_invaild_loss + gen_fake_vaild_loss) / 2
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
                gen_accuracy_list[mutation] = self._accuracy(fake_invaild_image_label, fake_invaild_label_one_hot)
                
                self.logger.info(f'Gen Iter:{iteration+1} {mutation}, Accu:{gen_accuracy_list[mutation]:.4f}')
                # ---------------------------
                # 评估采样的generator的性能
                # ---------------------------
                if w_or_theta=='theta':
                    # Max采样
                    generator_individual, generator_individual_latency, generator_individual_code = self._sample(mutate_generator, mode='Max', unique_name_of_arch='')
                    generator_population.append(generator_individual)                    
                    generator_population_latency.append(generator_individual_latency)
                    generator_population_code.append(generator_individual_code)
                    
                    # 只有Random采样
                    if epoch == CONFIG_SUPERNET['train_settings']['train_thetas_from_the_epoch'] and iteration == 0:
                        for i in range(self.sample_number-1):                        
                            random_generator_individual, random_generator_individual_latency, random_generator_individual_code = self._sample(mutate_generator, 
                                                                                                                                              mode='Random', 
                                                                                                                                              unique_name_of_arch='')                     
                            generator_population.append(random_generator_individual)                    
                            generator_population_latency.append(random_generator_individual_latency)
                            generator_population_code.append(random_generator_individual_code)
                    # 交叉+变异+Random采样
                    else:
                        cross_or_mutate = random.uniform(0, 1)
                        if len(self.individuals_gen) != 1:
                            # 随机选择交叉点a,b
                            a, b = 0, 0
                            while True:
                                a, b = sorted([random.randint(0,len(self.individuals_gen)-1), random.randint(0,len(self.individuals_gen)-1)])
                                if a != b:
                                    break
                        else:
                            cross_or_mutate = 1
                        
                        # 交叉采样
                        if cross_or_mutate < 0.5:                            
                            cross_code1, cross_code2 = self._cross(self.individuals_gen[a],self.individuals_gen[b])
                            
                            cross_gen1, cross_gen1_latency = self._sample_from_code(model=mutate_generator, 
                                                                                    code=cross_code1, 
                                                                                    unique_name_of_arch='')
                            generator_population.append(cross_gen1)                    
                            generator_population_latency.append(cross_gen1_latency)
                            generator_population_code.append(cross_code1)
                            
                            cross_gen2, cross_gen2_latency = self._sample_from_code(model=mutate_generator, 
                                                                                    code=cross_code2, 
                                                                                    unique_name_of_arch='')
                            generator_population.append(cross_gen2)                    
                            generator_population_latency.append(cross_gen2_latency)
                            generator_population_code.append(cross_code2)
                        # 变异采样
                        else:
                            mutate_code = self._mutate(self.individuals_gen[a])
                            mutate_gen, mudate_gen_latency = self._sample_from_code(model=mutate_generator, 
                                                                                    code=mutate_code, 
                                                                                    unique_name_of_arch='')
                            
                            generator_population.append(mutate_gen)                    
                            generator_population_latency.append(mudate_gen_latency)
                            generator_population_code.append(mutate_code)
                        # Random采样
                        for i in range(self.sample_number_gen):                        
                            random_generator_individual, random_generator_individual_latency, random_generator_individual_code = self._sample(mutate_generator, 
                                                                                                                                              mode='Random', 
                                                                                                                                              unique_name_of_arch='')
                            generator_population.append(random_generator_individual)                    
                            generator_population_latency.append(random_generator_individual_latency)
                            generator_population_code.append(random_generator_individual_code)
                    
                    for i, individual in enumerate(generator_population):    
                        individual.eval()
                        
                        with torch.no_grad():
                            sampled_gen_images = individual(noise, fake_invaild_label_one_hot.view(batch_size, 1, -1), self.temperature, 0, False)

                        sampled_gen_images = sampled_gen_images.view(batch_size, 1, -1).cuda(non_blocking=True)
                        
                        # sampled_gen_images_output, pe_code_fake = self.best_discriminator(sampled_gen_images, self.temperature, 0, False)
                        sampled_gen_images_output, pe_code_fake = self.best_discriminator_sample(sampled_gen_images, 
                                                                                                 self.temperature, 
                                                                                                 Variable(torch.Tensor([[0.0]]), requires_grad=True).cuda(non_blocking=True), 
                                                                                                 False)
                        # Accuracy
                        accuracy = self._accuracy(sampled_gen_images_output, fake_invaild_label_one_hot)
                        # Diversity
                        inception_score = torch.nn.KLDivLoss(reduction='batchmean')(val_output, fake_invaild_label_one_hot)
                        # _, pe_code_validate = self.best_discriminator(validate_image, self.temperature, 0, False)
                        _, pe_code_validate = self.best_discriminator_sample(validate_image, 
                                                                             self.temperature, 
                                                                             Variable(torch.Tensor([[0.0]]), requires_grad=True).cuda(non_blocking=True), 
                                                                             False)
                        diversity = self._diversity(inception_score.cuda(non_blocking=True), pe_code_fake, pe_code_validate, validate_label)                     
                    
                        self.individuals_gen.append(Individual(model=individual, 
                                                      mutation=mutation, 
                                                      epoch=epoch, 
                                                      iteration=iteration,
                                                      code=generator_population_code[i],
                                                      objectives=[accuracy, pow(generator_population_latency[i], -1), diversity]))
                 
            if w_or_theta=='theta':
                population_gen = Population(individuals=self.individuals_gen)

                fast_nondominated_sort(population=population_gen)
                
                mutation_gen_count = {}
                for mutation in mutation_list:
                    mutation_gen_count[mutation] = 0                   
                for front in population_gen.fronts:                        
                    mutation_gen_count[front.mutation] += 1
                # 使用max()函数获取最大值的 key
                mutation_kind = max(mutation_gen_count, key=lambda x: mutation_gen_count[x])
                generator = generator_list[mutation_kind]
                
                individuals_accuracy_list = torch.tensor([individual.objectives[0] for individual in self.individuals_gen], dtype=torch.float32)
                self.best_generator_sample = self.individuals_gen[torch.argmax(individuals_accuracy_list)].get_model()
                
                self.sample_number_gen = max((self.sample_number - math.ceil(len(population_gen.fronts) * 1.5 / len(mutation_list))), 0)                
                self.individuals_gen = population_gen.fronts
                print(f'Generator Fronts Number:{len(self.individuals_gen)}')
            else:
                mutation_kind = max(gen_accuracy_list, key=lambda x: gen_accuracy_list[x])
                generator = generator_list[mutation_kind]
                self.best_generator = generator
            # ---------------------------------------------------
            # 需要完善Generator的演化：
            # 3、通过精英策略选择最佳generator
            # 4、最佳generator的Supernet权重赋值给其余各Supernet，且保留最佳generator
            # ---------------------------------------------------
            if w_or_theta=='theta':
                self._intermediate_stats_logging(model_name='Generator',
                                                 train_kind='theta',
                                                 accuracy=torch.max(individuals_accuracy_list).item(),
                                                 step=iteration,
                                                 epoch=epoch,
                                                 len_loader=len(loader),
                                                 chosen_mutation=mutation_kind,
                                                 time_consumption=0)
            else:
                self._intermediate_stats_logging(model_name='Generator',
                                                 train_kind='w',
                                                 accuracy=dis_accuracy_list[mutation_kind],
                                                 step=iteration,
                                                 epoch=epoch,
                                                 len_loader=len(loader),
                                                 chosen_mutation=mutation_kind,
                                                 time_consumption=0)
            print()
            iteration += 1
            # if w_or_theta == 'w':
            #     break
            # else:
            #     if iteration == 4:
            #         break
    # def _validate(self, model, loader, epoch):
    #     model.eval()
    #     start_time = time.time()

    #     with torch.no_grad():
    #         for step, (X, y) in enumerate(loader):
    #             X, y = X.view(X.shape[0], -1, X.shape[1]).cuda(), y.cuda()
    #             N = X.shape[0]
                
    #             latency_to_accumulate = torch.Tensor([[0.0]]).cuda()
    #             outs, latency_to_accumulate = model(X, self.temperature, latency_to_accumulate)
    #             # loss = self.criterion(outs, y, latency_to_accumulate, self.losses_ce, self.losses_lat, N)
    #             loss = self.criterion(outs, y.long())

    #             self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Valid")
                
    #     top1_avg = self.top1.get_avg()
    #     self._epoch_stats_logging(start_time=start_time, epoch=epoch, val_or_train='val')
    #     for avg in [self.top1, self.top3, self.losses]:
    #         avg.reset()
    #     return top1_avg
    def _compute_gradient_penalty(self, discriminator, real_samples, fake_samples, latency_to_accumulate_dis, supernet_or_sample=True):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
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
        return gradient_penalty
    
    def _accuracy(self, output, target):
        # target = target.view(target.shape[0], -1).cuda()
        with torch.no_grad():
            _, pred = output.topk(k=1, dim=1)
            _, real = target.topk(k=1, dim=1)
            correct = pred.eq(real).sum().item()
            accuracy = correct / target.size(0)
        return accuracy
    
    def _diversity(self, inception_score, pe_code_fake, pe_code_validate, validate_label):
        average_pe_fake = torch.mean(pe_code_fake, axis=0)
                
        average_pe_validate = torch.zeros(pe_code_validate[0].shape, dtype=torch.float32).cuda(non_blocking=True)
        for code, label in zip(pe_code_validate, validate_label):
            if not label:
                average_pe_validate += code
        average_pe_validate = average_pe_validate / sum(validate_label)
        
        diversity = (torch.sum(inception_score / pow(abs(average_pe_validate - average_pe_fake), 2))/np.prod(average_pe_validate.shape)).item()
        return diversity
    
    # def _epoch_stats_logging(self, start_time, epoch, iteration, val_or_train, info_for_logger=''):
    #     # self.writer.add_scalar('train_vs_val/'+val_or_train+'_loss'+info_for_logger, self.losses.get_avg(), epoch)
    #     # self.writer.add_scalar('train_vs_val/'+val_or_train+'_top1'+info_for_logger, self.top1.get_avg(), epoch)
    #     # self.writer.add_scalar('train_vs_val/'+val_or_train+'_top3'+info_for_logger, self.top3.get_avg(), epoch)
    #     # self.writer.add_scalar('train_vs_val/'+val_or_train+'_losses_lat'+info_for_logger, self.losses_lat.get_avg(), epoch)
    #     # self.writer.add_scalar('train_vs_val/'+val_or_train+'_losses_ce'+info_for_logger, self.losses_ce.get_avg(), epoch)
        
    #     top1_avg = self.top1.get_avg()
    #     self.logger.info(info_for_logger+val_or_train + ": [Epoch{:3d}/{}] Dis Prec {:.4%} Time {:.2f}".format(
    #         epoch+1, self.cnt_epochs, top1_avg, time.time() - start_time))
        
    def _intermediate_stats_logging(self, model_name, train_kind, accuracy, step, epoch, len_loader, chosen_mutation, time_consumption):        
        if (step > 0) and ((step + 1) % self.print_freq == 0) or step == len_loader - 1:
            self.logger.info(
                model_name +' '+ train_kind +
                ": Epo:[{}/{}], Iter:[{}/{}], Accuracy:{:.4f}, Mutation:{}, Time{:.4f}".format(
                epoch + 1, self.cnt_epochs, step + 1, len_loader, accuracy, chosen_mutation, time_consumption))
            # if train_kind == 'theta':
            #     self.logger.info(
            #         model_name + train_kind +
            #         ": Epoch {:3d}/{} Iter {:03d}/{:03d} Accuracy {:.3f} Mutation {}".format(
            #         epoch + 1, self.cnt_epochs, step + 1, len_loader, accuracy, chosen_mutation))
            # else:
            #     self.logger.info(
            #         model_name + train_kind +
            #         ": Epoch {:3d}/{} Iter {:03d}/{:03d} Accuracy {:.3f} Mutation {} ".format(
            #         epoch + 1, self.cnt_epochs, step + 1, len_loader, accuracy, chosen_mutation))
    
    def _sample(self, model, mode, unique_name_of_arch):

        sampled_model = copy.deepcopy(model)
        # print(f'---------------------Original Model:\n{sampled_model}\n')
        ops_names = [op_name for op_name in self.lookup_table.lookup_table_operations]
        cnt_ops = len(ops_names)

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
        
        # # --------------------------------------------------------------------
        # # 该writh_new_ARCH_to_fbnet_transformer_modeldef函数需要修改，无需写入文档
        # # --------------------------------------------------------------------
        # self._writh_ARCH_to_file(arch_operations, my_unique_name_for_ARCH=unique_name_of_arch)
        
        return sampled_model, sampled_latency, chosen_code
    
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
        
        # # --------------------------------------------------------------------
        # # 该writh_new_ARCH_to_fbnet_transformer_modeldef函数需要修改，无需写入文档
        # # --------------------------------------------------------------------
        # self._writh_ARCH_to_file(arch_operations, my_unique_name_for_ARCH=unique_name_of_arch)
        
        return sampled_model, sampled_latency
    
    def _cross(self, individual1, individual2):
        code1 = individual1.code
        code2 = individual2.code
        
        # 随机取start和end，且两数不相等
        start, end = 0, 0
        while start == end:
            start, end = sorted([random.randint(0,len(code1)-1), random.randint(0,len(code1)-1)])
            
        for i in range(start, end+1):
            code1[i], code2[i] = code2[i], code1[i]
        
        return code1, code2
    
    def _mutate(self, individual):
        code = individual.code
        
        mutate_index = random.randint(0,len(code)-1)
        # 当前第mutate_index上的值为current_num
        current_num = code[mutate_index]
        while current_num == code[mutate_index]:
            code[mutate_index] = random.randint(0, len(self.lookup_table.lookup_table_operations)-1)
        
        return code
    
    def _weighted_random_int(self):
        total = sum(self.invert_ratio)
        r = random.uniform(0, total)
        s = 0
        for i, w in enumerate(self.invert_ratio):
            s += w
            if r < s:
                return i
        
    def write_ARCH_to_file(ops_names, my_unique_name_for_ARCH):
    # assert len(ops_names) == 6 # For FBNet_Transformer

        if my_unique_name_for_ARCH in MODEL_ARCH:
            print("The specification with the name", my_unique_name_for_ARCH, "already written \
                to the fbnet_building_blocks.fbnet_modeldef. Please, create a new name \
                or delete the specification from fbnet_building_blocks.fbnet_modeldef (by hand)")
            assert my_unique_name_for_ARCH not in MODEL_ARCH
        
        ### create text to insert
        
        text_to_write = "    \"" + my_unique_name_for_ARCH + "\": {\n\
                \"block_op_type\": [\n"

        ops = ["[\"" + str(op) + "\"], " for op in ops_names]
        ops_lines = [op for op in ops]
        ops_lines = [''.join(line) for line in ops_lines]
        text_to_write += '            ' + '\n            '.join(ops_lines)

        e = [(op_name[-1] if op_name[-2] == 'e' else '1') for op_name in ops_names]
        e = [[op_name.split('_')[1].split('h')[1], 
            op_name.split('_')[2].split('f')[1],]\
            for op_name in ops_names]
        stages = ''
        for i,_e in enumerate(e):
            newline = '                    [[128, '+str(_e[0])+', '+str(int(128/int(_e[0])))+', '+str(e[0][1])+', 0.1]], # stage '+str(i+1)+'\n'  
            stages += newline
            # print(newline)
        # print(stages)
        
        text_to_write += "\n\
                ],\n\
                \"block_cfg\": {\n\
                    \"first\": [128, 16],\n\
                    \"stages\": [\n\
    "+stages+"\
                    ],\n\
                    \"backbone\": [num for num in range("+str(len(e)+1)+")],\n\
                },\n\
            },\n\
    }\
    "
        ### open file and find place to insert
        with open('./fbnet_building_blocks/fbnet_modeldef.py') as f1:
            lines = f1.readlines()
        end_of_MODEL_ARCH_id = next(i for i in reversed(range(len(lines))) if lines[i].strip() == '}')
        text_to_write = lines[:end_of_MODEL_ARCH_id] + [text_to_write]
        with open('./fbnet_building_blocks/fbnet_modeldef.py', 'w') as f2:
            f2.writelines(text_to_write)
            
    def _model_save(self, model, gen_or_dis, epoch, iteration, code, objectives, save_path):
        save_path = save_path+'/'+gen_or_dis+'/'+f'Epoch{epoch+1}'+'/'+f'Iter{iteration+1}'
        os.makedirs(save_path, exist_ok=True) # Create directories if needed, ignore if already exist
        
        objective = ''
        if gen_or_dis == 'Discriminator':
            objectives_name = ['Accuracy', 'Latency']
        elif gen_or_dis == 'Generator':
            objectives_name = ['Accuracy', 'Latency', 'Diversity']
        for obj, name in zip(objectives, objectives_name):
            objective += (name + ':' + str(obj) + ' ')
            
        model_name = str(code) + '-' + objective + '.pt'        
        torch.save(model, os.path.join(save_path, model_name))