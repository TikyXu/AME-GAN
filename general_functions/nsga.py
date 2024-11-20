import torch
import functools
import numpy as np

from supernet_functions.lookup_table_builder import CANDIDATE_BLOCKS_TRANSFORMER, SEARCH_SPACE_TRANSFORMER

class Individual(object):
    """Represents one individual"""

    def __init__(self, mutation, epoch, iteration, code, objectives=[], dominated_count=0, dominated_solutions=[], rank=0, crowding_distance=0):
        self.mutation = mutation
        self.epoch = epoch
        self.iteration = iteration
        self.code = code
        self.objectives = objectives        
        self.dominated_count = dominated_count
        self.dominated_solutions = dominated_solutions
        self.rank = rank
        self.crowding_distance = crowding_distance
    
    def get_search_space_number(self):
        return self.search_space_number
        
    def dominates(self, objective):
        dominated = True
        if self.objectives != objective:
            for item1, item2 in zip(self.objectives, objective):
                if item1 < item2:
                    dominated = False
        else:
            dominated = False
        # print(f'{self.objectives}vs{objective} -- Dominated:{dominated}')
        return dominated
    
    def get_model_info(self):
        return self.mutation, self.epoch, self.iteration, self.code, self.objectives
    
       
class Population(object):
    """Represents population - a group of Individuals,
    can merge with another population"""
    
    def __init__(self, individuals=None):
        self.population = individuals
        self.fronts = []
        
    def __len__(self):
        return len(self.population)
        
    def __iter__(self):
        """Allows for iterating over Individuals"""
        
        return self.population.__iter__()
        
    def extend(self, individual):
        """Creates new population that consists of
        old individuals ans new_individuals"""
        
        self.population.extend(individual)   
           
        
def fast_nondominated_sort(model_kind, population):
    population.fronts = []
    front_index = []

    # print(f'{model_kind} new individuals:')
    for index, individual in enumerate(population):
        individual.dominated_count = 0
        # print(f'{model_kind} Indv[{index}]:[{individual.code}] - ({individual.objectives})')
        for other_individual in population:
            if individual.dominates(other_individual.objectives) == True:
                individual.dominated_solutions.append(other_individual)
            elif other_individual.dominates(individual.objectives) == True:
                individual.dominated_count += 1
        
        # print(f'Indiv[{index}]:{individual.objectives},Dominated Count:{individual.dominated_count}')
                
        if individual.dominated_count == 0:
            population.fronts.append(individual)
            
            individual.rank = 0
            front_index.append(index)

    # for i, ind in zip(front_index, population.fronts):
    #     print(f'{model_kind}--Front[{i}]:[{ind.code}] - ({ind.objectives})')

    return front_index


def test():
    accuracy = torch.tensor([1, 1, 1, 2, 2, 3, 3, 4], dtype=torch.float32)
    latency = torch.tensor([2, 3, 4, 1, 2, 2, 3, 1], dtype=torch.float32)

    individuals = []

    for i in range(len(accuracy)):
        individuals.append(Individual(objectives=[accuracy[i], latency[i]]))

    population = Population(individuals=individuals)

    fast_nondominated_sort(population=population)

    print(f'\nFront position:')
    for individual in population.fronts:
        print(f'{individual.objectives}')

# calculate_crowding_distance(population.fronts)


    
