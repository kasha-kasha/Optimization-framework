import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import random
import matplotlib
# 设置中文字体和绘图参数
matplotlib.use('TkAgg')
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class Individual:
    def __init__(self, variables):
        self.variables = variables
        self.objectives = None
        self.rank = None
        self.crowding_distance = 0.0
        self.domination_count = 0
        self.dominated_solutions = []

class NSGA2:
    def __init__(self, n_vars=30, pop_size=100, max_gen=300, cx_prob=0.8, mut_prob=0.1):
        self.n_vars = n_vars
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        
    def dominates(self, ind1, ind2):
        """判断ind1是否支配ind2"""
        obj1, obj2 = ind1.objectives, ind2.objectives
        better = any(obj1[i] < obj2[i] for i in range(len(obj1)))
        worse = any(obj1[i] > obj2[i] for i in range(len(obj1)))
        return better and not worse
    
    def fast_non_dominated_sort(self, population):
        """快速非支配排序"""
        fronts = [[]]
        
        for p in population:
            p.domination_count = 0
            p.dominated_solutions = []
            
            for q in population:
                if self.dominates(p, q):
                    p.dominated_solutions.append(q)
                elif self.dominates(q, p):
                    p.domination_count += 1
            
            if p.domination_count == 0:
                p.rank = 1
                fronts[0].append(p)
        
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 2
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        
        return fronts[:-1]
    
    def calculate_crowding_distance(self, front):
        """计算拥挤距离"""
        if len(front) <= 2:
            for ind in front:
                ind.crowding_distance = float('inf')
            return
        
        for ind in front:
            ind.crowding_distance = 0
        
        for obj_idx in range(2):  # 2个目标函数
            front.sort(key=lambda x: x.objectives[obj_idx])
            front[0].crowding_distance = front[-1].crowding_distance = float('inf')
            
            obj_range = front[-1].objectives[obj_idx] - front[0].objectives[obj_idx]
            if obj_range > 0:
                for i in range(1, len(front) - 1):
                    distance = (front[i + 1].objectives[obj_idx] - 
                              front[i - 1].objectives[obj_idx]) / obj_range
                    front[i].crowding_distance += distance
    
    def tournament_selection(self, population):
        """锦标赛选择"""
        tournament = random.sample(population, 2)
        return min(tournament, key=lambda x: (x.rank, -x.crowding_distance))
    
    def crossover(self, parent1, parent2):
        """模拟二进制交叉"""
        if random.random() > self.cx_prob:
            return parent1, parent2
        
        child1_vars = parent1.variables.copy()
        child2_vars = parent2.variables.copy()
        
        for i in range(self.n_vars):
            if random.random() <= 0.5:
                if abs(parent1.variables[i] - parent2.variables[i]) > 1e-14:
                    y1, y2 = sorted([parent1.variables[i], parent2.variables[i]])
                    rand = random.random()
                    beta = (2 * rand) ** (1/21) if rand <= 0.5 else (1/(2*(1-rand))) ** (1/21)
                    
                    c1 = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                    c2 = 0.5 * ((y1 + y2) + beta * (y2 - y1))
                    
                    child1_vars[i] = max(0, min(1, c1))
                    child2_vars[i] = max(0, min(1, c2))
        
        child1 = Individual(child1_vars)
        child2 = Individual(child2_vars)
        child1.objectives = self.evaluate_zdt1(child1_vars)
        child2.objectives = self.evaluate_zdt1(child2_vars)
        return child1, child2
    
    def mutation(self, individual):
        """多项式变异"""
        if random.random() > self.mut_prob:
            return individual
        
        mutated_vars = individual.variables.copy()
        eta_m = 20  # 分布指数
        
        for i in range(self.n_vars):
            if random.random() <= 1.0/self.n_vars:
                y = mutated_vars[i]
                yl, yu = 0.0, 1.0  # 变量边界
                
                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)
                
                rand = random.random()
                mut_pow = 1.0 / (eta_m + 1)
                
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1))
                    deltaq = 1.0 - (val ** mut_pow)
                
                y = y + deltaq * (yu - yl)
                mutated_vars[i] = max(yl, min(yu, y))
        
        mutated = Individual(mutated_vars)
        mutated.objectives = self.evaluate_zdt1(mutated_vars)
        return mutated
    
    def evaluate_zdt1(self, variables):
        """ZDT1目标函数"""
        f1 = variables[0]
        g = 1 + 9 * np.sum(variables[1:]) / (self.n_vars - 1)
        f2 = g * (1 - np.sqrt(f1 / g))
        return np.array([f1, f2])
    
    def initialize_population(self):
        """初始化种群"""
        population = []
        for _ in range(self.pop_size):
            variables = np.random.uniform(0, 1, self.n_vars)
            individual = Individual(variables)
            individual.objectives = self.evaluate_zdt1(variables)
            population.append(individual)
        return population
    
    def evolve(self):
        """进化过程"""
        population = self.initialize_population()
        
        for generation in range(self.max_gen):
            # 非支配排序
            fronts = self.fast_non_dominated_sort(population)
            for front in fronts:
                self.calculate_crowding_distance(front)
            
            # 生成子代
            offspring = []
            while len(offspring) < self.pop_size:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                offspring.extend([child1, child2])
            
            # 环境选择
            combined = population + offspring
            fronts = self.fast_non_dominated_sort(combined)
            new_population = []
            
            for front in fronts:
                self.calculate_crowding_distance(front)
                if len(new_population) + len(front) <= self.pop_size:
                    new_population.extend(front)
                else:
                    front.sort(key=lambda x: -x.crowding_distance)
                    new_population.extend(front[:self.pop_size - len(new_population)])
                    break
            
            population = new_population
            
            if (generation + 1) % 10 == 0:
                print(f"Generation {generation + 1}/{self.max_gen}")
        
        return population

def get_true_pareto_front(n_points=1000):
    """获取真实Pareto前沿"""
    f1 = np.linspace(0, 1, n_points)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])


def main():
    """主函数"""


if __name__ == "__main__":
    main()