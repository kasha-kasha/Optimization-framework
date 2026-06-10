import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import random
import matplotlib
from scipy.spatial.distance import cdist

# 设置中文字体和绘图参数
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class Individual:
    def __init__(self, variables):
        self.variables = variables
        self.objectives = None
        self.raw_fitness = 0.0
        self.density = 0.0
        self.fitness = 0.0
        self.domination_count = 0
        self.dominated_solutions = []

class SPEA2:
    def __init__(self, n_vars=30, pop_size=100, max_gen=300, cx_prob=0.7, mut_prob=0.1):
        self.n_vars = n_vars
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.archive_size = pop_size  # 外部档案大小
    
    def dominates(self, ind1, ind2):
        """判断ind1是否支配ind2"""
        obj1, obj2 = ind1.objectives, ind2.objectives
        better = any(obj1[i] < obj2[i] for i in range(len(obj1)))
        worse = any(obj1[i] > obj2[i] for i in range(len(obj1)))
        return better and not worse
    
    def calculate_raw_fitness(self, population):
        """计算原始适应度"""
        for p in population:
            p.domination_count = 0
            p.dominated_solutions = []
        
        # 计算支配关系
        for i, p in enumerate(population):
            for j, q in enumerate(population):
                if i != j:
                    if self.dominates(p, q):
                        p.dominated_solutions.append(q)
                    elif self.dominates(q, p):
                        p.domination_count += 1
        
        # 计算原始适应度
        for p in population:
            p.raw_fitness = p.domination_count
    
    def calculate_density(self, population):
        """计算密度估计"""
        if len(population) <= 1:
            for p in population:
                p.density = 0.0
            return
        
        # 计算目标空间中的距离矩阵
        objectives = np.array([ind.objectives for ind in population])
        distances = cdist(objectives, objectives, metric='euclidean')
        
        # 对每个个体，找到第k近邻的距离
        k = int(np.sqrt(len(population)))
        k = max(1, min(k, len(population) - 1))
        
        for i, p in enumerate(population):
            sorted_distances = np.sort(distances[i])
            p.density = 1.0 / (sorted_distances[k] + 2.0)  # 加2避免除零
    
    def calculate_fitness(self, population):
        """计算最终适应度"""
        self.calculate_raw_fitness(population)
        self.calculate_density(population)
        
        for p in population:
            p.fitness = p.raw_fitness + p.density
    
    def environmental_selection(self, population):
        """环境选择"""
        self.calculate_fitness(population)
        
        # 找出非支配解
        non_dominated = [p for p in population if p.raw_fitness == 0]
        
        if len(non_dominated) <= self.archive_size:
            # 如果非支配解数量不超过档案大小，添加支配解
            dominated = [p for p in population if p.raw_fitness > 0]
            dominated.sort(key=lambda x: x.fitness)
            
            archive = non_dominated.copy()
            remaining = self.archive_size - len(archive)
            archive.extend(dominated[:remaining])
            
            return archive
        else:
            # 如果非支配解数量超过档案大小，需要截断
            return self.truncate_archive(non_dominated)
    
    def truncate_archive(self, archive):
        """截断档案"""
        if len(archive) <= self.archive_size:
            return archive
        
        # 按适应度排序
        archive.sort(key=lambda x: x.fitness)
        
        # 使用聚类方法截断
        while len(archive) > self.archive_size:
            # 计算每个解到其他解的最小距离
            objectives = np.array([ind.objectives for ind in archive])
            distances = cdist(objectives, objectives, metric='euclidean')
            
            # 找到最小距离最小的解（最拥挤的）
            min_distances = []
            for i in range(len(archive)):
                other_distances = [distances[i][j] for j in range(len(archive)) if i != j]
                min_distances.append(min(other_distances))
            
            # 移除最拥挤的解
            min_idx = min_distances.index(min(min_distances))
            archive.pop(min_idx)
        
        return archive
    
    def tournament_selection(self, population):
        """锦标赛选择"""
        tournament = random.sample(population, 2)
        return min(tournament, key=lambda x: x.fitness)
    
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
                    eta_c = 20  # 分布指数
                    beta = (2 * rand) ** (1/(eta_c + 1)) if rand <= 0.5 else (1/(2*(1-rand))) ** (1/(eta_c + 1))
                    
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
                yl, yu = 0.0, 1.0
                
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
        """SPEA2进化过程"""
        # 初始化种群
        population = self.initialize_population()
        archive = []
        
        for generation in range(self.max_gen):
            # 合并种群和档案
            combined = population + archive
            
            # 环境选择
            archive = self.environmental_selection(combined)
            
            # 生成子代
            offspring = []
            while len(offspring) < self.pop_size:
                parent1 = self.tournament_selection(archive)
                parent2 = self.tournament_selection(archive)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                offspring.extend([child1, child2])
            
            # 更新种群
            population = offspring
            
            if (generation + 1) % 10 == 0:
                print(f"Generation {generation + 1}/{self.max_gen}")
        
        # 最终环境选择
        final_archive = self.environmental_selection(population + archive)
        return final_archive

def get_true_pareto_front(n_points=1000):
    """获取真实Pareto前沿"""
    f1 = np.linspace(0, 1, n_points)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])


def main():
    """main"""

if __name__ == "__main__":
    main()
