import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import random
import matplotlib

# 设置中文字体和绘图参数
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class Particle:
    """粒子类"""
    def __init__(self, position, velocity=None):
        self.position = np.array(position)
        self.velocity = np.array(velocity) if velocity is not None else np.zeros_like(position)
        self.objectives = None
        self.pbest_position = self.position.copy()
        self.pbest_objectives = None
        self.domination_count = 0
        self.dominated_solutions = []
        self.rank = None
        self.crowding_distance = 0.0

class MOPSO:
    """标准多目标粒子群优化算法"""
    
    def __init__(self, n_vars=10, pop_size=100, max_gen=300, c1=2.0, c2=2.0, w=0.9):
        self.n_vars = n_vars
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.c1 = c1  # 学习因子1
        self.c2 = c2  # 学习因子2
        self.w = w    # 惯性权重
        self.archive = []  # 外部档案
        self.archive_size = 100  # 档案大小
        
    def evaluate_zdt1(self, position):
        """ZDT1目标函数"""
        # f1 = position[0]
        # g = 1 + 9 * np.sum(position[1:]) / (self.n_vars - 1)
        # f2 = g * (1 - np.sqrt(f1 / g))
        # f1 = position[0]
        # g = 1 + 9 * np.sum(position[1:]) / (self.n_vars - 1)
        # f2 = g * (1 - (f1 / g) ** 2)
        # f1 = position[0]
        # g = 1 + 9 * np.sum(position[1:]) / (self.n_vars - 1)
        # f2 = g * (1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1))
        x1 = position[0]
        f1 = 1 - np.exp(-4 * x1) * (np.sin(6 * np.pi * x1) ** 6)

        g = 1 + 9 * (np.sum(position[1:]) / (self.n_vars - 1)) ** 0.25
        f2 = g * (1 - (f1 / g) ** 2)

        return np.array([f1, f2])

    def evaluate_zdt2(self, position):
        """ZDT2目标函数"""
        f1 = position[0]
        g = 1 + 9 * np.sum(position[1:]) / (self.n_vars - 1)
        f2 = g * (1 - (f1 / g) ** 2)
        return np.array([f1, f2])

    def evaluate_zdt3(self, position):
        """ZDT3目标函数"""
        f1 = position[0]
        g = 1 + 9 * np.sum(position[1:]) / (self.n_vars - 1)
        f2 = g * (1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1))
        return np.array([f1, f2])

    def dominates(self, obj1, obj2):
        """判断obj1是否支配obj2"""
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
                if self.dominates(p.objectives, q.objectives):
                    p.dominated_solutions.append(q)
                elif self.dominates(q.objectives, p.objectives):
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
            for particle in front:
                particle.crowding_distance = float('inf')
            return
        
        for particle in front:
            particle.crowding_distance = 0
        
        for obj_idx in range(2):  # 2个目标函数
            front.sort(key=lambda x: x.objectives[obj_idx])
            front[0].crowding_distance = front[-1].crowding_distance = float('inf')
            
            obj_range = front[-1].objectives[obj_idx] - front[0].objectives[obj_idx]
            if obj_range > 0:
                for i in range(1, len(front) - 1):
                    distance = (front[i + 1].objectives[obj_idx] - 
                              front[i - 1].objectives[obj_idx]) / obj_range
                    front[i].crowding_distance += distance
    
    def update_archive(self, population):
        """更新外部档案 - 标准方法"""
        # 合并当前档案和种群
        combined = self.archive + population
        
        # 对合并后的解集进行非支配排序，只保留非支配解
        fronts = self.fast_non_dominated_sort(combined)
        if fronts:
            self.archive = fronts[0]
        
        # 如果档案超过大小限制，使用拥挤距离进行修剪
        if len(self.archive) > self.archive_size:
            self.calculate_crowding_distance(self.archive)
            self.archive.sort(key=lambda x: -x.crowding_distance)
            self.archive = self.archive[:self.archive_size]
    
    def select_global_best(self, particle):
        """选择全局最优解 - 标准MOPSO方法"""
        if not self.archive:
            return particle.pbest_position
        
        # 标准MOPSO：从档案中随机选择一个解作为全局最优
        gbest = random.choice(self.archive)
        return gbest.position
    
    def update_velocity(self, particle, gbest_position):
        """更新粒子速度 - 标准PSO公式"""
        r1, r2 = random.random(), random.random()
        
        # 计算速度更新
        cognitive = self.c1 * r1 * (particle.pbest_position - particle.position)
        social = self.c2 * r2 * (gbest_position - particle.position)
        
        particle.velocity = self.w * particle.velocity + cognitive + social
        
        # 限制速度范围（修正：提高速度上限）
        v_max = 0.5 * (1.0 - 0.0)  # 搜索空间范围的50%
        particle.velocity = np.clip(particle.velocity, -v_max, v_max)
    
    def update_position(self, particle):
        """更新粒子位置"""
        particle.position += particle.velocity
        
        # 限制位置范围
        particle.position = np.clip(particle.position, 0, 1)
    
    def initialize_population(self):
        """初始化种群"""
        population = []
        for _ in range(self.pop_size):
            position = np.random.uniform(0, 1, self.n_vars)
            velocity = np.random.uniform(-0.1, 0.1, self.n_vars)
            particle = Particle(position, velocity)
            particle.objectives = self.evaluate_zdt1(position)
            particle.pbest_objectives = particle.objectives.copy()
            population.append(particle)
        return population
    
    def evolve(self):
        """进化过程 - 标准MOPSO"""
        # 初始化种群
        population = self.initialize_population()
        
        # 初始化档案
        self.update_archive(population)
        
        for generation in range(self.max_gen):
            # 更新每个粒子
            for particle in population:
                # 选择全局最优
                gbest_position = self.select_global_best(particle)
                
                # 更新速度和位置
                self.update_velocity(particle, gbest_position)
                self.update_position(particle)
                
                # 评估新位置
                particle.objectives = self.evaluate_zdt1(particle.position)
                
                # 更新个体最优 - 标准MOPSO方法
                if self.dominates(particle.objectives, particle.pbest_objectives):
                    particle.pbest_position = particle.position.copy()
                    particle.pbest_objectives = particle.objectives.copy()
                elif not self.dominates(particle.pbest_objectives, particle.objectives):
                    # 如果互不支配，随机选择一个
                    if random.random() < 0.5:
                        particle.pbest_position = particle.position.copy()
                        particle.pbest_objectives = particle.objectives.copy()
            
            # 更新档案
            self.update_archive(population)
            
            # 动态调整惯性权重（线性递减）
            self.w = 0.9 - 0.5 * (generation / self.max_gen)
            
            if (generation + 1) % 10 == 0:
                print(f"Generation {generation + 1}/{self.max_gen}, Archive size: {len(self.archive)}")
        
        return self.archive

def get_true_pareto_front(n_points=1000):
    """获取真实Pareto前沿"""
    f1 = np.linspace(0, 1, n_points)
    f2 = 1 - np.sqrt(f1)
    return np.column_stack([f1, f2])


def main():
    """主函数"""


if __name__ == "__main__":
    main()
