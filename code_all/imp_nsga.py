import numpy as np
import matplotlib.pyplot as plt
from typing import List
import random
import math
import matplotlib
matplotlib.use('TkAgg')
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

class ImprovedNSGA2:
    def __init__(self, n_vars=30, pop_size=100, max_gen=300, cx_prob=0.8, mut_prob=0.1, var_bounds=None):
        self.n_vars = n_vars
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        # 变量边界设置：默认所有变量在[0,1]，ZDT4/ZDT6等需要特殊设置
        if var_bounds is None:
            self.var_bounds = [(0.0, 1.0)] * n_vars
        else:
            self.var_bounds = var_bounds
        # SBX分布指数上下限（将基于多样性自适应）
        self.eta_c_min, self.eta_c_max = 5, 20
        self.msoa_start_gen = max_gen // 3  # MSOA开始代数
        # 多样性度量与自适应参数
        self.diversity_ema = None  # 指标EMA
        self.diversity_alpha = 0.2  # EMA平滑
        self.diversity_baseline = None  # 初始/基线多样性
        self.bias_low, self.bias_high = 0.2, 1.0  # 偏置系数区间
        self.low_diversity_factor = 0.8  # 【改进9】提高低多样性时的变异增强因子
        # ====== 为EMA有效性分析新增：历史记录 ======
        self.spacing_history = []            # 原始Spacing D(t)
        self.diversity_norm_history = []     # 归一化多样性 D_norm(t)
        self.diversity_ema_history = []      # EMA平滑后的多样性 D_EMA(t)
        # 自适应参数历史（用于联动可视化）
        self.eta_c_history = []              # SBX分布指数随代数变化
        self.mut_prob_history = []           # 实际使用的变异概率随代数变化
        self.msoa_scale_history = []         # MSOA局部搜索扰动尺度随代数变化
    
    def latin_hypercube_init(self):
        """拉丁超立方初始化 - 支持自定义变量边界"""
        samples = np.zeros((self.pop_size, self.n_vars))
        for i in range(self.n_vars):
            lower, upper = self.var_bounds[i]
            # 在[0,1]区间生成LHS样本，然后映射到实际边界
            lhs_samples = np.random.uniform(0, 1, self.pop_size)
            lhs_samples = np.random.permutation(lhs_samples)
            samples[:, i] = lower + lhs_samples * (upper - lower)
        return samples
    
    def dominates(self, ind1, ind2):
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
        
        for obj_idx in range(2):
            front.sort(key=lambda x: x.objectives[obj_idx])
            front[0].crowding_distance = front[-1].crowding_distance = float('inf')
            obj_range = front[-1].objectives[obj_idx] - front[0].objectives[obj_idx]
            if obj_range > 0:
                for i in range(1, len(front) - 1):
                    distance = (front[i + 1].objectives[obj_idx] - front[i - 1].objectives[obj_idx]) / obj_range
                    front[i].crowding_distance += distance

    def _euclidean(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _filter_nondominated(self, pop):
        nd = []
        for p in pop:
            dominated = False
            for q in pop:
                if p is q:
                    continue
                if self.dominates(q, p):
                    dominated = True
                    break
            if not dominated:
                nd.append(p)
        return nd

    def compute_spacing(self, population):
        """近邻间距的标准差作为多样性（越小越均匀，越大越分散）。"""
        nd = self._filter_nondominated(population)
        if len(nd) <= 2:
            return 0.0
        dists = []
        for i, p in enumerate(nd):
            dmin = float('inf')
            for j, q in enumerate(nd):
                if i == j:
                    continue
                d = self._euclidean(p.objectives, q.objectives)
                if d < dmin:
                    dmin = d
            dists.append(0.0 if dmin == float('inf') else dmin)
        mean_d = sum(dists) / len(dists)
        if len(dists) <= 1:
            return 0.0
        var = sum((d - mean_d) ** 2 for d in dists) / (len(dists) - 1)
        return math.sqrt(var)

    def _update_diversity(self, population):
        spacing = self.compute_spacing(population)
        if self.diversity_baseline is None:
            self.diversity_baseline = spacing if spacing > 1e-12 else 1.0
        # 归一化多样性（0~1+），采用基线做尺度
        diversity_norm = max(0.0, min(1.5, spacing / (self.diversity_baseline + 1e-12)))
        if self.diversity_ema is None:
            self.diversity_ema = diversity_norm
        else:
            self.diversity_ema = self.diversity_alpha * diversity_norm + (1 - self.diversity_alpha) * self.diversity_ema
        # 记录多样性历史（用于EAM分析与可视化）
        self.spacing_history.append(spacing)
        self.diversity_norm_history.append(diversity_norm)
        self.diversity_ema_history.append(self.diversity_ema)
        return self.diversity_ema
    
    
    def adaptive_sbx_crossover(self, parent1, parent2, generation):
        """自适应SBX交叉 - 改进版：增强多样性维护"""
        if random.random() > self.cx_prob:
            return parent1, parent2
        
        # 基于多样性的自适应分布指数：多样性低 -> eta_c低（更探索）
        diversity_norm = self.diversity_ema if self.diversity_ema is not None else 1.0
        eta_c = self.eta_c_max - (self.eta_c_max - self.eta_c_min) * max(0.0, min(1.0, diversity_norm))
        
        child1_vars = parent1.variables.copy()
        child2_vars = parent2.variables.copy()
        
        for i in range(self.n_vars):
            lower, upper = self.var_bounds[i]
            if random.random() <= 0.5 and abs(parent1.variables[i] - parent2.variables[i]) > 1e-14:
                y1, y2 = sorted([parent1.variables[i], parent2.variables[i]])
                # 【改进1】恢复标准SBX的完整随机范围[0, 1]，增强探索能力
                rand = random.random()
                
                # 【改进2】改进的定向SBX：只在父代质量差异较大时应用，且降低偏置强度
                rank1 = parent1.rank if parent1.rank is not None else float('inf')
                rank2 = parent2.rank if parent2.rank is not None else float('inf')
                cd1 = parent1.crowding_distance if parent1.crowding_distance != float('inf') else 0
                cd2 = parent2.crowding_distance if parent2.crowding_distance != float('inf') else 0
                quality_diff = abs(rank1 - rank2) + abs(cd1 - cd2)
                
                # 【改进3】提高质量差异阈值，减少偏置使用频率
                if quality_diff > 2.0:
                    # rank越小越好，crowding_distance越大越好
                    parent1_score = -rank1 + cd1
                    parent2_score = -rank2 + cd2
                    # 【改进4】降低偏置强度：从0.5降到0.2，并且只在多样性充足时应用
                    if diversity_norm > 0.3:  # 多样性充足时才应用偏置
                        bias_factor = 0.2  # 降低偏置强度
                        target_edge = 0.2 if parent1_score > parent2_score else 0.8
                        rand = (1 - bias_factor) * rand + bias_factor * target_edge
                
                beta = (2 * rand) ** (1 / (eta_c + 1)) if rand <= 0.5 else (1 / (2 * (1 - rand))) ** (1 / (eta_c + 1))
                
                c1 = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                c2 = 0.5 * ((y1 + y2) + beta * (y2 - y1))
                
                # 使用实际变量边界进行约束
                child1_vars[i] = max(lower, min(upper, c1))
                child2_vars[i] = max(lower, min(upper, c2))
        
        child1 = Individual(child1_vars)
        child2 = Individual(child2_vars)
        child1.objectives = self.evaluate_zdt1(child1_vars)
        child2.objectives = self.evaluate_zdt1(child2_vars)
        return child1, child2
    
    def polynomial_mutation(self, individual):
        """多项式变异 - 改进版：增强探索能力"""
        # 【改进10】自适应变异概率：多样性低 -> 增强变异
        diversity_norm = self.diversity_ema if self.diversity_ema is not None else 1.0
        mut_prob_multiplier = 1.0 + self.low_diversity_factor * (1.0 - max(0.0, min(1.0, diversity_norm)))
        mut_prob = self.mut_prob * mut_prob_multiplier
        if random.random() > mut_prob:
            return individual
        
        mutated_vars = individual.variables.copy()
        # 【改进11】多样性低时降低eta_m（分布指数），使变异更大
        eta_m = 15 if diversity_norm < 0.3 else 20
        
        # 【改进12】多样性低时，提高单基因变异概率
        per_gene_prob = 1.0 / self.n_vars
        per_gene_prob_multiplier = 1.0 + 1.0 * (1.0 - max(0.0, min(1.0, diversity_norm)))  # 从0.5提高到1.0
        per_gene_prob *= per_gene_prob_multiplier
        
        for i in range(self.n_vars):
            if random.random() <= per_gene_prob:
                y = mutated_vars[i]
                yl, yu = self.var_bounds[i]  # 使用实际变量边界
                
                # 计算归一化的delta
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
    
    def msoa_local_search(self, individual, generation, population):
        """MSOA局部搜索 - 改进版：增强多样性维护"""
        # 【改进5】延迟MSOA启动，降低应用频率，增强探索
        if generation < self.msoa_start_gen or random.random() > 0.2:
            return individual
        
        # 海鸥优化局部搜索
        improved_vars = individual.variables.copy()
        # 【改进6】增大扰动范围，增强局部搜索的探索能力
        perturbation_scale = 0.02 if self.diversity_ema is not None and self.diversity_ema < 0.3 else 0.01
        for i in range(self.n_vars):
            lower, upper = self.var_bounds[i]
            # 扰动幅度按变量范围缩放
            var_range = upper - lower
            perturbation = np.random.normal(0, perturbation_scale * var_range)
            improved_vars[i] = max(lower, min(upper, improved_vars[i] + perturbation))
        
        improved_individual = Individual(improved_vars)
        improved_individual.objectives = self.evaluate_zdt1(improved_vars)
        
        # 【改进7】放松接受准则：更倾向于保持多样性
        # 接受准则：支配改进 或 同rank但更均匀（拥挤距离/最近邻更大）
        if self.dominates(improved_individual, individual):
            return improved_individual
        
        # 计算最近邻距离提升
        def nn_dist(ind, pool):
            best = float('inf')
            for q in pool:
                if q is ind:
                    continue
                d = self._euclidean(ind.objectives, q.objectives)
                if d < best:
                    best = d
            return 0.0 if best == float('inf') else best
        
        # 临时加入以测度
        temp_pool = population + [improved_individual]
        nn_old = nn_dist(individual, population)
        nn_new = nn_dist(improved_individual, temp_pool)
        
        # 【改进8】降低接受阈值：从5%降到2%，更宽松地接受多样性改进
        if nn_new > nn_old * 1.02:  # 2%提升阈值
            return improved_individual
        return individual
    
    def tournament_selection(self, population):
        """锦标赛选择"""
        tournament = random.sample(population, 2)
        return min(tournament, key=lambda x: (x.rank, -x.crowding_distance))
    
    def evaluate_zdt1(self, variables):
        """ZDT1目标函数"""
        f1 = variables[0]
        g = 1 + 9 * np.sum(variables[1:]) / (self.n_vars - 1)
        f2 = g * (1 - np.sqrt(f1 / g))
        return np.array([f1, f2])
    
    def evolve(self):
        """改进的进化过程"""
        # 每次进化前重置多样性与自适应参数历史，保证结果干净可重复
        self.diversity_baseline = None
        self.diversity_ema = None
        self.spacing_history = []
        self.diversity_norm_history = []
        self.diversity_ema_history = []
        self.eta_c_history = []
        self.mut_prob_history = []
        self.msoa_scale_history = []

        # 拉丁超立方初始化
        lhs_samples = self.latin_hypercube_init()
        population = []
        for i in range(self.pop_size):
            individual = Individual(lhs_samples[i])
            individual.objectives = self.evaluate_zdt1(individual.variables)
            population.append(individual)
        
        for generation in range(self.max_gen):
            # 更新多样性状态（用于自适应参数）
            self._update_diversity(population)

            # ====== 为EMA与自适应算子联动分析记录参数轨迹 ======
            # 使用当前EMA多样性估计一个“代表性”的代际自适应参数
            diversity_norm_for_adapt = self.diversity_ema if self.diversity_ema is not None else 1.0
            diversity_norm_for_adapt = max(0.0, min(1.0, diversity_norm_for_adapt))

            # 1) 基于多样性的SBX分布指数 eta_c（与adaptive_sbx_crossover中的逻辑保持一致）
            eta_c_gen = self.eta_c_max - (self.eta_c_max - self.eta_c_min) * diversity_norm_for_adapt

            # 2) 自适应变异概率（与polynomial_mutation中的整体放大因子一致）
            mut_prob_multiplier = 1.0 + self.low_diversity_factor * (1.0 - diversity_norm_for_adapt)
            mut_prob_gen = self.mut_prob * mut_prob_multiplier

            # 3) MSOA局部搜索扰动尺度（与msoa_local_search中的逻辑保持一致）
            msoa_scale_gen = 0.02 if (self.diversity_ema is not None and self.diversity_ema < 0.3) else 0.01

            self.eta_c_history.append(eta_c_gen)
            self.mut_prob_history.append(mut_prob_gen)
            self.msoa_scale_history.append(msoa_scale_gen)
            # 非支配排序
            fronts = self.fast_non_dominated_sort(population)
            for front in fronts:
                self.calculate_crowding_distance(front)
            
            # 生成子代
            offspring = []
            while len(offspring) < self.pop_size:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                child1, child2 = self.adaptive_sbx_crossover(parent1, parent2, generation)
                child1 = self.polynomial_mutation(child1)
                child2 = self.polynomial_mutation(child2)
                child1 = self.msoa_local_search(child1, generation, population)
                child2 = self.msoa_local_search(child2, generation, population)
                offspring.extend([child1, child2])
            
            # 环境选择 - 非支配等级+拥挤度距离联合排序，边界保护
            combined = population + offspring
            fronts = self.fast_non_dominated_sort(combined)
            new_population = []
            
            # 按非支配等级逐层填充种群
            for front in fronts:
                self.calculate_crowding_distance(front)
                
                # 如果当前前沿可以完全加入
                if len(new_population) + len(front) <= self.pop_size:
                    new_population.extend(front)
                else:
                    # 需要从当前前沿中选择部分个体
                    remaining_slots = self.pop_size - len(new_population)
                    
                    # 边界保护：优先保留边界极值点（仅在当前前沿内）
                    boundary_individuals = set()
                    m_obj = len(front[0].objectives)
                    for obj_idx in range(m_obj):
                        min_ind = min(front, key=lambda x: x.objectives[obj_idx])
                        max_ind = max(front, key=lambda x: x.objectives[obj_idx])
                        boundary_individuals.add(min_ind)
                        boundary_individuals.add(max_ind)
                    
                    # 分离边界个体和非边界个体
                    boundary_list = list(boundary_individuals)
                    non_boundary = [ind for ind in front if ind not in boundary_individuals]
                    
                    # 如果边界个体数量不超过剩余位置，先加入所有边界个体
                    if len(boundary_list) <= remaining_slots:
                        new_population.extend(boundary_list)
                        remaining_slots -= len(boundary_list)
                        
                        # 按拥挤度距离排序，填充剩余位置
                        if remaining_slots > 0 and non_boundary:
                            non_boundary.sort(key=lambda x: -x.crowding_distance)
                            new_population.extend(non_boundary[:remaining_slots])
                    else:
                        # 边界个体数量超过剩余位置，按拥挤度选择
                        front.sort(key=lambda x: -x.crowding_distance)
                        new_population.extend(front[:remaining_slots])
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
    """main"""


if __name__ == "__main__":
    main()