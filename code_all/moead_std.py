import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import random
import matplotlib
import pandas as pd
import os
# 设置中文字体和绘图参数
matplotlib.use('TkAgg')
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class Individual:
    def __init__(self, variables):
        self.variables = np.array(variables)
        self.objectives = None

class MOEAD:
    def __init__(self, n_vars=30, pop_size=100, max_gen=300, cx_prob=0.8, mut_prob=0.1,
                 neighborhood_size=None, problem='zdt1'):
        """
        MOEA/D算法
        
        参数:
        n_vars: 变量维度
        pop_size: 种群大小
        max_gen: 最大迭代次数
        cx_prob: 交叉概率
        mut_prob: 变异概率
        neighborhood_size: 邻域大小（默认根据问题类型自适应设置）
        problem: 测试问题 ('zdt1', 'zdt2', 'zdt3', 'zdt6')
        """
        self.n_vars = n_vars
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.problem = problem
        self.n_obj = 2  # ZDT问题都是2目标
        
        # 根据问题类型自适应设置邻域大小
        # ZDT2是非凸问题，需要更大的邻域（30-40%）
        # ZDT1和ZDT3是凸/分段问题，可以使用较小邻域（20%）
        # ZDT6是复杂问题，需要中等邻域（25%）
        if neighborhood_size is None:
            if problem == 'zdt2':
                # ZDT2非凸，使用更大邻域
                neighborhood_size = max(15, int(pop_size * 0.3))
            elif problem == 'zdt6':
                # ZDT6复杂，使用中等邻域
                neighborhood_size = max(12, int(pop_size * 0.25))
            else:
                # ZDT1和ZDT3，使用标准邻域
                neighborhood_size = max(10, int(pop_size * 0.2))
        self.neighborhood_size = min(neighborhood_size, pop_size)
        
        # 初始化权重向量
        self.weights = self._generate_weight_vectors()
        
        # 计算邻域
        self.neighborhoods = self._calculate_neighborhoods()
        
        # 参考点（理想点）
        self.z = np.full(self.n_obj, np.inf)
        
    def _generate_weight_vectors(self):
        """生成均匀分布的权重向量（2目标问题）"""
        weights = []
        # 对于2目标问题，生成均匀分布的权重向量
        # 确保权重向量归一化（w1 + w2 = 1）
        # 避免权重为0，使用小的epsilon
        epsilon = 1e-6
        for i in range(self.pop_size):
            if self.pop_size == 1:
                w1 = 0.5
            else:
                # 生成从epsilon到1-epsilon的均匀分布
                w1 = epsilon + (1 - 2*epsilon) * i / (self.pop_size - 1)
            w2 = 1.0 - w1
            # 确保权重向量严格归一化
            weights.append(np.array([w1, w2]))
        return np.array(weights)
    
    def _calculate_neighborhoods(self):
        """计算每个权重向量的邻域"""
        neighborhoods = []
        for i in range(self.pop_size):
            # 计算与其他权重向量的距离
            distances = []
            for j in range(self.pop_size):
                dist = np.linalg.norm(self.weights[i] - self.weights[j])
                distances.append((dist, j))
            # 排序并选择最近的neighborhood_size个
            distances.sort(key=lambda x: x[0])
            neighbors = [idx for _, idx in distances[:self.neighborhood_size]]
            neighborhoods.append(neighbors)
        return neighborhoods
    
    def _techebycheff(self, objectives, weight, z):
        """切比雪夫分解方法"""
        # 标准切比雪夫分解：max(λ_i * (f_i - z_i))
        # z是参考点（理想点，最小值），所以f_i - z_i应该非负
        diff = objectives - z
        # 确保diff非负（因为z是理想点，应该是最小值）
        diff = np.maximum(diff, 0)
        # 切比雪夫分解：max(λ_i * (f_i - z_i))
        # 注意：权重向量已经归一化（w1 + w2 = 1）
        te_value = np.max(weight * diff)
        return te_value
    
    def evaluate(self, variables):
        """根据问题类型评估目标函数"""
        if self.problem == 'zdt1':
            return self._evaluate_zdt1(variables)
        elif self.problem == 'zdt2':
            return self._evaluate_zdt2(variables)
        elif self.problem == 'zdt3':
            return self._evaluate_zdt3(variables)
        elif self.problem == 'zdt6':
            return self._evaluate_zdt6(variables)
        else:
            raise ValueError(f"Unknown problem: {self.problem}")
    
    def _evaluate_zdt1(self, variables):
        """ZDT1目标函数"""
        f1 = variables[0]
        g = 1 + 9 * np.sum(variables[1:]) / (self.n_vars - 1)
        f2 = g * (1 - np.sqrt(f1 / g))
        return np.array([f1, f2])
    
    def _evaluate_zdt2(self, variables):
        """ZDT2目标函数"""
        f1 = variables[0]
        g = 1 + 9 * np.sum(variables[1:]) / (self.n_vars - 1)
        f2 = g * (1 - (f1 / g) ** 2)
        return np.array([f1, f2])
    
    def _evaluate_zdt3(self, variables):
        """ZDT3目标函数"""
        f1 = variables[0]
        g = 1 + 9 * np.sum(variables[1:]) / (self.n_vars - 1)
        f2 = g * (1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1))
        return np.array([f1, f2])
    
    def _evaluate_zdt6(self, variables):
        """ZDT6目标函数"""
        f1 = 1 - np.exp(-4 * variables[0]) * (np.sin(6 * np.pi * variables[0])) ** 6
        g = 1 + 9 * (np.sum(variables[1:]) / (self.n_vars - 1)) ** 0.25
        f2 = g * (1 - (f1 / g) ** 2)
        return np.array([f1, f2])
    
    def _sbx_crossover(self, parent1, parent2):
        """模拟二进制交叉"""
        if random.random() > self.cx_prob:
            return parent1.copy(), parent2.copy()
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        eta_c = 20  # 分布指数
        
        for i in range(self.n_vars):
            if random.random() <= 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    y1, y2 = sorted([parent1[i], parent2[i]])
                    rand = random.random()
                    beta = (2 * rand) ** (1 / (eta_c + 1)) if rand <= 0.5 else (1 / (2 * (1 - rand))) ** (1 / (eta_c + 1))
                    
                    c1 = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                    c2 = 0.5 * ((y1 + y2) + beta * (y2 - y1))
                    
                    child1[i] = max(0, min(1, c1))
                    child2[i] = max(0, min(1, c2))
        
        return child1, child2
    
    def _polynomial_mutation(self, individual):
        """多项式变异"""
        if random.random() > self.mut_prob:
            return individual.copy()
        
        mutated = individual.copy()
        eta_m = 20  # 分布指数
        
        for i in range(self.n_vars):
            if random.random() <= 1.0 / self.n_vars:
                y = mutated[i]
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
                mutated[i] = max(yl, min(yu, y))
        
        return mutated
    
    def initialize_population(self):
        """初始化种群"""
        population = []
        # 初始化参考点为很大的值
        self.z = np.full(self.n_obj, 1e10)
        
        for _ in range(self.pop_size):
            variables = np.random.uniform(0, 1, self.n_vars)
            individual = Individual(variables)
            individual.objectives = self.evaluate(variables)
            # 更新参考点（理想点，取最小值）
            self.z = np.minimum(self.z, individual.objectives)
            population.append(individual)
        return population
    
    def evolve(self):
        """进化过程 - 标准MOEA/D算法"""
        population = self.initialize_population()
        
        for generation in range(self.max_gen):
            # 随机打乱子问题的顺序（提高多样性）
            indices = list(range(self.pop_size))
            random.shuffle(indices)
            
            for i in indices:
                # 从邻域中选择两个父代
                neighbors = self.neighborhoods[i]
                p1_idx = random.choice(neighbors)
                p2_idx = random.choice(neighbors)
                
                parent1 = population[p1_idx]
                parent2 = population[p2_idx]
                
                # 交叉和变异
                child_vars1, child_vars2 = self._sbx_crossover(parent1.variables, parent2.variables)
                child_vars = self._polynomial_mutation(child_vars1)
                
                # 评估子代
                child_obj = self.evaluate(child_vars)
                
                # 更新参考点（理想点）
                self.z = np.minimum(self.z, child_obj)
                
                # 更新邻域中的解（标准MOEA/D更新策略）
                # 对于邻域中的每个解，如果子代对于该解的权重向量更好则更新
                # 确保当前子问题i也被考虑更新（如果不在邻域中）
                update_list = neighbors.copy()
                if i not in update_list:
                    update_list.append(i)
                
                # 对于非凸问题（ZDT2），使用更激进的更新策略
                # 限制每次更新的解数量，避免过度更新导致多样性丢失
                update_candidates = []
                for j in update_list:
                    # 计算当前解和子代的切比雪夫值（使用解j的权重向量）
                    current_te = self._techebycheff(population[j].objectives, self.weights[j], self.z)
                    child_te = self._techebycheff(child_obj, self.weights[j], self.z)
                    
                    # 如果子代更好，加入候选列表
                    if child_te < current_te:
                        improvement = current_te - child_te
                        update_candidates.append((j, improvement))
                
                # 按改进程度排序，优先更新改进最大的解
                update_candidates.sort(key=lambda x: x[1], reverse=True)
                
                # 对于非凸问题，限制更新数量以保持多样性
                if self.problem == 'zdt2':
                    max_updates = min(len(update_candidates), max(2, self.neighborhood_size // 3))
                else:
                    max_updates = len(update_candidates)
                
                # 执行更新
                for j, _ in update_candidates[:max_updates]:
                    population[j].variables = child_vars.copy()
                    population[j].objectives = child_obj.copy()
            
            if (generation + 1) % 50 == 0:
                # 计算当前种群的Pareto前沿覆盖率
                f1_values = [ind.objectives[0] for ind in population]
                f2_values = [ind.objectives[1] for ind in population]
                print(f"Generation {generation + 1}/{self.max_gen} | "
                      f"f1范围: [{min(f1_values):.4f}, {max(f1_values):.4f}] | "
                      f"f2范围: [{min(f2_values):.4f}, {max(f2_values):.4f}]")
        
        return population

def get_true_pareto_front(problem='zdt1', n_points=1000):
    """获取真实Pareto前沿"""
    if problem == 'zdt1':
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - np.sqrt(f1)
    elif problem == 'zdt2':
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - f1 ** 2
    elif problem == 'zdt3':
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - np.sqrt(f1) - f1 * np.sin(10 * np.pi * f1)
    elif problem == 'zdt6':
        # ZDT6的真实前沿：当g=1时（即x2...xn=0），f2 = 1 - (f1)^2
        # f1 = 1 - exp(-4*x1) * sin^6(6*pi*x1)
        x1 = np.linspace(0, 1, n_points)
        f1 = 1 - np.exp(-4 * x1) * (np.sin(6 * np.pi * x1)) ** 6
        # 对于ZDT6，当g=1时，f2 = g * (1 - (f1/g)^2) = 1 - f1^2
        f2 = 1 - f1 ** 2
        # 按f1排序
        sorted_idx = np.argsort(f1)
        f1 = f1[sorted_idx]
        f2 = f2[sorted_idx]
    else:
        raise ValueError(f"Unknown problem: {problem}")
    
    return np.column_stack([f1, f2])

def visualize_results(population, true_pareto, problem='zdt1', save_path=None):
    """可视化结果"""
    if save_path is None:
        save_path = f'{problem}_moead_results.png'
    
    plt.figure(figsize=(12, 5))
    
    # 提取目标函数值
    f1 = [ind.objectives[0] for ind in population]
    f2 = [ind.objectives[1] for ind in population]
    
    # 子图1：算法结果
    plt.subplot(1, 2, 1)
    plt.scatter(f1, f2, c='red', alpha=0.6, s=20)
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title(f'MOEA/D算法结果 - {problem.upper()}')
    plt.grid(True, alpha=0.3)
    
    # 子图2：对比图
    plt.subplot(1, 2, 2)
    plt.scatter(f1, f2, c='red', alpha=0.6, s=20, label='MOEA/D结果')
    plt.plot(true_pareto[:, 0], true_pareto[:, 1], 'b-', linewidth=2, label='真实Pareto前沿')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title(f'算法结果 vs 真实Pareto前沿 - {problem.upper()}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"结果已保存到: {save_path}")

def run_all_problems(num_runs=10):
    """在所有ZDT问题上运行MOEA/D算法
    
    参数:
    num_runs: 每个问题运行的次数（默认10次）
    """
    problems = ['zdt1', 'zdt2', 'zdt3', 'zdt6']
    
    # 参数配置（使用MOEA/D的默认参数）
    n_vars = 30          # 变量维度
    pop_size = 100       # 种群大小
    max_gen = 300        # 最大迭代次数
    cx_prob = 0.8        # 交叉概率
    mut_prob = 0.1       # 变异概率
    
    # MOEA/D特有参数（默认设为pop_size的20%，即20）
    neighborhood_size = None  # None表示使用默认值（pop_size的20%）
    
    # 存储所有运行的结果（用于保存到CSV）
    all_runs_results = {}
    
    # 存储最后一次运行的结果（用于可视化和返回）
    last_run_results = {}
    
    print(f"\n开始运行{num_runs}次独立实验...")
    print(f"参数设置: 种群大小={pop_size}, 迭代次数={max_gen}, 变量数={n_vars}")
    print("=" * 60)
    
    # 对每个问题运行多次
    for problem in problems:
        print(f"\n{'='*60}")
        print(f"问题: {problem.upper()}")
        print(f"{'='*60}")
        
        # 存储当前问题的所有运行结果
        problem_runs = []
        
        for run in range(num_runs):
            print(f"\n运行第 {run+1}/{num_runs} 次")
            print(f"参数: 种群{pop_size}, 迭代{max_gen}, 交叉{cx_prob}, 变异{mut_prob}")
            
            # 运行算法
            moead = MOEAD(n_vars=n_vars, pop_size=pop_size, max_gen=max_gen,
                          cx_prob=cx_prob, mut_prob=mut_prob, 
                          neighborhood_size=neighborhood_size, problem=problem)
            
            # 显示实际使用的邻域大小（只在第一次显示）
            if run == 0:
                actual_neighborhood = moead.neighborhood_size
                if problem == 'zdt2':
                    print(f"MOEA/D特有参数: 邻域大小{actual_neighborhood} (pop_size的30%，针对非凸问题优化)")
                elif problem == 'zdt6':
                    print(f"MOEA/D特有参数: 邻域大小{actual_neighborhood} (pop_size的25%，针对复杂问题优化)")
                else:
                    print(f"MOEA/D特有参数: 邻域大小{actual_neighborhood} (pop_size的20%，标准设置)")
            
            final_population = moead.evolve()
            
            # 性能分析
            f1 = [ind.objectives[0] for ind in final_population]
            f2 = [ind.objectives[1] for ind in final_population]
            print(f"运行 {run+1} 完成 - f1范围: [{min(f1):.4f}, {max(f1):.4f}], f2范围: [{min(f2):.4f}, {max(f2):.4f}]")
            
            # 保存当前运行的结果
            problem_runs.append({
                'population': final_population,
                'f1_range': (min(f1), max(f1)),
                'f2_range': (min(f2), max(f2))
            })
            
            # 保存最后一次运行的结果用于可视化
            if run == num_runs - 1:
                true_pareto = get_true_pareto_front(problem)
                last_run_results[problem] = {
                    'population': final_population,
                    'true_pareto': true_pareto,
                    'f1_range': (min(f1), max(f1)),
                    'f2_range': (min(f2), max(f2))
                }
        
        # 保存当前问题的所有运行结果
        all_runs_results[problem] = problem_runs
    
    # 可视化最后一次运行的结果
    print("\n" + "="*60)
    print("可视化最后一次运行的结果...")
    print("="*60)
    for problem, result_data in last_run_results.items():
        visualize_results(result_data['population'], result_data['true_pareto'], problem)
        print(f"\n{problem.upper()}结果分析:")
        print(f"f1范围: [{result_data['f1_range'][0]:.4f}, {result_data['f1_range'][1]:.4f}]")
        print(f"f2范围: [{result_data['f2_range'][0]:.4f}, {result_data['f2_range'][1]:.4f}]")
    
    print("\n" + "="*50)
    print("所有问题优化完成！")
    print("="*50)
    
    # 保存所有运行的结果到CSV文件
    print("\n正在保存Pareto前沿坐标点到CSV文件...")
    save_moead_results_to_csv(all_runs_results)
    print("所有数据保存完成！")
    
    return last_run_results

def save_moead_results_to_csv(all_runs_results):
    """
    将MOEA/D的运行结果保存到CSV文件（追加模式，不覆盖已有数据）
    格式与unified_algorithm_comparison.py保持一致
    
    参数:
    all_runs_results: 字典，格式为 {problem: [run1_result, run2_result, ...]}
    """
    # 创建输出目录
    output_dir = "pareto_front_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    algorithm_name = "MOEA/D"
    # 文件名中使用安全的算法名称（替换斜杠）
    algorithm_name_safe = algorithm_name.replace("/", "-")
    
    # 处理每个问题
    for problem, runs_data in all_runs_results.items():
        print(f"\n处理问题: {problem.upper()}")
        
        filename = os.path.join(output_dir, f"{problem}_{algorithm_name_safe}_all_runs.csv")
        
        # 检查文件是否存在
        existing_data = None
        max_run = 0
        
        if os.path.exists(filename):
            print(f"  读取现有文件: {filename}")
            try:
                existing_data = pd.read_csv(filename, encoding='utf-8-sig')
                if 'run' in existing_data.columns:
                    max_run = existing_data['run'].max()
                    print(f"  现有最大运行号: {max_run}")
            except Exception as e:
                print(f"  读取现有文件时出错: {e}，将创建新文件")
                existing_data = None
                max_run = 0
        
        # 存储当前运行的数据
        new_data = []
        
        for run_idx, result in enumerate(runs_data):
            # 提取目标函数值
            population = result['population']
            f1_values = [ind.objectives[0] for ind in population]
            f2_values = [ind.objectives[1] for ind in population]
            
            # 添加运行次数标记（从现有最大运行号+1开始）
            current_run = max_run + run_idx + 1
            for f1, f2 in zip(f1_values, f2_values):
                new_data.append({
                    'run': current_run,
                    'f1': f1,
                    'f2': f2
                })
        
        # 创建新的DataFrame
        df_new = pd.DataFrame(new_data)
        
        # 如果有现有数据，合并
        if existing_data is not None:
            df = pd.concat([existing_data, df_new], ignore_index=True)
            print(f"  新增运行: {len(runs_data)} 次")
            print(f"  新增解数量: {len(df_new)}")
        else:
            df = df_new
            print(f"  新文件，运行次数: {len(runs_data)}")
        
        # 保存到CSV
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"已保存: {filename}")
        print(f"  - 总运行次数: {df['run'].max()}")
        print(f"  - 总解数量: {len(df)}")
    
    print(f"\n所有数据已保存到 '{output_dir}' 目录")

def main():
    """主函数"""

    # 运行所有问题（每个问题运行10次）
    num_runs =30
    results = run_all_problems(num_runs=num_runs)

if __name__ == "__main__":
    main()

