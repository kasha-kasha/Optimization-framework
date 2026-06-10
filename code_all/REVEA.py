# 生成完整的 RVEA 双目标代码文件
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import random
import matplotlib
import pandas as pd
import os

# 设置中文字体和绘图参数
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class Individual:
    def __init__(self, variables):
        self.variables = np.array(variables)
        self.objectives = None
        self.apd = None  # 角度惩罚距离（RVEA特有）


class RVEA:
    def __init__(self, n_vars=30, pop_size=100, max_gen=300, cx_prob=0.9, mut_prob=0.1,
                 alpha=2.0, fr=0.1, problem='zdt1'):
        """
        RVEA算法（Reference Vector Guided Evolutionary Algorithm）

        参数:
        n_vars: 变量维度
        pop_size: 种群大小（初始种群，实际由参考向量数量决定）
        max_gen: 最大迭代次数
        cx_prob: 交叉概率
        mut_prob: 变异概率
        alpha: APD惩罚因子（控制收敛与多样性的平衡，默认2.0）
        fr: 参考向量自适应频率（每fr*max_gen代调整一次，默认0.1）
        problem: 测试问题 ('zdt1', 'zdt2', 'zdt3', 'zdt6')
        """
        self.n_vars = n_vars
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.alpha = alpha
        self.fr = fr
        self.problem = problem
        self.n_obj = 2  # ZDT问题都是2目标

        # 生成参考向量（均匀分布的单位向量）
        self.ref_vectors = self._generate_reference_vectors()
        self.n_ref = len(self.ref_vectors)

        # 参考点（理想点，用于目标归一化）
        self.z = np.full(self.n_obj, np.inf)
        # 最差点（nadir点，用于目标归一化）
        self.z_max = np.full(self.n_obj, -np.inf)

        # 参考向量惩罚角（用于APD计算）
        self.penalty_angles = self._calculate_penalty_angles()

        # 自适应参数
        self.adaptive_gen = int(max_gen * fr)  # 触发自适应的代数间隔

    def _generate_reference_vectors(self):
        """生成均匀分布的参考向量（2目标问题）"""
        # 使用简单均匀分布生成参考向量
        # 对于2目标，在[0,1]区间均匀采样后归一化
        vectors = []
        n_partitions = self.pop_size - 1  # 分区数

        for i in range(self.pop_size):
            if self.pop_size == 1:
                v = np.array([0.5, 0.5])
            else:
                # 生成从epsilon开始的均匀分布，避免边界问题
                epsilon = 1e-6
                v1 = epsilon + (1 - 2*epsilon) * i / n_partitions
                v2 = 1.0 - v1
                v = np.array([v1, v2])
            # 归一化为单位向量
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                v = v / norm
            vectors.append(v)

        return np.array(vectors)

    def _calculate_penalty_angles(self):
        """计算参考向量间的最小夹角，用于APD中的惩罚项"""
        n = len(self.ref_vectors)
        angles = np.full(n, np.pi)  # 初始化为最大角度

        for i in range(n):
            min_angle = np.pi
            for j in range(n):
                if i != j:
                    # 计算两个单位向量的夹角
                    cos_angle = np.dot(self.ref_vectors[i], self.ref_vectors[j])
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    if angle < min_angle:
                        min_angle = angle
            angles[i] = min_angle

        return angles

    def _normalize_objectives(self, objectives):
        """目标归一化：将目标值映射到[0,1]区间"""
        # 使用理想点和最差点进行归一化
        normalized = np.zeros(self.n_obj)
        for i in range(self.n_obj):
            denom = self.z_max[i] - self.z[i]
            if abs(denom) < 1e-10:
                normalized[i] = 0.5  # 避免除零
            else:
                normalized[i] = (objectives[i] - self.z[i]) / denom
            # 裁剪到[0,1]
            normalized[i] = np.clip(normalized[i], 0.0, 1.0)
        return normalized

    def _calculate_apd(self, objectives, ref_idx, generation):
        """
        计算角度惩罚距离（Angle Penalized Distance）

        APD = d * (1 + P * angle^alpha)
        其中：
        - d: 归一化目标到原点的距离
        - angle: 归一化目标向量与参考向量的夹角
        - P: 自适应惩罚系数，随代数增加而增大（强调收敛性）
        """
        # 归一化目标值
        norm_obj = self._normalize_objectives(objectives)

        # 计算到原点的距离（收敛性度量）
        distance = np.linalg.norm(norm_obj)

        # 计算与参考向量的夹角（多样性度量）
        if distance < 1e-10:
            angle = 0.0
        else:
            obj_direction = norm_obj / distance
            cos_angle = np.dot(obj_direction, self.ref_vectors[ref_idx])
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)

        # 自适应惩罚系数：随代数增加，更强调收敛
        # P = (generation / max_gen)^alpha
        progress = generation / self.max_gen
        P = progress ** self.alpha

        # 角度惩罚距离
        # 使用最小邻域角作为归一化因子
        penalty_angle = self.penalty_angles[ref_idx]
        if penalty_angle < 1e-10:
            penalty_angle = 1e-10

        apd = distance * (1 + P * (angle / penalty_angle) ** self.alpha)

        return apd

    def _associate_to_reference(self, objectives, generation):
        """
        将目标向量关联到最近的参考向量
        返回: (最佳参考向量索引, APD值)
        """
        best_idx = 0
        best_apd = float('inf')

        for i in range(self.n_ref):
            apd = self._calculate_apd(objectives, i, generation)
            if apd < best_apd:
                best_apd = apd
                best_idx = i

        return best_idx, best_apd

    def _reference_vector_adaptation(self, population):
        """
        参考向量自适应调整
        根据当前种群的分布调整参考向量的方向
        """
        # 计算当前种群的归一化目标中心
        if len(population) == 0:
            return

        center = np.zeros(self.n_obj)
        for ind in population:
            center += self._normalize_objectives(ind.objectives)
        center /= len(population)

        # 如果种群分布偏离中心，轻微调整参考向量方向
        # 这里使用简单的缩放策略
        for i in range(self.n_ref):
            # 根据种群中心偏移调整参考向量
            adjusted = self.ref_vectors[i] + 0.1 * center
            norm = np.linalg.norm(adjusted)
            if norm > 1e-10:
                self.ref_vectors[i] = adjusted / norm

        # 重新计算惩罚角
        self.penalty_angles = self._calculate_penalty_angles()

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
        """模拟二进制交叉（SBX）"""
        if random.random() > self.cx_prob:
            return parent1.copy(), parent2.copy()

        child1 = parent1.copy()
        child2 = parent2.copy()
        eta_c = 30  # 分布指数（RVEA通常使用较大值）

        for i in range(self.n_vars):
            if random.random() <= 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    y1, y2 = sorted([parent1[i], parent2[i]])
                    rand = random.random()

                    if rand <= 0.5:
                        beta = (2 * rand) ** (1 / (eta_c + 1))
                    else:
                        beta = (1 / (2 * (1 - rand))) ** (1 / (eta_c + 1))

                    c1 = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                    c2 = 0.5 * ((y1 + y2) + beta * (y2 - y1))

                    child1[i] = max(0, min(1, c1))
                    child2[i] = max(0, min(1, c2))

        return child1, child2

    def _polynomial_mutation(self, individual):
        """多项式变异（PM）"""
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
        # 初始化理想点和最差点
        self.z = np.full(self.n_obj, np.inf)
        self.z_max = np.full(self.n_obj, -np.inf)

        for _ in range(self.pop_size):
            variables = np.random.uniform(0, 1, self.n_vars)
            individual = Individual(variables)
            individual.objectives = self.evaluate(variables)

            # 更新理想点和最差点
            self.z = np.minimum(self.z, individual.objectives)
            self.z_max = np.maximum(self.z_max, individual.objectives)

            population.append(individual)

        return population

    def _environmental_selection(self, combined_pop, generation):
        """
        环境选择：基于APD的精英保留策略
        每个参考向量只保留APD最小的个体
        """
        # 为每个个体计算其关联的参考向量和APD
        associations = []  # (个体索引, 参考向量索引, APD值)

        for i, ind in enumerate(combined_pop):
            ref_idx, apd = self._associate_to_reference(ind.objectives, generation)
            associations.append((i, ref_idx, apd))
            ind.apd = apd

        # 按参考向量分组，每组保留APD最小的个体
        selected = []
        ref_groups = [[] for _ in range(self.n_ref)]

        for ind_idx, ref_idx, apd in associations:
            ref_groups[ref_idx].append((ind_idx, apd))

        # 每个参考向量选择APD最小的个体
        for group in ref_groups:
            if len(group) > 0:
                # 按APD排序，选择最小的
                group.sort(key=lambda x: x[1])
                selected_idx = group[0][0]
                selected.append(combined_pop[selected_idx])

        # 如果选择的个体少于pop_size，补充APD次优的个体
        if len(selected) < self.pop_size:
            # 收集所有未被选中的个体
            selected_set = set(id(ind) for ind in selected)
            remaining = [(i, ind.apd) for i, ind in enumerate(combined_pop) 
                        if id(ind) not in selected_set]
            remaining.sort(key=lambda x: x[1])

            needed = self.pop_size - len(selected)
            for i in range(min(needed, len(remaining))):
                selected.append(combined_pop[remaining[i][0]])

        # 如果仍然超过pop_size，截断
        if len(selected) > self.pop_size:
            # 按APD排序，保留pop_size个
            selected_with_apd = [(ind, ind.apd) for ind in selected]
            selected_with_apd.sort(key=lambda x: x[1])
            selected = [ind for ind, _ in selected_with_apd[:self.pop_size]]

        return selected

    def evolve(self):
        """进化过程 - 标准RVEA算法"""
        population = self.initialize_population()

        for generation in range(self.max_gen):
            offspring = []

            # 生成子代
            for i in range(self.pop_size):
                # 随机选择两个父代
                p1_idx = random.randint(0, self.pop_size - 1)
                p2_idx = random.randint(0, self.pop_size - 1)

                parent1 = population[p1_idx]
                parent2 = population[p2_idx]

                # 交叉和变异
                child_vars1, _ = self._sbx_crossover(parent1.variables, parent2.variables)
                child_vars = self._polynomial_mutation(child_vars1)

                # 评估子代
                child = Individual(child_vars)
                child.objectives = self.evaluate(child_vars)

                # 更新理想点和最差点
                self.z = np.minimum(self.z, child.objectives)
                self.z_max = np.maximum(self.z_max, child.objectives)

                offspring.append(child)

            # 合并父代和子代
            combined = population + offspring

            # 环境选择
            population = self._environmental_selection(combined, generation)

            # 参考向量自适应（每隔adaptive_gen代执行一次）
            if self.adaptive_gen > 0 and (generation + 1) % self.adaptive_gen == 0:
                self._reference_vector_adaptation(population)

            # 输出进度信息
            if (generation + 1) % 50 == 0:
                f1_values = [ind.objectives[0] for ind in population]
                f2_values = [ind.objectives[1] for ind in population]
                print(f"Generation {generation + 1}/{self.max_gen} | "
                      f"f1范围: [{min(f1_values):.4f}, {max(f1_values):.4f}] | "
                      f"f2范围: [{min(f2_values):.4f}, {max(f2_values):.4f}] | "
                      f"参考向量: {self.n_ref}")

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
        x1 = np.linspace(0, 1, n_points)
        f1 = 1 - np.exp(-4 * x1) * (np.sin(6 * np.pi * x1)) ** 6
        f2 = 1 - f1 ** 2
        sorted_idx = np.argsort(f1)
        f1 = f1[sorted_idx]
        f2 = f2[sorted_idx]
    else:
        raise ValueError(f"Unknown problem: {problem}")

    return np.column_stack([f1, f2])


def visualize_results(population, true_pareto, problem='zdt1', save_path=None):
    """可视化结果"""
    if save_path is None:
        save_path = f'{problem}_rvea_results.png'

    plt.figure(figsize=(12, 5))

    # 提取目标函数值
    f1 = [ind.objectives[0] for ind in population]
    f2 = [ind.objectives[1] for ind in population]

    # 子图1：算法结果
    plt.subplot(1, 2, 1)
    plt.scatter(f1, f2, c='red', alpha=0.6, s=20)
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title(f'RVEA算法结果 - {problem.upper()}')
    plt.grid(True, alpha=0.3)

    # 子图2：对比图
    plt.subplot(1, 2, 2)
    plt.scatter(f1, f2, c='red', alpha=0.6, s=20, label='RVEA结果')
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
    """在所有ZDT问题上运行RVEA算法

    参数:
    num_runs: 每个问题运行的次数（默认10次）
    """
    problems = ['zdt1', 'zdt2', 'zdt3', 'zdt6']

    # 参数配置（RVEA标准参数）
    n_vars = 30          # 变量维度
    pop_size = 100       # 种群大小（决定参考向量数量）
    max_gen = 300        # 最大迭代次数
    cx_prob = 0.9        # 交叉概率（RVEA通常较高）
    mut_prob = 0.1       # 变异概率

    # RVEA特有参数
    alpha = 2.0          # APD惩罚因子
    fr = 0.1             # 参考向量自适应频率

    # 存储所有运行的结果
    all_runs_results = {}
    last_run_results = {}

    print(f"\\n开始运行{num_runs}次独立实验...")
    print(f"参数设置: 种群大小={pop_size}, 迭代次数={max_gen}, 变量数={n_vars}")
    print(f"RVEA特有参数: alpha={alpha}, fr={fr}")
    print("=" * 60)

    for problem in problems:
        print(f"\\n{'='*60}")
        print(f"问题: {problem.upper()}")
        print(f"{'='*60}")

        problem_runs = []

        for run in range(num_runs):
            print(f"\\n运行第 {run+1}/{num_runs} 次")
            print(f"参数: 种群{pop_size}, 迭代{max_gen}, 交叉{cx_prob}, 变异{mut_prob}")

            rvea = RVEA(n_vars=n_vars, pop_size=pop_size, max_gen=max_gen,
                        cx_prob=cx_prob, mut_prob=mut_prob,
                        alpha=alpha, fr=fr, problem=problem)

            if run == 0:
                print(f"RVEA特有参数: alpha={alpha} (APD惩罚因子), fr={fr} (自适应频率)")
                print(f"参考向量数量: {rvea.n_ref}")

            final_population = rvea.evolve()

            # 性能分析
            f1 = [ind.objectives[0] for ind in final_population]
            f2 = [ind.objectives[1] for ind in final_population]
            print(f"运行 {run+1} 完成 - f1范围: [{min(f1):.4f}, {max(f1):.4f}], f2范围: [{min(f2):.4f}, {max(f2):.4f}]")

            problem_runs.append({
                'population': final_population,
                'f1_range': (min(f1), max(f1)),
                'f2_range': (min(f2), max(f2))
            })

            if run == num_runs - 1:
                true_pareto = get_true_pareto_front(problem)
                last_run_results[problem] = {
                    'population': final_population,
                    'true_pareto': true_pareto,
                    'f1_range': (min(f1), max(f1)),
                    'f2_range': (min(f2), max(f2))
                }

        all_runs_results[problem] = problem_runs

    # 可视化最后一次运行的结果
    print("\\n" + "="*60)
    print("可视化最后一次运行的结果...")
    print("="*60)
    for problem, result_data in last_run_results.items():
        visualize_results(result_data['population'], result_data['true_pareto'], problem)
        print(f"\\n{problem.upper()}结果分析:")
        print(f"f1范围: [{result_data['f1_range'][0]:.4f}, {result_data['f1_range'][1]:.4f}]")
        print(f"f2范围: [{result_data['f2_range'][0]:.4f}, {result_data['f2_range'][1]:.4f}]")

    print("\\n" + "="*50)
    print("所有问题优化完成！")
    print("="*50)

    # 保存结果到CSV
    print("\\n正在保存Pareto前沿坐标点到CSV文件...")
    save_rvea_results_to_csv(all_runs_results)
    print("所有数据保存完成！")

    return last_run_results


def save_rvea_results_to_csv(all_runs_results):
    """
    将RVEA的运行结果保存到CSV文件（追加模式）
    格式与MOEA/D代码保持一致
    """
    output_dir = "pareto_front_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    algorithm_name = "RVEA"

    for problem, runs_data in all_runs_results.items():
        print(f"\\n处理问题: {problem.upper()}")

        filename = os.path.join(output_dir, f"{problem}_{algorithm_name}_all_runs.csv")

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
            population = result['population']
            f1_values = [ind.objectives[0] for ind in population]
            f2_values = [ind.objectives[1] for ind in population]

            current_run = max_run + run_idx + 1
            for f1, f2 in zip(f1_values, f2_values):
                new_data.append({
                    'run': current_run,
                    'f1': f1,
                    'f2': f2
                })

        # 创建新的DataFrame
        df_new = pd.DataFrame(new_data)

        # 合并数据
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

    print(f"\\n所有数据已保存到 '{output_dir}' 目录")

def main():
    """主函数"""

    # 运行所有问题
    num_runs = 30
    results = run_all_problems(num_runs=num_runs)


if __name__ == "__main__":
    main()


# 保存到文件
#     output_path = '/mnt/agents/output/rvea_zdt.py'
#     with open(output_path, 'w', encoding='utf-8') as f:
#         f.write(rvea_code)
#
#     print(f"文件已保存到: {output_path}")
#
#     print(f"文件大小: {len(rvea_code)} 字符")

