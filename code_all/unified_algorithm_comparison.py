import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import random
import matplotlib
import time
from scipy.spatial.distance import cdist
import pandas as pd
import os

# 设置中文字体和绘图参数
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入四个算法
from mopso import MOPSO, Particle
from imp_nsga import ImprovedNSGA2, Individual as ImpIndividual
from nsga_std import NSGA2, Individual as NSGAIndividual
from spea2 import SPEA2, Individual as SPEA2Individual

class ZDTTestFunctions:
    """ZDT测试函数集合"""
    
    @staticmethod
    def zdt1(variables, n_vars=30):
        """ZDT1: 凸Pareto前沿"""
        f1 = variables[0]
        g = 1 + 9 * np.sum(variables[1:]) / (n_vars - 1)
        f2 = g * (1 - np.sqrt(f1 / g))
        return np.array([f1, f2])
    
    @staticmethod
    def zdt2(variables, n_vars=30):
        """ZDT2: 非凸Pareto前沿"""
        f1 = variables[0]
        g = 1 + 9 * np.sum(variables[1:]) / (n_vars - 1)
        f2 = g * (1 - (f1 / g) ** 2)
        return np.array([f1, f2])
    
    @staticmethod
    def zdt3(variables, n_vars=30):
        """ZDT3: 不连续Pareto前沿"""
        f1 = variables[0]
        g = 1 + 9 * np.sum(variables[1:]) / (n_vars - 1)
        f2 = g * (1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1))
        return np.array([f1, f2])
    
    @staticmethod
    def zdt4(variables, n_vars=10):
        """ZDT4: 多模态问题"""
        f1 = variables[0]
        g = 1 + 10 * (n_vars - 1) + np.sum(variables[1:] ** 2 - 10 * np.cos(4 * np.pi * variables[1:]))
        f2 = g * (1 - np.sqrt(f1 / g))
        return np.array([f1, f2])
    
    @staticmethod
    def zdt6(variables, n_vars=10):
        """ZDT6: 非均匀分布"""
        f1 = 1 - np.exp(-4 * variables[0]) * (np.sin(6 * np.pi * variables[0])) ** 6
        g = 1 + 9 * (np.sum(variables[1:]) / (n_vars - 1)) ** 0.25
        f2 = g * (1 - (f1 / g) ** 2)
        return np.array([f1, f2])
    
    @staticmethod
    def get_true_pareto_front(function_name, n_points=1000):
        """获取真实Pareto前沿"""
        if function_name == 'zdt1':
            f1 = np.linspace(0, 1, n_points)
            f2 = 1 - np.sqrt(f1)
        elif function_name == 'zdt2':
            f1 = np.linspace(0, 1, n_points)
            f2 = 1 - f1 ** 2
        elif function_name == 'zdt3':
            f1 = np.linspace(0, 1, n_points)
            f2 = 1 - np.sqrt(f1) - f1 * np.sin(10 * np.pi * f1)
        elif function_name == 'zdt4':
            f1 = np.linspace(0, 1, n_points)
            f2 = 1 - np.sqrt(f1)
        elif function_name == 'zdt6':
            f1 = np.linspace(0, 1, n_points)
            f2 = 1 - f1 ** 2
        else:
            raise ValueError(f"不支持的测试函数: {function_name}")
        
        return np.column_stack([f1, f2])

class UnifiedAlgorithmRunner:
    """统一算法运行器"""
    
    def __init__(self, pop_size=50, max_gen=100, n_vars=30):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.n_vars = n_vars
        self.zdt_functions = ZDTTestFunctions()
    
    def get_variable_bounds(self, problem):
        """获取不同问题的变量边界"""
        if problem in ['zdt1', 'zdt2', 'zdt3']:
            # ZDT1/2/3: 所有变量都在[0, 1]
            return [(0.0, 1.0)] * self.n_vars
        elif problem == 'zdt4':
            # ZDT4: x1∈[0,1], x2...x10∈[-5,5]
            bounds = [(0.0, 1.0)]
            bounds.extend([(-5.0, 5.0)] * (self.n_vars - 1))
            return bounds
        elif problem == 'zdt6':
            # ZDT6: 所有变量都在[0, 1]
            return [(0.0, 1.0)] * self.n_vars
        else:
            return [(0.0, 1.0)] * self.n_vars
        
    def run_mopso(self, problem='zdt1'):
        """运行MOPSO算法"""
        print(f"运行MOPSO算法求解{problem.upper()}问题...")
        
        # 创建MOPSO实例
        mopso = MOPSO(n_vars=self.n_vars, pop_size=self.pop_size, max_gen=self.max_gen)
        
        # 修改目标函数
        if problem == 'zdt1':
            mopso.evaluate_zdt1 = lambda pos: self.zdt_functions.zdt1(pos, self.n_vars)
        elif problem == 'zdt2':
            mopso.evaluate_zdt1 = lambda pos: self.zdt_functions.zdt2(pos, self.n_vars)
        elif problem == 'zdt3':
            mopso.evaluate_zdt1 = lambda pos: self.zdt_functions.zdt3(pos, self.n_vars)
        elif problem == 'zdt4':
            mopso.evaluate_zdt1 = lambda pos: self.zdt_functions.zdt4(pos, self.n_vars)
        elif problem == 'zdt6':
            mopso.evaluate_zdt1 = lambda pos: self.zdt_functions.zdt6(pos, self.n_vars)
        
        start_time = time.time()
        result = mopso.evolve()
        end_time = time.time()
        
        return {
            'algorithm': 'MOPSO',
            'problem': problem,
            'result': result,
            'runtime': end_time - start_time,
            'pop_size': self.pop_size,
            'max_gen': self.max_gen
        }
    
    def run_improved_nsga2(self, problem='zdt1'):
        """运行改进NSGA-II算法"""
        print(f"运行改进NSGA-II算法求解{problem.upper()}问题...")
        
        # 获取变量边界
        var_bounds = self.get_variable_bounds(problem)
        
        # 创建改进NSGA-II实例（传入变量边界）
        improved_nsga2 = ImprovedNSGA2(n_vars=self.n_vars, pop_size=self.pop_size, max_gen=self.max_gen, var_bounds=var_bounds)
        
        # 修改目标函数
        if problem == 'zdt1':
            improved_nsga2.evaluate_zdt1 = lambda vars: self.zdt_functions.zdt1(vars, self.n_vars)
        elif problem == 'zdt2':
            improved_nsga2.evaluate_zdt1 = lambda vars: self.zdt_functions.zdt2(vars, self.n_vars)
        elif problem == 'zdt3':
            improved_nsga2.evaluate_zdt1 = lambda vars: self.zdt_functions.zdt3(vars, self.n_vars)
        elif problem == 'zdt4':
            improved_nsga2.evaluate_zdt1 = lambda vars: self.zdt_functions.zdt4(vars, self.n_vars)
        elif problem == 'zdt6':
            improved_nsga2.evaluate_zdt1 = lambda vars: self.zdt_functions.zdt6(vars, self.n_vars)
        
        start_time = time.time()
        result = improved_nsga2.evolve()
        end_time = time.time()
        
        return {
            'algorithm': '改进NSGA-II',
            'problem': problem,
            'result': result,
            'runtime': end_time - start_time,
            'pop_size': self.pop_size,
            'max_gen': self.max_gen
        }
    
    def run_nsga2(self, problem='zdt1'):
        """运行标准NSGA-II算法"""
        print(f"运行标准NSGA-II算法求解{problem.upper()}问题...")
        
        # 创建标准NSGA-II实例
        nsga2 = NSGA2(n_vars=self.n_vars, pop_size=self.pop_size, max_gen=self.max_gen)
        
        # 修改目标函数
        if problem == 'zdt1':
            nsga2.evaluate_zdt1 = lambda vars: self.zdt_functions.zdt1(vars, self.n_vars)
        elif problem == 'zdt2':
            nsga2.evaluate_zdt1 = lambda vars: self.zdt_functions.zdt2(vars, self.n_vars)
        elif problem == 'zdt3':
            nsga2.evaluate_zdt1 = lambda vars: self.zdt_functions.zdt3(vars, self.n_vars)
        elif problem == 'zdt4':
            nsga2.evaluate_zdt1 = lambda vars: self.zdt_functions.zdt4(vars, self.n_vars)
        elif problem == 'zdt6':
            nsga2.evaluate_zdt1 = lambda vars: self.zdt_functions.zdt6(vars, self.n_vars)
        
        start_time = time.time()
        result = nsga2.evolve()
        end_time = time.time()
        
        return {
            'algorithm': '标准NSGA-II',
            'problem': problem,
            'result': result,
            'runtime': end_time - start_time,
            'pop_size': self.pop_size,
            'max_gen': self.max_gen
        }
    
    def run_spea2(self, problem='zdt1'):
        """运行SPEA2算法"""
        print(f"运行SPEA2算法求解{problem.upper()}问题...")
        
        # 创建SPEA2实例
        spea2 = SPEA2(n_vars=self.n_vars, pop_size=self.pop_size, max_gen=self.max_gen)
        
        # 修改目标函数
        if problem == 'zdt1':
            spea2.evaluate_zdt1 = lambda vars: self.zdt_functions.zdt1(vars, self.n_vars)
        elif problem == 'zdt2':
            spea2.evaluate_zdt1 = lambda vars: self.zdt_functions.zdt2(vars, self.n_vars)
        elif problem == 'zdt3':
            spea2.evaluate_zdt1 = lambda vars: self.zdt_functions.zdt3(vars, self.n_vars)
        elif problem == 'zdt4':
            spea2.evaluate_zdt1 = lambda vars: self.zdt_functions.zdt4(vars, self.n_vars)
        elif problem == 'zdt6':
            spea2.evaluate_zdt1 = lambda vars: self.zdt_functions.zdt6(vars, self.n_vars)
        
        start_time = time.time()
        result = spea2.evolve()
        end_time = time.time()
        
        return {
            'algorithm': 'SPEA2',
            'problem': problem,
            'result': result,
            'runtime': end_time - start_time,
            'pop_size': self.pop_size,
            'max_gen': self.max_gen
        }
    
    def run_algorithm(self, algorithm_name, problem='zdt1'):
        """运行指定算法"""
        algorithm_name_lower = algorithm_name.lower()
        
        if algorithm_name_lower == 'mopso':
            return self.run_mopso(problem)
        elif (algorithm_name_lower == 'improved_nsga2' or 
              algorithm_name_lower == 'imp_nsga' or 
              algorithm_name_lower == '改进nsga-ii' or
              algorithm_name_lower == '改进nsga2'):
            return self.run_improved_nsga2(problem)
        elif (algorithm_name_lower == 'nsga2' or 
              algorithm_name_lower == 'nsga_std' or
              algorithm_name_lower == '标准nsga-ii' or
              algorithm_name_lower == '标准nsga2'):
            return self.run_nsga2(problem)
        elif algorithm_name_lower == 'spea2':
            return self.run_spea2(problem)
        else:
            raise ValueError(f"不支持的算法: {algorithm_name}")
    
    def run_all_algorithms(self, problem='zdt1'):
        """运行所有算法"""
        algorithms = ['MOPSO', '改进NSGA-II', '标准NSGA-II', 'SPEA2']
        results = []
        
        print(f"\n开始求解{problem.upper()}问题...")
        print(f"参数设置: 种群大小={self.pop_size}, 迭代次数={self.max_gen}, 变量数={self.n_vars}")
        print("=" * 60)
        
        for alg in algorithms:
            try:
                result = self.run_algorithm(alg, problem)
                results.append(result)
                print(f"{alg}完成，运行时间: {result['runtime']:.2f}秒")
            except Exception as e:
                print(f"{alg}运行失败: {str(e)}")
        
        return results
    
    def visualize_results(self, results, problem='zdt1', save_path=None):
        """可视化结果"""
        if not results:
            print("没有结果可以可视化")
            return
        
        # 获取真实Pareto前沿
        true_pareto = self.zdt_functions.get_true_pareto_front(problem)
        
        # 创建子图
        n_algorithms = len(results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, result in enumerate(results):
            ax = axes[i]
            
            # 提取目标函数值
            if result['algorithm'] == 'MOPSO':
                f1 = [particle.objectives[0] for particle in result['result']]
                f2 = [particle.objectives[1] for particle in result['result']]
            else:
                f1 = [ind.objectives[0] for ind in result['result']]
                f2 = [ind.objectives[1] for ind in result['result']]
            
            # 绘制结果
            ax.scatter(f1, f2, c=colors[i], alpha=0.6, s=20, label=f"{result['algorithm']}结果")
            ax.plot(true_pareto[:, 0], true_pareto[:, 1], 'k-', linewidth=2, label='真实Pareto前沿')
            ax.set_xlabel('f1')
            ax.set_ylabel('f2')
            ax.set_title(f"{result['algorithm']} - {problem.upper()}")
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # 隐藏多余的子图
        for i in range(n_algorithms, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{problem}_四种算法对比.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
        print(f"结果已保存到: {save_path}")
        
        # 创建对比图
        self.create_comparison_plot(results, true_pareto, problem)
    
    def create_comparison_plot(self, results, true_pareto, problem='zdt1'):
        """创建对比图"""
        plt.figure(figsize=(12, 8))
        
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, result in enumerate(results):
            # 提取目标函数值
            if result['algorithm'] == 'MOPSO':
                f1 = [particle.objectives[0] for particle in result['result']]
                f2 = [particle.objectives[1] for particle in result['result']]
            else:
                f1 = [ind.objectives[0] for ind in result['result']]
                f2 = [ind.objectives[1] for ind in result['result']]
            
            plt.scatter(f1, f2, c=colors[i], alpha=0.6, s=20, label=f"{result['algorithm']}")
        
        plt.plot(true_pareto[:, 0], true_pareto[:, 1], 'k-', linewidth=3, label='真实Pareto前沿')
        plt.xlabel('f1', fontsize=12)
        plt.ylabel('f2', fontsize=12)
        plt.title(f'{problem.upper()}问题 - 四种算法Pareto前沿对比', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        comparison_path = f"{problem}_四种算法Pareto前沿对比.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        # plt.show()
        print(f"对比图已保存到: {comparison_path}")
    
    def print_performance_summary(self, results):
        """打印性能摘要"""
        print("\n" + "=" * 60)
        print("性能摘要")
        print("=" * 60)
        
        for result in results:
            print(f"\n{result['algorithm']}:")
            print(f"  运行时间: {result['runtime']:.2f}秒")
            print(f"  种群大小: {result['pop_size']}")
            print(f"  迭代次数: {result['max_gen']}")
            
            # 提取目标函数值
            if result['algorithm'] == 'MOPSO':
                f1 = [particle.objectives[0] for particle in result['result']]
                f2 = [particle.objectives[1] for particle in result['result']]
                print(f"  档案大小: {len(result['result'])}")
            else:
                f1 = [ind.objectives[0] for ind in result['result']]
                f2 = [ind.objectives[1] for ind in result['result']]
                print(f"  种群大小: {len(result['result'])}")
            
            print(f"  f1范围: [{min(f1):.4f}, {max(f1):.4f}]")
            print(f"  f2范围: [{min(f2):.4f}, {max(f2):.4f}]")

def main():
    """主函数"""
    print("support:  MOPSO, improve_NSGA-II, std_NSGA-II, SPEA2")
    print("support: ZDT1, ZDT2, ZDT3, ZDT4, ZDT6")
    print("=" * 60)
    problem_list = [ 'zdt1','zdt2','zdt3','zdt4','zdt6']
    for problem in problem_list:

        # 直接在代码中设置参数
        pop_size = 100        # 种群大小
        max_gen = 300         # 迭代次数
        n_vars = 30           # 变量数
        # problem = 'zdt4'      # 测试函数
        num_runs = 30        # 运行次数

        print(f"\n参数设置:")
        print(f"种群大小: {pop_size}")
        print(f"迭代次数: {max_gen}")
        print(f"变量数: {n_vars}")
        print(f"测试函数: {problem.upper()}")
        print(f"运行次数: {num_runs}")
        print("=" * 30)

        # 运行多次独立实验，每次都保存结果
        print(f"\n开始运行{num_runs}次独立实验...")
        all_results = {}  # 存储所有运行的结果
        last_results = None  # 保存最后一次的结果用于可视化

        for run in range(num_runs):
            print(f"\n{'='*60}")
            print(f"运行第 {run+1}/{num_runs} 次")
            print(f"{'='*60}")

            # 运行所有算法
            runner = UnifiedAlgorithmRunner(pop_size=pop_size, max_gen=max_gen, n_vars=n_vars)
            results = runner.run_all_algorithms(problem)

            # 保存最后一次运行的结果用于可视化
            last_results = results

            # 将结果添加到所有结果中（使用setdefault避免KeyError）
            for result in results:
                algorithm = result['algorithm']
                if algorithm not in all_results:
                    all_results[algorithm] = []
                all_results[algorithm].append(result)

        # 可视化最后一次运行的结果
        if last_results is not None:
            print("\n正在可视化最后一次运行的结果...")
            # 创建runner用于可视化
            viz_runner = UnifiedAlgorithmRunner(pop_size=pop_size, max_gen=max_gen, n_vars=n_vars)
            viz_runner.visualize_results(last_results, problem)
            viz_runner.print_performance_summary(last_results)

        # 保存Pareto前沿坐标点到CSV文件
        print("\n正在保存Pareto前沿坐标点到CSV文件...")
        save_all_runs_to_csv(all_results, problem, num_runs)
        print("所有数据保存完成！")

def save_all_runs_to_csv(all_results, problem, num_runs):
    """
    将所有运行的结果保存到CSV文件（追加模式，不覆盖已有数据）
    每种算法一个表格，包含所有运行的Pareto前沿坐标点
    """
    # 创建输出目录
    output_dir = "pareto_front_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 处理每种算法
    for algorithm_name, runs_data in all_results.items():
        print(f"\n处理算法: {algorithm_name}")
        
        filename = f"{output_dir}/{problem}_{algorithm_name}_all_runs.csv"
        
        # 检查文件是否存在
        existing_data = None
        max_run = 0
        
        if os.path.exists(filename):
            print(f"  读取现有文件: {filename}")
            existing_data = pd.read_csv(filename, encoding='utf-8-sig')
            max_run = existing_data['run'].max()
            print(f"  现有最大运行号: {max_run}")
        
        # 存储当前运行的数据
        new_data = []
        
        for run_idx, result in enumerate(runs_data):
            # 提取目标函数值
            if algorithm_name == 'MOPSO':
                f1_values = [particle.objectives[0] for particle in result['result']]
                f2_values = [particle.objectives[1] for particle in result['result']]
            else:
                f1_values = [ind.objectives[0] for ind in result['result']]
                f2_values = [ind.objectives[1] for ind in result['result']]
            
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

if __name__ == "__main__":
    main()
