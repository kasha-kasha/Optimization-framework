#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSGA-II, MOPSO, SPEA2, 改进NSGA-II，SPEA2
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev
from scipy.integrate import trapz
import time
import random
from datetime import datetime
import sys
import os
import pandas as pd

# 添加路径以导入算法模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nsga_std import NSGA2, Individual as NSGAIndividual
from mopso import MOPSO, Particle
from spea2 import SPEA2, Individual as SPEA2Individual
from imp_nsga import ImprovedNSGA2, Individual as ImprovedNSGA2Individual
from moead_std import MOEAD, Individual as MOEADIndividual

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 原始路径点（固定不变）
points = np.array([
    [-0.250, 0.000, 0.950],
    [-0.325, -0.275, 0.738],
    [-0.350, -0.200, 0.744],
    [-0.375, -0.125, 0.732],
    [-0.391, -0.078, 0.717],
    [-0.466, -0.135, 0.732],
    [-0.454, -0.186, 0.742],
    [-0.440, -0.245, 0.746],
    [-0.419, -0.282, 0.740]
])

# 固定z坐标（mid_point的z坐标）
MID_Z = 0.8


class TrajectoryEvaluator:
    """轨迹评估器：计算运行时间和关节冲击"""
    
    def __init__(self, fixed_points):
        """
        Parameters:
        -----------
        fixed_points : array_like
            固定的路径点，形状为 (n_points, 3)
        """
        self.fixed_points = np.array(fixed_points)
        self.n_segments = len(fixed_points) - 1
    
    def evaluate(self, mid_points_xy):
        """
        评估轨迹
        
        Parameters:
        -----------
        mid_points_xy : array_like
            mid_point的xy坐标，形状为 (n_segments, 2)
            每个mid_point的格式为 [x, y]
        
        Returns:
        --------
        objectives : tuple
            (execution_time, joint_impact)
        """
        try:
            # 构建完整路径点序列
            new_points = []
            for i in range(self.n_segments):
                p1 = self.fixed_points[i]
                # mid_point的xy来自优化变量，z固定为MID_Z
                mid_point = np.array([mid_points_xy[i][0], mid_points_xy[i][1], MID_Z])
                new_points.append(p1)
                new_points.append(mid_point)
            # 添加最后一个固定点
            new_points.append(self.fixed_points[-1])
            new_points = np.array(new_points)
            
            # 生成B样条轨迹
            trajectory_result = self._generate_bspline_trajectory(new_points)
            
            if trajectory_result is None:
                # 如果轨迹生成失败，返回惩罚值
                return (1e6, 1e6)
            
            # 计算运行时间（基于轨迹总长度和速度）
            execution_time = self._calculate_execution_time(trajectory_result)
            
            # 计算关节冲击（基于轨迹的jerk）
            joint_impact = self._calculate_joint_impact(trajectory_result)
            
            return (execution_time, joint_impact)
            
        except Exception as e:
            print(f"评估轨迹时出错: {e}")
            return (1e6, 1e6)
    
    def _generate_bspline_trajectory(self, points, num_points=200):
        """生成B样条轨迹"""
        try:
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            # B样条拟合（三次样条）
            tck, u = splprep([x, y, z], s=0, k=min(3, len(points)-1))
            u_new = np.linspace(0, 1, num_points)
            x_smooth, y_smooth, z_smooth = splev(u_new, tck)
            
            # 计算速度、加速度和jerk
            positions = np.column_stack([x_smooth, y_smooth, z_smooth])
            velocities = np.gradient(positions, axis=0) * num_points  # 归一化到时间
            accelerations = np.gradient(velocities, axis=0) * num_points
            jerks = np.gradient(accelerations, axis=0) * num_points
            
            return {
                'positions': positions,
                'velocities': velocities,
                'accelerations': accelerations,
                'jerks': jerks,
                'num_points': num_points
            }
        except Exception as e:
            print(f"生成B样条轨迹失败: {e}")
            return None
    
    def _calculate_execution_time(self, trajectory_result):
        """计算运行时间（基于轨迹长度和速度）"""
        try:
            positions = trajectory_result['positions']
            velocities = trajectory_result['velocities']
            
            # 计算轨迹总长度
            distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            total_length = np.sum(distances)
            
            # 计算平均速度
            speed_magnitudes = np.linalg.norm(velocities, axis=1)
            avg_speed = np.mean(speed_magnitudes[1:])  # 排除第一个点
            
            # 运行时间 = 总长度 / 平均速度
            if avg_speed > 1e-6:
                execution_time = total_length / avg_speed
            else:
                execution_time = total_length / 0.1  # 默认速度
            
            return execution_time
        except Exception as e:
            print(f"计算运行时间失败: {e}")
            return 1e6
    
    def _calculate_joint_impact(self, trajectory_result):
        """计算关节冲击（基于轨迹的jerk）"""
        try:
            jerks = trajectory_result['jerks']
            # 计算jerk的幅值
            jerk_magnitudes = np.linalg.norm(jerks, axis=1)
            
            # 关节冲击 = jerk的RMS值
            joint_impact = np.sqrt(np.mean(jerk_magnitudes**2))
            
            # 添加jerk平方积分项（惩罚高jerk）
            jerk_squared_integral = trapz(jerk_magnitudes**2, dx=1.0/trajectory_result['num_points'])
            joint_impact += 0.1 * jerk_squared_integral
            
            return joint_impact
        except Exception as e:
            print(f"计算关节冲击失败: {e}")
            return 1e6


# 创建全局评估器
evaluator = TrajectoryEvaluator(points)
n_segments = len(points) - 1


def get_variable_bounds():
    """获取优化变量边界"""
    # 每个mid_point的xy坐标范围
    # 基于相邻两个固定点的中点，允许一定的偏移
    bounds = []
    for i in range(n_segments):
        p1 = points[i]
        p2 = points[i + 1]
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
        # 允许±0.1的偏移
        bounds.append((mid_x - 0.1, mid_x + 0.1))  # x坐标
        bounds.append((mid_y - 0.1, mid_y + 0.1))  # y坐标
    return bounds


def objective_function(variables):
    """目标函数：将优化变量转换为目标值"""
    # 将一维变量数组转换为mid_points_xy格式
    mid_points_xy = []
    for i in range(n_segments):
        x = variables[i * 2]
        y = variables[i * 2 + 1]
        mid_points_xy.append([x, y])
    
    # 评估轨迹
    execution_time, joint_impact = evaluator.evaluate(mid_points_xy)
    
    return np.array([execution_time, joint_impact])


# ==========================================
# 算法适配器
# ==========================================

class MidPointNSGA2Adapter:
    """NSGA-II算法适配器"""
    
    def __init__(self, pop_size=50, max_gen=100):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.var_bounds = get_variable_bounds()
        self.n_vars = len(self.var_bounds)
        
        # 创建NSGA2实例
        self.nsga2 = NSGA2(
            n_vars=self.n_vars,
            pop_size=pop_size,
            max_gen=max_gen,
            cx_prob=0.9,
            mut_prob=1.0/self.n_vars
        )
        
        # 重写crossover方法以支持变量边界
        def crossover_with_bounds(parent1, parent2):
            if random.random() > self.nsga2.cx_prob:
                child1 = NSGAIndividual(parent1.variables.copy())
                child2 = NSGAIndividual(parent2.variables.copy())
                child1.objectives = objective_function(child1.variables)
                child2.objectives = objective_function(child2.variables)
                return child1, child2
            
            child1_vars = parent1.variables.copy()
            child2_vars = parent2.variables.copy()
            
            for i in range(self.n_vars):
                if random.random() <= 0.5:
                    if abs(parent1.variables[i] - parent2.variables[i]) > 1e-14:
                        y1, y2 = sorted([parent1.variables[i], parent2.variables[i]])
                        rand = random.random()
                        eta_c = 20
                        beta = (2 * rand) ** (1/(eta_c + 1)) if rand <= 0.5 else (1/(2*(1-rand))) ** (1/(eta_c + 1))
                        
                        c1 = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                        c2 = 0.5 * ((y1 + y2) + beta * (y2 - y1))
                        
                        lower, upper = self.var_bounds[i]
                        child1_vars[i] = np.clip(c1, lower, upper)
                        child2_vars[i] = np.clip(c2, lower, upper)
            
            child1 = NSGAIndividual(child1_vars)
            child2 = NSGAIndividual(child2_vars)
            child1.objectives = objective_function(child1_vars)
            child2.objectives = objective_function(child2_vars)
            return child1, child2
        
        # 重写mutation方法以支持变量边界
        def mutation_with_bounds(individual):
            if random.random() > self.nsga2.mut_prob:
                return individual
            
            mutated_vars = individual.variables.copy()
            eta_m = 20
            
            for i in range(self.n_vars):
                if random.random() <= 1.0/self.n_vars:
                    y = mutated_vars[i]
                    lower, upper = self.var_bounds[i]
                    
                    var_range = upper - lower
                    if var_range < 1e-10:
                        continue
                    
                    delta1 = (y - lower) / var_range
                    delta2 = (upper - y) / var_range
                    
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
                    
                    y = y + deltaq * var_range
                    mutated_vars[i] = np.clip(y, lower, upper)
            
            mutated = NSGAIndividual(mutated_vars)
            mutated.objectives = objective_function(mutated_vars)
            return mutated
        
        # 重写初始化方法
        def init_with_bounds():
            population = []
            for i in range(self.nsga2.pop_size):
                variables = np.array([np.random.uniform(lower, upper) 
                                     for lower, upper in self.var_bounds])
                individual = NSGAIndividual(variables)
                individual.objectives = objective_function(variables)
                population.append(individual)
            return population
        
        self.nsga2.crossover = crossover_with_bounds
        self.nsga2.mutation = mutation_with_bounds
        self.nsga2.initialize_population = init_with_bounds
        self.nsga2.evaluate_zdt1 = objective_function
    
    def optimize(self):
        """执行优化"""
        print("运行NSGA-II算法...")
        result = self.nsga2.evolve()
        
        # 提取Pareto前沿（rank=1的解）
        pareto_front = [ind for ind in result if ind.rank == 1]
        
        return pareto_front


class MidPointMOPSOAdapter:
    """MOPSO算法适配器"""
    
    def __init__(self, pop_size=50, max_gen=100):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.var_bounds = get_variable_bounds()
        self.n_vars = len(self.var_bounds)
        
        # 创建MOPSO实例
        self.mopso = MOPSO(
            n_vars=self.n_vars,
            pop_size=pop_size,
            max_gen=max_gen,
            c1=2.0,
            c2=2.0,
            w=0.9
        )
        
        # 保存原始初始化方法
        original_init = self.mopso.initialize_population
        
        # 重写初始化方法以支持变量边界
        def init_with_bounds():
            population = []
            for _ in range(self.mopso.pop_size):
                # 在变量边界内随机初始化位置
                position = np.array([np.random.uniform(lower, upper) 
                                     for lower, upper in self.var_bounds])
                # 计算速度范围（基于变量范围）
                velocity_ranges = [(upper - lower) * 0.1 for lower, upper in self.var_bounds]
                velocity = np.array([np.random.uniform(-vr, vr) for vr in velocity_ranges])
                particle = Particle(position, velocity)
                particle.objectives = objective_function(position)
                particle.pbest_objectives = particle.objectives.copy()
                particle.pbest_position = particle.position.copy()
                population.append(particle)
            return population
        
        # 保存原始更新位置方法
        original_update_position = self.mopso.update_position
        
        # 重写更新位置方法以支持变量边界
        def update_position_with_bounds(particle):
            # 更新位置
            particle.position = particle.position + particle.velocity
            # 限制在变量边界内
            for i, (lower, upper) in enumerate(self.var_bounds):
                particle.position[i] = np.clip(particle.position[i], lower, upper)
                # 如果位置被限制，将对应速度置零（反弹边界）
                if particle.position[i] <= lower or particle.position[i] >= upper:
                    particle.velocity[i] = 0
        
        # 设置目标函数
        def evaluate_trajectory(position):
            # 确保位置在变量边界内
            position = np.array(position)
            for i, (lower, upper) in enumerate(self.var_bounds):
                position[i] = np.clip(position[i], lower, upper)
            return objective_function(position)
        
        self.mopso.initialize_population = init_with_bounds
        self.mopso.update_position = update_position_with_bounds
        self.mopso.evaluate_zdt1 = evaluate_trajectory
    
    def optimize(self):
        """执行优化"""
        print("运行MOPSO算法...")
        result = self.mopso.evolve()
        
        # 提取Pareto前沿
        pareto_front = [p for p in result if p.rank == 1]
        
        # 如果Pareto前沿为空或解都相同，输出警告
        if len(pareto_front) > 0:
            objectives = [p.objectives for p in pareto_front]
            unique_objectives = set([tuple(obj) for obj in objectives])
            if len(unique_objectives) == 1:
                print(f"  警告: MOPSO找到的所有解都相同（可能是算法参数问题）")
        
        return pareto_front


class MidPointSPEA2Adapter:
    """SPEA2算法适配器"""
    
    def __init__(self, pop_size=50, max_gen=100):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.var_bounds = get_variable_bounds()
        self.n_vars = len(self.var_bounds)
        
        # 创建SPEA2实例
        self.spea2 = SPEA2(
            n_vars=self.n_vars,
            pop_size=pop_size,
            max_gen=max_gen,
            cx_prob=0.7,
            mut_prob=1.0/self.n_vars
        )
        
        # 重写crossover和mutation方法（类似NSGA2）
        def crossover_with_bounds(parent1, parent2):
            if random.random() > self.spea2.cx_prob:
                child1 = SPEA2Individual(parent1.variables.copy())
                child2 = SPEA2Individual(parent2.variables.copy())
                child1.objectives = objective_function(child1.variables)
                child2.objectives = objective_function(child2.variables)
                return child1, child2
            
            child1_vars = parent1.variables.copy()
            child2_vars = parent2.variables.copy()
            
            for i in range(self.n_vars):
                if random.random() <= 0.5:
                    if abs(parent1.variables[i] - parent2.variables[i]) > 1e-14:
                        y1, y2 = sorted([parent1.variables[i], parent2.variables[i]])
                        rand = random.random()
                        eta_c = 20
                        beta = (2 * rand) ** (1/(eta_c + 1)) if rand <= 0.5 else (1/(2*(1-rand))) ** (1/(eta_c + 1))
                        
                        c1 = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                        c2 = 0.5 * ((y1 + y2) + beta * (y2 - y1))
                        
                        lower, upper = self.var_bounds[i]
                        child1_vars[i] = np.clip(c1, lower, upper)
                        child2_vars[i] = np.clip(c2, lower, upper)
            
            child1 = SPEA2Individual(child1_vars)
            child2 = SPEA2Individual(child2_vars)
            child1.objectives = objective_function(child1_vars)
            child2.objectives = objective_function(child2_vars)
            return child1, child2
        
        def mutation_with_bounds(individual):
            if random.random() > self.spea2.mut_prob:
                return individual
            
            mutated_vars = individual.variables.copy()
            eta_m = 20
            
            for i in range(self.n_vars):
                if random.random() <= 1.0/self.n_vars:
                    y = mutated_vars[i]
                    lower, upper = self.var_bounds[i]
                    
                    var_range = upper - lower
                    if var_range < 1e-10:
                        continue
                    
                    delta1 = (y - lower) / var_range
                    delta2 = (upper - y) / var_range
                    
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
                    
                    y = y + deltaq * var_range
                    mutated_vars[i] = np.clip(y, lower, upper)
            
            mutated = SPEA2Individual(mutated_vars)
            mutated.objectives = objective_function(mutated_vars)
            return mutated
        
        def init_with_bounds():
            population = []
            for i in range(self.spea2.pop_size):
                variables = np.array([np.random.uniform(lower, upper) 
                                     for lower, upper in self.var_bounds])
                individual = SPEA2Individual(variables)
                individual.objectives = objective_function(variables)
                population.append(individual)
            return population
        
        self.spea2.crossover = crossover_with_bounds
        self.spea2.mutation = mutation_with_bounds
        self.spea2.initialize_population = init_with_bounds
        self.spea2.evaluate_zdt1 = objective_function
    
    def optimize(self):
        """执行优化"""
        print("运行SPEA2算法...")
        result = self.spea2.evolve()
        
        # 提取Pareto前沿（raw_fitness=0的解）
        pareto_front = [ind for ind in result if ind.raw_fitness == 0]
        
        return pareto_front


class MidPointImprovedNSGA2Adapter:
    """改进NSGA-II算法适配器（使用imp_nsga.py中的ImprovedNSGA2）"""
    
    def __init__(self, pop_size=50, max_gen=100, optimize_for_convergence=True):
        """
        Parameters:
        -----------
        optimize_for_convergence : bool
            如果True，优化参数以提升收敛性（减少MSOA开销，提高交叉概率）
            如果False，保持原始改进策略（注重多样性）
        """
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.var_bounds = get_variable_bounds()
        self.n_vars = len(self.var_bounds)
        self.optimize_for_convergence = optimize_for_convergence
        
        # 根据优化目标调整参数
        if optimize_for_convergence:
            # 优化收敛性：提高交叉概率，降低MSOA启动代数（减少局部搜索开销）
            cx_prob = 0.9  # 与标准NSGA-II一致
            msoa_start_gen_ratio = 0.5  # MSOA在50%代数后启动（原来33%）
        else:
            # 保持多样性优先策略
            cx_prob = 0.8
            msoa_start_gen_ratio = 1.0/3  # 原始设置
        
        # 使用imp_nsga.py中的ImprovedNSGA2类
        self.improved_nsga2 = ImprovedNSGA2(
            n_vars=self.n_vars,
            pop_size=pop_size,
            max_gen=max_gen,
            cx_prob=cx_prob,
            mut_prob=1.0/self.n_vars,
            var_bounds=self.var_bounds
        )
        
        # 调整MSOA启动代数（如果优化收敛性）
        if optimize_for_convergence:
            self.improved_nsga2.msoa_start_gen = int(max_gen * msoa_start_gen_ratio)
        
        # 设置目标函数（重写evaluate_zdt1方法）
        self.improved_nsga2.evaluate_zdt1 = objective_function
    
    def optimize(self):
        """执行优化"""
        print("运行改进NSGA-II算法...")
        if self.optimize_for_convergence:
            print(f"  优化模式：收敛性优先（cx_prob={self.improved_nsga2.cx_prob}, "
                  f"MSOA启动代数={self.improved_nsga2.msoa_start_gen})")
        else:
            print(f"  优化模式：多样性优先（cx_prob={self.improved_nsga2.cx_prob}, "
                  f"MSOA启动代数={self.improved_nsga2.msoa_start_gen})")
        print(f"  使用拉丁超立方初始化 + 自适应SBX + 定向SBX + MSOA局部搜索")
        
        # 运行改进算法
        result = self.improved_nsga2.evolve()
        
        # 提取Pareto前沿（rank=1的解）
        pareto_front = [ind for ind in result if ind.rank == 1]
        
        return pareto_front


class MidPointMOEADAdapter:
    """MOEA-D算法适配器"""
    
    def __init__(self, pop_size=50, max_gen=100):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.var_bounds = get_variable_bounds()
        self.n_vars = len(self.var_bounds)
        
        # 创建MOEA-D实例（注意：MOEA-D需要problem参数，但这里我们使用自定义目标函数）
        # 先创建一个临时实例以获取结构
        self.moead = MOEAD(
            n_vars=self.n_vars,
            pop_size=pop_size,
            max_gen=max_gen,
            cx_prob=0.8,
            mut_prob=1.0/self.n_vars,
            problem='zdt1'  # 临时值，会被重写
        )
        
        # 重写初始化方法以支持变量边界
        def init_with_bounds():
            population = []
            # 初始化参考点
            self.moead.z = np.full(self.moead.n_obj, 1e10)
            
            for _ in range(self.moead.pop_size):
                variables = np.array([np.random.uniform(lower, upper) 
                                     for lower, upper in self.var_bounds])
                individual = MOEADIndividual(variables)
                individual.objectives = objective_function(variables)
                # 更新参考点
                self.moead.z = np.minimum(self.moead.z, individual.objectives)
                population.append(individual)
            return population
        
        # 重写交叉方法以支持变量边界（使用闭包保存self引用）
        adapter_self = self  # 保存self引用
        def sbx_crossover_with_bounds(parent1, parent2):
            if random.random() > adapter_self.moead.cx_prob:
                return parent1.copy(), parent2.copy()
            
            child1 = parent1.copy()
            child2 = parent2.copy()
            eta_c = 20
            
            for i in range(adapter_self.n_vars):
                if random.random() <= 0.5:
                    if abs(parent1[i] - parent2[i]) > 1e-14:
                        y1, y2 = sorted([parent1[i], parent2[i]])
                        rand = random.random()
                        beta = (2 * rand) ** (1/(eta_c + 1)) if rand <= 0.5 else (1/(2*(1-rand))) ** (1/(eta_c + 1))
                        
                        c1 = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                        c2 = 0.5 * ((y1 + y2) + beta * (y2 - y1))
                        
                        lower, upper = adapter_self.var_bounds[i]
                        child1[i] = np.clip(c1, lower, upper)
                        child2[i] = np.clip(c2, lower, upper)
            
            return child1, child2
        
        # 重写变异方法以支持变量边界（使用闭包保存self引用）
        def polynomial_mutation_with_bounds(individual):
            if random.random() > adapter_self.moead.mut_prob:
                return individual.copy()
            
            mutated = individual.copy()
            eta_m = 20
            
            for i in range(adapter_self.n_vars):
                if random.random() <= 1.0/adapter_self.n_vars:
                    y = mutated[i]
                    lower, upper = adapter_self.var_bounds[i]
                    
                    var_range = upper - lower
                    if var_range < 1e-10:
                        continue
                    
                    delta1 = (y - lower) / var_range
                    delta2 = (upper - y) / var_range
                    
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
                    
                    y = y + deltaq * var_range
                    mutated[i] = np.clip(y, lower, upper)
            
            return mutated
        
        # 保存原始方法（如果需要）
        # 替换方法 - 使用闭包保存self引用
        self.moead._sbx_crossover = lambda p1, p2: sbx_crossover_with_bounds(p1, p2)
        self.moead._polynomial_mutation = polynomial_mutation_with_bounds
        self.moead.initialize_population = init_with_bounds
        
        # 重写evaluate方法以使用自定义目标函数
        original_evaluate = self.moead.evaluate
        def custom_evaluate(variables):
            return objective_function(variables)
        self.moead.evaluate = custom_evaluate
    
    def optimize(self):
        """执行优化"""
        print("运行MOEA-D算法...")
        result = self.moead.evolve()
        
        # MOEA-D返回的是整个种群，需要提取Pareto前沿
        # 使用非支配排序提取Pareto前沿
        pareto_front = []
        
        # 简单的非支配排序
        for i, ind1 in enumerate(result):
            is_dominated = False
            for j, ind2 in enumerate(result):
                if i != j:
                    # 检查ind2是否支配ind1
                    if all(ind2.objectives <= ind1.objectives) and any(ind2.objectives < ind1.objectives):
                        is_dominated = True
                        break
            if not is_dominated:
                pareto_front.append(ind1)
        
        return pareto_front


# ==========================================
# 主函数：运行所有算法并对比
# ==========================================

def run_all_algorithms():
    """运行所有算法并对比结果"""
    print("="*60)
    print("mid_point位置多目标优化 - 四种算法对比")
    print("="*60)
    print(f"优化变量数量: {len(get_variable_bounds())}")
    print(f"路径段数量: {n_segments}")
    print(f"固定路径点: {len(points)}")
    print()
    
    # 创建算法适配器
    # 改进NSGA-II使用收敛性优先模式（与标准NSGA-II对比）
    adapters = {
        'NSGA-II': MidPointNSGA2Adapter(pop_size=100, max_gen=300),
        'MOPSO': MidPointMOPSOAdapter(pop_size=100, max_gen=300),
        'SPEA2': MidPointSPEA2Adapter(pop_size=100, max_gen=300),
        '改进NSGA-II': MidPointImprovedNSGA2Adapter(pop_size=100, max_gen=300, optimize_for_convergence=True),
        'MOEA-D': MidPointMOEADAdapter(pop_size=100, max_gen=300)

    }
    
    results = {}
    
    # 运行每个算法
    for alg_name, adapter in adapters.items():
        print(f"\n{'='*60}")
        print(f"运行 {alg_name} 算法")
        print(f"{'='*60}")
        try:
            start_time = time.time()
            pareto_front = adapter.optimize()
            elapsed_time = time.time() - start_time
            
            results[alg_name] = {
                'pareto_front': pareto_front,
                'execution_time': elapsed_time,
                'pareto_count': len(pareto_front)
            }
            print(f"{alg_name} 完成，运行时间: {elapsed_time:.2f}秒，Pareto解数量: {len(pareto_front)}")
        except Exception as e:
            print(f"{alg_name} 运行失败: {str(e)}")
            import traceback
            traceback.print_exc()
            results[alg_name] = {
                'pareto_front': [],
                'execution_time': 0,
                'pareto_count': 0,
                'error': str(e)
            }
    
    # 保存轨迹数据到CSV文件
    save_trajectories_to_csv(results)
    

    
    return results




def get_solution_variables(solution):
    """统一获取解的优化变量（兼容不同算法类型）"""
    if hasattr(solution, 'variables'):
        return solution.variables
    elif hasattr(solution, 'position'):
        return solution.position
    else:
        raise AttributeError(f"解对象没有variables或position属性: {type(solution)}")



def save_trajectories_to_csv(results):
    """将四个算法的轨迹结果保存为CSV文件"""
    if not results:
        print("没有结果可以保存")
        return
    
    print("\n" + "="*60)
    print("保存轨迹数据到CSV文件...")
    print("="*60)
    
    # 从pareto前沿中选择代表性解（最小关节冲击的解）
    selected_solutions = {}
    for alg_name, result in results.items():
        pareto_front = result['pareto_front']
        if len(pareto_front) > 0:
            # 选择最小关节冲击的解
            best_solution = min(pareto_front, key=lambda x: x.objectives[1])
            selected_solutions[alg_name] = best_solution
    
    # 为每个算法保存轨迹数据
    for alg_name, solution in selected_solutions.items():
        try:
            # 从优化变量构建mid_points_xy（兼容不同算法类型）
            variables = get_solution_variables(solution)
            mid_points_xy = []
            for i in range(n_segments):
                x = variables[i * 2]
                y = variables[i * 2 + 1]
                mid_points_xy.append([x, y])
            
            # 构建完整路径点序列
            new_points = []
            for i in range(n_segments):
                p1 = points[i]
                mid_point = np.array([mid_points_xy[i][0], mid_points_xy[i][1], MID_Z])
                new_points.append(p1)
                new_points.append(mid_point)
            new_points.append(points[-1])
            new_points = np.array(new_points)
            
            # 生成B样条轨迹
            trajectory_result = evaluator._generate_bspline_trajectory(new_points, num_points=200)
            
            if trajectory_result is not None:
                positions = trajectory_result['positions']
                velocities = trajectory_result['velocities']
                accelerations = trajectory_result['accelerations']
                jerks = trajectory_result['jerks']
                
                # 计算时间序列（归一化到0-1，然后按实际时间缩放）
                num_points = len(positions)
                t_normalized = np.linspace(0, 1, num_points)
                
                # 计算实际时间（基于运行时间）
                execution_time = solution.objectives[0]
                t_actual = t_normalized * execution_time
                
                # 计算速度、加速度、jerk的幅值
                velocity_magnitudes = np.linalg.norm(velocities, axis=1)
                acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
                jerk_magnitudes = np.linalg.norm(jerks, axis=1)
                
                # 构建DataFrame
                trajectory_data = {
                    'time': t_actual,
                    'time_normalized': t_normalized,
                    'x': positions[:, 0],
                    'y': positions[:, 1],
                    'z': positions[:, 2],
                    'vx': velocities[:, 0],
                    'vy': velocities[:, 1],
                    'vz': velocities[:, 2],
                    'velocity_magnitude': velocity_magnitudes,
                    'ax': accelerations[:, 0],
                    'ay': accelerations[:, 1],
                    'az': accelerations[:, 2],
                    'acceleration_magnitude': acceleration_magnitudes,
                    'jx': jerks[:, 0],
                    'jy': jerks[:, 1],
                    'jz': jerks[:, 2],
                    'jerk_magnitude': jerk_magnitudes
                }
                
                df = pd.DataFrame(trajectory_data)
                
                # 创建文件名（去除特殊字符，替换为下划线）
                safe_alg_name = alg_name.replace('(', '_').replace(')', '_').replace(' ', '_')
                csv_filename = f"trajectory_{safe_alg_name}.csv"
                
                # 保存到CSV文件（添加错误处理，处理文件被占用的情况）
                try:
                    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
                    print(f"✓ {alg_name}: 轨迹数据已保存到 {csv_filename}")
                except PermissionError:
                    # 如果文件被占用，尝试添加时间戳
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv_filename_alt = f"trajectory_{safe_alg_name}_{timestamp}.csv"
                    df.to_csv(csv_filename_alt, index=False, encoding='utf-8-sig')
                    print(f"⚠ {alg_name}: 原文件被占用，轨迹数据已保存到 {csv_filename_alt}")
                    csv_filename = csv_filename_alt  # 更新文件名用于后续使用
                print(f"  数据点数量: {len(df)}")
                print(f"  关节冲击: {solution.objectives[1]:.4f}")
                print(f"  运行时间: {solution.objectives[0]:.4f}s")
                print(f"  轨迹长度: {np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)):.4f}m")
                
                # 同时保存优化变量信息到单独的CSV文件
                optimization_info = {
                    'algorithm': [alg_name],
                    'execution_time': [solution.objectives[0]],
                    'joint_impact': [solution.objectives[1]],
                    'trajectory_length': [np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))],
                    'num_points': [num_points]
                }
                
                # 添加优化变量（mid_point的xy坐标）
                for i in range(n_segments):
                    optimization_info[f'mid_point_{i+1}_x'] = [mid_points_xy[i][0]]
                    optimization_info[f'mid_point_{i+1}_y'] = [mid_points_xy[i][1]]
                
                df_info = pd.DataFrame(optimization_info)
                info_filename = f"optimization_info_{safe_alg_name}.csv"
                try:
                    df_info.to_csv(info_filename, index=False, encoding='utf-8-sig')
                    print(f"  优化信息已保存到 {info_filename}")
                except PermissionError:
                    # 如果文件被占用，尝试添加时间戳
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    info_filename_alt = f"optimization_info_{safe_alg_name}_{timestamp}.csv"
                    df_info.to_csv(info_filename_alt, index=False, encoding='utf-8-sig')
                    print(f"  原文件被占用，优化信息已保存到 {info_filename_alt}")
                
            else:
                print(f"✗ {alg_name}: 轨迹生成失败，无法保存")
                
        except Exception as e:
            print(f"✗ {alg_name}: 保存轨迹数据时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n所有轨迹数据保存完成！")




if __name__ == "__main__":
    # 运行所有算法
    results = run_all_algorithms()
