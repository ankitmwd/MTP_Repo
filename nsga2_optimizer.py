"""
NSGA-II Multi-Objective Optimization for EVCS Placement

This module implements the Non-dominated Sorting Genetic Algorithm II (NSGA-II)
to optimize multiple objectives simultaneously:
1. Minimize total cost (setup + operating)
2. Maximize coverage (demand served)
3. Minimize average distance to charging stations

Reference: Deb, K., et al. (2002). A fast and elitist multiobjective 
           genetic algorithm: NSGA-II.

Author: EVCS Optimization Team
Date: 2024
"""

import numpy as np
from typing import List, Tuple, Dict
from deap import base, creator, tools, algorithms
import random
import warnings
warnings.filterwarnings('ignore')


class NSGA2Optimizer:
    """
    NSGA-II optimizer for multi-objective EVCS placement.
    
    Objectives:
    - Minimize total cost
    - Maximize demand coverage
    - Minimize average service distance
    """
    
    def __init__(self, data: Dict, 
                 population_size: int = 50,
                 n_generations: int = 100,
                 crossover_prob: float = 0.9,
                 mutation_prob: float = 0.1):
        """
        Initialize NSGA-II optimizer.
        
        Parameters:
        -----------
        data : Dict
            Dictionary containing demand_zones, candidate_sites, distance_matrix
        population_size : int
            Size of genetic algorithm population
        n_generations : int
            Number of generations to evolve
        crossover_prob : float
            Probability of crossover
        mutation_prob : float
            Probability of mutation
        """
        self.data = data
        self.demand_zones = data['demand_zones']
        self.candidate_sites = data['candidate_sites']
        self.distance_matrix = data['distance_matrix']
        
        self.n_zones = len(self.demand_zones)
        self.n_sites = len(self.candidate_sites)
        self.population_size = population_size
        self.n_generations = n_generations
        
        # Problem parameters
        self.max_service_distance = 5.0  # km
        self.operating_cost_per_kwh = 4.0  # INR per kWh
        self.budget = 50000000  # Total budget in INR
        
        # Setup DEAP
        self._setup_deap()
        
    def _setup_deap(self):
        """Setup DEAP framework for NSGA-II."""
        # Create fitness classes (with error handling for multiple imports)
        try:
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, -1.0))
        except RuntimeError:
            # Already exists, delete and recreate
            if hasattr(creator, "FitnessMulti"):
                del creator.FitnessMulti
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, -1.0))
        
        # Negative weights: minimize cost, maximize coverage, minimize distance
        try:
            creator.create("Individual", list, fitness=creator.FitnessMulti)
        except RuntimeError:
            if hasattr(creator, "Individual"):
                del creator.Individual
            creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        self.toolbox = base.Toolbox()
        
        # Individual: binary vector (1 = site selected, 0 = not selected)
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                             self.toolbox.attr_bool, n=self.n_sites)
        self.toolbox.register("population", tools.initRepeat, list, 
                             self.toolbox.individual)
        
        # Genetic operators
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("select", tools.selNSGA2)
        
    def _evaluate_individual(self, individual: List[int]) -> Tuple[float, float, float]:
        """
        Evaluate an individual (site selection) on three objectives.
        
        Parameters:
        -----------
        individual : List[int]
            Binary vector representing site selection
            
        Returns:
        --------
        Tuple[float, float, float]
            (total_cost, -coverage, avg_distance)
            Note: coverage is negated because we maximize it
        """
        selected_sites = np.array(individual)
        selected_indices = np.where(selected_sites == 1)[0]
        
        # Objective 1: Total cost
        total_cost = 0.0
        cost_column = 'total_setup_cost' if 'total_setup_cost' in self.candidate_sites.columns else 'setup_cost'
        for j in selected_indices:
            total_cost += self.candidate_sites.iloc[j][cost_column]
        
        # Budget constraint penalty
        if total_cost > self.budget:
            return (1e10, -1e10, 1e10)  # Penalize infeasible solutions
        
        # Objective 2: Coverage (demand served)
        total_coverage = 0.0
        total_distance = 0.0
        zones_covered = 0
        
        for i in range(self.n_zones):
            zone_demand = self.demand_zones.iloc[i]['demand']
            min_distance = np.inf
            served = False
            
            # Find nearest selected site
            for j in selected_indices:
                dist = self.distance_matrix[i, j]
                if dist <= self.max_service_distance:
                    served = True
                    min_distance = min(min_distance, dist)
            
            if served:
                total_coverage += zone_demand
                total_distance += min_distance
                zones_covered += 1
        
        # Objective 3: Average distance (only for covered zones)
        avg_distance = total_distance / max(zones_covered, 1)
        
        # Return: (cost, -coverage, avg_distance)
        # Coverage negated because DEAP maximizes, but we want to maximize coverage
        return (total_cost, -total_coverage, avg_distance)
    
    def solve(self) -> Dict:
        """
        Solve multi-objective optimization using NSGA-II.
        
        Returns:
        --------
        Dict
            Pareto-optimal solutions with objectives and site selections
        """
        print("\n=== NSGA-II Multi-Objective Optimizer ===")
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.n_generations}")
        
        # Initialize population
        population = self.toolbox.population(n=self.population_size)
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Evolution loop
        for generation in range(self.n_generations):
            # Select parents
            offspring = algorithms.varAnd(population, self.toolbox, 
                                         cxpb=0.9, mutpb=0.1)
            
            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Select next generation
            population = self.toolbox.select(offspring + population, 
                                            self.population_size)
            
            if generation % 20 == 0:
                # Print statistics
                fits = [ind.fitness.values for ind in population]
                costs = [f[0] for f in fits]
                coverages = [-f[1] for f in fits]  # Negate back
                distances = [f[2] for f in fits]
                
                print(f"  Generation {generation}:")
                print(f"    Cost: {np.mean(costs):.2f} ± {np.std(costs):.2f}")
                print(f"    Coverage: {np.mean(coverages):.2f} ± {np.std(coverages):.2f}")
                print(f"    Distance: {np.mean(distances):.2f} ± {np.std(distances):.2f}")
        
        # Extract Pareto front
        pareto_front = tools.sortNondominated(population, len(population),
                                            first_front_only=True)[0]
        
        print(f"\n[OK] Found {len(pareto_front)} Pareto-optimal solutions")
        
        # Convert to solution format
        solutions = []
        for ind in pareto_front:
            selected_sites = np.array(ind)
            cost, neg_coverage, avg_dist = ind.fitness.values
            coverage = -neg_coverage  # Convert back
            
            solutions.append({
                'selected_sites': selected_sites,
                'cost': cost,
                'coverage': coverage,
                'avg_distance': avg_dist,
                'n_sites': np.sum(selected_sites)
            })
        
        return {
            'pareto_solutions': solutions,
            'population': population
        }

