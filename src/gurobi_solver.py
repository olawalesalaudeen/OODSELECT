import numpy as np
import gurobipy as gp
from gurobipy import GRB
from gurobipy import nlfunc
from tqdm import tqdm

class GurobiSolver:
    """
    Gurobi-based solver for optimal OOD sample selection.

    Uses mixed-integer programming to find the set of OOD samples that maximizes
    correlation with ID accuracy.
    """

    def __init__(self, X_train_tensor, transformed_y_train, time_limit=300, disp=0):
        """
        Initialize the Gurobi solver.

        Args:
            X_train_tensor: Binary correctness matrix of shape (# models, # samples)
            transformed_y_train: ID accuracies after probit transform, shape (# models,)
            time_limit: Maximum time for optimization (seconds)
            disp: Display flag for Gurobi output (0=quiet, 1=verbose)
        """
        self.X_train_tensor = X_train_tensor
        self.num_ood_samples = X_train_tensor.shape[1]
        self.num_models_train = X_train_tensor.shape[0]
        self.transformed_y_train = transformed_y_train
        self.time_limit = time_limit
        self.disp = disp

    def solve_grid(self, N_grid):
        """
        Solve for multiple values of N.

        Args:
            N_grid: List of sample sizes to solve for

        Returns:
            Dictionary mapping N to list of selected indices
        """
        results = {}
        for N in tqdm(N_grid):
            results[N] = self.solve(N)
        return results

    def solve(self, N):
        """
        Solve for optimal selection of N samples.

        Args:
            N: Number of OOD samples to select

        Returns:
            List of indices of selected samples
        """
        m = gp.Model("spurious")

        x = m.addVars(self.num_ood_samples, vtype=GRB.BINARY, name="x") # decision variables
        model_accuracies = m.addVars(self.num_models_train, vtype=GRB.CONTINUOUS, name="acc", lb=0, ub=1)
        transformed_model_accuracies = m.addVars(self.num_models_train, vtype=GRB.CONTINUOUS, name="trans_acc",
                                               lb=-GRB.INFINITY, ub=GRB.INFINITY)

        mean_x = m.addVar(vtype=GRB.CONTINUOUS, name="mean_x", lb=-GRB.INFINITY, ub=GRB.INFINITY)
        mean_y = m.addVar(vtype=GRB.CONTINUOUS, name="mean_y", lb=-GRB.INFINITY, ub=GRB.INFINITY)

        x_centered = m.addVars(self.num_models_train, vtype=GRB.CONTINUOUS, name="x_centered",
                             lb=-GRB.INFINITY, ub=GRB.INFINITY)
        y_centered = m.addVars(self.num_models_train, vtype=GRB.CONTINUOUS, name="y_centered",
                             lb=-GRB.INFINITY, ub=GRB.INFINITY)

        covariance = m.addVar(vtype=GRB.CONTINUOUS, name="covariance", lb=-GRB.INFINITY, ub=GRB.INFINITY)
        var_x = m.addVar(vtype=GRB.CONTINUOUS, name="var_x", lb=0, ub=GRB.INFINITY)
        var_y = m.addVar(vtype=GRB.CONTINUOUS, name="var_y", lb=0, ub=GRB.INFINITY)

        obj = m.addVar(vtype=GRB.CONTINUOUS, name='obj', lb=-1, ub=1)

        m.addConstr(x.sum() == N, "size_total")

        # compute correlation
        m.addConstr(mean_x == gp.quicksum(transformed_model_accuracies[i] for i in range(self.num_models_train))
                   / self.num_models_train, "mean_x_defn")
        m.addConstr(mean_y == gp.quicksum(self.transformed_y_train[i] for i in range(self.num_models_train))
                   / self.num_models_train, "mean_y_defn")

        for i in range(self.num_models_train):
            m.addConstr(x_centered[i] == transformed_model_accuracies[i] - mean_x, f"x_centered_{i}")
            m.addConstr(y_centered[i] == self.transformed_y_train[i] - mean_y, f"y_centered_{i}")

        m.addConstr(covariance == gp.quicksum(x_centered[i] * y_centered[i] for i in range(self.num_models_train))
                   / self.num_models_train, "covariance_defn")

        m.addConstr(var_x == gp.quicksum(x_centered[i] * x_centered[i] for i in range(self.num_models_train))
                   / self.num_models_train + 1e-8, "var_x_defn")
        m.addConstr(var_y == gp.quicksum(y_centered[i] * y_centered[i] for i in range(self.num_models_train))
                   / self.num_models_train + 1e-8, "var_y_defn")

        m.addConstr(obj == covariance / nlfunc.sqrt(var_x * var_y), "correlation_defn")

        # Model accuracy constraints
        for i in range(self.num_models_train):
            m.addConstr(model_accuracies[i] == sum([self.X_train_tensor[i, j] * x[j]
                                                  for j in range(self.num_ood_samples)])/self.num_ood_samples,
                       "acc_defn")

        # probit function approximation
        a = -0.416
        b = -0.717

        p_complement = m.addVars(self.num_models_train, vtype=GRB.CONTINUOUS, name="p_comp", lb=0, ub=1)
        log_term = m.addVars(self.num_models_train, vtype=GRB.CONTINUOUS, name="log_term",
                           lb=-GRB.INFINITY, ub=GRB.INFINITY)
        sqrt_term = m.addVars(self.num_models_train, vtype=GRB.CONTINUOUS, name="sqrt_term", lb=0, ub=GRB.INFINITY)
        is_greater = m.addVars(self.num_models_train, vtype=GRB.BINARY, name="is_greater")

        for i in range(self.num_models_train):
            m.addGenConstrIndicator(is_greater[i], True, model_accuracies[i] >= 0.5, name=f"greater_than_{i}")
            m.addGenConstrIndicator(is_greater[i], False, model_accuracies[i] <= 0.5, name=f"not_greater_{i}")
            m.addConstr(p_complement[i] == is_greater[i] * (1 - model_accuracies[i]) +
                       (1 - is_greater[i]) * model_accuracies[i], f"p_comp_{i}")

            m.addConstr(log_term[i] == -nlfunc.log(2 * p_complement[i]), f"neg_log_term_{i}")
            m.addConstr(sqrt_term[i] == nlfunc.sqrt(b**2 - 4*a*log_term[i]), f"sqrt_term_{i}")

            m.addConstr(
                transformed_model_accuracies[i] ==
                is_greater[i] * (-b - sqrt_term[i])/(2*a) +
                (1 - is_greater[i]) * (-(-b - sqrt_term[i])/(2*a)),
                f"z_calc_{i}"
            )

        m.setObjective(obj, GRB.MINIMIZE)
        m.setParam('TimeLimit', self.time_limit)
        m.setParam('OutputFlag', self.disp)
        m.optimize()

        self.model = m
        self.selected_idxs = [i for i in range(self.num_ood_samples) if x[i].X == 1]

        return self.selected_idxs
