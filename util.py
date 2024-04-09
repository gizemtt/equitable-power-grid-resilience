import math
from collections import defaultdict
import os
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import pyomo.environ as pe
import pyomo.opt as po
from scipy.spatial.distance import cdist
from shapely.geometry import LineString

currentdir=os.getcwd()
DATA_DIR = os.path.join(currentdir, 'data','input')
RESULTS_DIR = os.path.join(currentdir, 'data','output')



PFTYPES = ['lpdc', 'lpac']


def r_hat_builder_disc(grp, thresholds=None):
    xi_max = grp['xi'].max()
    if xi_max <= thresholds[0]:
        return pd.Series(1 * np.ones(grp.shape[0]), index=grp.index).astype(int)
    for r, threshold in enumerate(thresholds, start=1):
        if xi_max < threshold:
            return pd.Series(r * np.ones(grp.shape[0]), index=grp.index).astype(int)
    else:
        return pd.Series(len(thresholds) * np.ones(grp.shape[0]), index=grp.index).astype(int)


def r_builder_cont(grp, xi_max=1.00):
    result, xis = list(), list()
    r = 1
    for idx, row in grp.iterrows():
        if not result or row['xi'] == xis[-1] or (row['xi'] > xi_max and xis[-1] > xi_max):
            pass
        else:
            r += 1
        result.append(r)
        xis.append(row['xi'])
    return pd.Series(result, index=grp.index).apply(lambda r: range(1, r + 1))


def r_hat_builder_cont(grp, xi_max=None):
    if xi_max is None:
        xi_max = 1.00
    return grp['r'].max() + int((grp['xi'] <= xi_max).all())


def tigerdam_disc(data):
    r_max = 3
    diameter = 1.0 / (1.0 + 0.5 * np.sqrt(3))
    thresholds = [0] + [round(diameter * (1 + (r - 1) * 0.5 * np.sqrt(3)), 8)
                        for r in range(1, r_max)]
    # copy the data, compute r and c values
    datacp = data.copy()
    datacp = datacp.sort_values(['k', 'xi'])
    datacp['r_hat'] = datacp.groupby('k', group_keys=False)\
                            .apply(r_hat_builder_disc, thresholds=thresholds)
    datacp['r'] = datacp['r_hat'].apply(lambda r: range(1, r + 1))
    datacp = datacp.explode('r')
    datacp['c'] = datacp['r'] * datacp['size']
    datacp['xi'] = datacp['xi'].gt(datacp['r'].apply(lambda r: thresholds[r - 1]))
    # compute and return the uncertainty specs
    K = datacp['k'].unique().tolist()
    r_hat = datacp[['k', 'r_hat']].drop_duplicates().set_index('k')['r_hat'].to_dict()
    R = {k: list(range(1, r_hat[k] + 1)) for k in K}
    Omega = datacp['omega'].unique().tolist()
    probability = {omega: 1 / len(Omega) for omega in Omega}
    xi = datacp.set_index(['k', 'r', 'omega'])['xi'].astype(int)
    xi = xi.loc[xi == 1].to_dict()
    c = datacp[['k', 'r', 'c']].drop_duplicates().set_index(['k', 'r'])['c'].to_dict()
    return {'r_hat': r_hat, 'R': R, 'Omega': Omega,
            'probability': probability, 'xi': xi, 'c': c}


def sandbag_disc(data):
    r_max = 5
    height = 0.25
    thresholds = [0] + [height * r for r in range(1, r_max)]
    # copy the data, compute r and c values
    datacp = data.copy()
    datacp = datacp.sort_values(['k', 'xi'])
    datacp['r_hat'] = datacp.groupby('k', group_keys=False)\
                            .apply(r_hat_builder_disc, thresholds=thresholds)
    datacp['r'] = datacp['r_hat'].apply(lambda r: range(1, r + 1))
    datacp = datacp.explode('r')
    datacp['c'] = height * datacp['size'] #size perimeter
    datacp['xi'] = datacp['xi'].gt(datacp['r'].apply(lambda r: thresholds[r - 1]))
    # compute and return the uncertainty specs
    K = datacp['k'].unique().tolist()
    r_hat = datacp[['k', 'r_hat']].drop_duplicates().set_index('k')['r_hat'].to_dict()
    R = {k: list(range(1, r_hat[k] + 1)) for k in K}
    Omega = datacp['omega'].unique().tolist()
    probability = {omega: 1 / len(Omega) for omega in Omega}
    xi = datacp.set_index(['k', 'r', 'omega'])['xi'].astype(int)
    xi = xi.loc[xi == 1].to_dict()
    c = datacp[['k', 'r', 'c']].drop_duplicates().set_index(['k', 'r'])['c'].to_dict()
    return {'r_hat': r_hat, 'R': R, 'Omega': Omega,
            'probability': probability, 'xi': xi, 'c': c}


def sandbag_cont(data):
    xi_max = 1.00
    # copy the data, compute r and c values
    datacp = data.copy()
    datacp = datacp.sort_values(['k', 'xi'])
    datacp['r'] = datacp.groupby('k', group_keys=False)\
                        .apply(r_builder_cont, xi_max=xi_max)
    datacp = datacp.explode('r')
    tmp = datacp.groupby(['k', 'omega'])['xi']\
                .min()\
                .to_frame()\
                .reset_index()\
                .sort_values(['k', 'xi'])\
                .set_index(['k', 'omega'])
    tmp['xi'] = tmp.groupby('k')['xi'].diff(1).fillna(tmp['xi'])
    tmp['r'] = datacp.groupby(['k', 'omega'])['r'].max()
    tmp = tmp.reset_index()\
              .drop_duplicates(['k', 'r'])\
              .set_index(['k', 'r'])['xi']\
              .to_dict()
    datacp['c'] = datacp[['k', 'r']].apply(lambda key: tmp.get(tuple(key)), axis=1) * datacp['size']
    # compute and return the uncertainty specs
    K = data['k'].unique().tolist()
    r_hat = datacp.groupby('k').apply(r_hat_builder_cont, xi_max=1.00).to_dict()
    R = {k: list(range(1, r_hat[k] + 1)) for k in K}
    Omega = datacp['omega'].unique().tolist()
    probability = {omega: 1 / len(Omega) for omega in Omega}
    xi = datacp.set_index(['k', 'r', 'omega'])['xi'].gt(0).astype(int).to_dict()
    c = datacp.set_index(['k', 'r'])['c'].to_dict()
    return {'r_hat': r_hat, 'R': R, 'Omega': Omega,
            'probability': probability, 'xi': xi, 'c': c}


def max_budget(c, xi, r_hat, **specs):
    maxlevels = {(k, omega): r for (k, r, omega) in sorted(xi.keys())}
    newlevels = {(k, rp)
                 for (k, _), r in maxlevels.items()
                 for rp in range(1, r + 1)
                 if r < r_hat[k]}
    c_star = sum(c[key] for key in newlevels)
    return c_star


def max_budget_by_omega(c, xi, r_hat, **specs):
    maxlevels = {(k, omega): r for (k, r, omega) in sorted(xi.keys())}
    Omega = {omega for (_, _, omega) in xi.keys()}
    newlevels = {(k, omega, rp)
                 for (k, omega), r in maxlevels.items()
                 for rp in range(1, r + 1)
                 if r < r_hat[k]}
    c_star = {omega: sum(c[k, r] for (k, omegap, r) in newlevels if omegap == omega)
              for omega in Omega}
    return c_star


def max_budget_by_k(c, xi, r_hat, **specs):
    maxlevels = {(k, omega): r for (k, r, omega) in sorted(xi.keys())}
    K = {k for (k, _, _) in xi.keys()}
    newlevels = {(k, rp)
                 for (k, _), r in maxlevels.items()
                 for rp in range(1, r + 1)
                 if r < r_hat[k]}
    c_star = {k: sum(c[k, r] for (kp, r) in newlevels if kp == k)
              for k in K}
    return c_star


def update_budget_constraint(solver, instance, f):
    instance.f = f
    if hasattr(instance, 'con_resource_hi'):
        solver.remove_constraint(instance.con_resource_hi)
        instance.del_component('con_resource_hi')
    lhs = sum(instance.c[k, r] * instance.x[k, r]
              for k in instance.R for r in instance.R[k])
    rhs = instance.f
    instance.con_resource_hi = pe.Constraint(expr=(lhs <= rhs))
    solver.add_constraint(instance.con_resource_hi)


def cut_first_stage_solution(solver, instance, x):
    if hasattr(instance, 'con_cut_optimal'):
        solver.remove_constraint(instance.con_cut_optimal)
        instance.del_component('con_cut_optimal')
    lhs = sum(instance.x[k, r]
              for k, r in x.index
              if x.loc[k, r] == 1)
    rhs = x.sum() - 1
    if rhs >= 0:
        instance.con_cut_optimal = pe.Constraint(expr=(lhs <= rhs))
        solver.add_constraint(instance.con_cut_optimal)


def similarity(xa, xb, c):

    mask_both = (xa == 1) & (xb == 1)
    mask_only_a = (xa == 1) & (xb == 0)
    mask_only_b = (xb == 1) & (xa == 0)

    sr_both = c.loc[mask_both]
    sr_only_a = c.loc[mask_only_a]
    sr_only_b = c.loc[mask_only_b]

    return ((int(sr_both.sum()), int(sr_only_a.sum()), int(sr_only_b.sum())),
            (set(sr_both.index), set(sr_only_a.index), set(sr_only_b.index)))


def remap(head_nodes, tail_nodes):

    distance = pd.DataFrame(cdist(head_nodes[['lon', 'lat']],
                                  tail_nodes[['lon', 'lat']]),
                            index=head_nodes.index,
                            columns=tail_nodes.index)
    threshold = 0.50
    distance = distance.where(distance.values <= threshold).stack()
    distance_A = defaultdict(set)
    distance_B = defaultdict(set)
    for (a, b), d in distance.items():
        distance_A[a].add(b)
        distance_B[b].add(a)

    def objective_rule(m):
        return sum(m.x[(a, b)] * distance.loc[(a, b)] for a, b in m.C)

    def inclusivity_of_A(m, a):
        if distance_A[a]:
            return sum(m.x[(a, b)] for b in distance_A[a]) >= 1
        else:
            return pe.Constraint.Skip

    def exclusivity_of_B(m, b):
        if distance_B[b]:
            return sum(m.x[(a, b)] for a in distance_B[b]) <= 1
        else:
            return pe.Constraint.Skip

    m = pe.ConcreteModel()
    m.A = pe.Set(initialize=distance.index.levels[0])
    m.B = pe.Set(initialize=distance.index.levels[1])
    m.C = pe.Set(within=m.A*m.B, initialize=distance.index)
    m.x = pe.Var(m.C, domain=pe.NonNegativeReals, bounds=(0, 1))
    m.obj = pe.Objective(sense=pe.minimize, rule=objective_rule)
    m.inclusivity_of_A = pe.Constraint(m.A, rule=inclusivity_of_A)
    m.exclusivity_of_B = pe.Constraint(m.B, rule=exclusivity_of_B)
    results = po.SolverFactory('gurobi').solve(m)

    pairs = [(a, b) for a, b in m.C if m.x[(a, b)].value > 0]
    sol = gpd.GeoDataFrame()
    sol['head_node'] = [a for a, _ in pairs]
    sol['tail_node'] = [b for _, b in pairs]
    sol['distance'] = [distance.loc[(a, b)] for a, b in pairs]
    sol['head_geometry'] = head_nodes.loc[sol['head_node'], 'geometry'].values
    sol['tail_geometry'] = tail_nodes.loc[sol['tail_node'], 'geometry'].values
    sol['line_geometry'] = sol[['head_geometry', 'tail_geometry']].apply(LineString, axis=1)

    return sol


def sub_scenario_plot(df):

    data = pd.pivot_table(df, index='k', columns='omega', values='xi')

    # plot setup
    gridspec_kw = {'width_ratios': [4, 1],
                   'height_ratios': [1, 3]}
    fig, axes = plt.subplots(2, 2, gridspec_kw=gridspec_kw,
                             figsize=(12, 6), sharex='col', sharey='row')
    fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 0.5})

    # data setup
    Omega = data.columns.tolist()
    indicator = data.notna()
    sub_cnt = indicator.sum(axis=1)
    scenario_cnt = indicator.sum(axis=0)

    # main
    axes[1, 0].imshow(data.T, vmin=0.00, vmax=1.00, aspect='auto')
    axes[1, 0].set_xlim([-1, sub_cnt.shape[0]])
    axes[1, 0].set_xticks(range(sub_cnt.shape[0]))
    axes[1, 0].set_xticklabels(sub_cnt.index, rotation=90)
    axes[1, 0].set_xlabel('Substations')
    axes[1, 0].set_ylim(-1, len(Omega))
    axes[1, 0].set_yticks(range(len(Omega)))
    axes[1, 0].set_yticklabels(Omega)
    axes[1, 0].set_ylabel('Scenario')

    # top
    axes[0, 0].bar(range(sub_cnt.shape[0]), sub_cnt, color='k')
    axes[0, 0].spines['right'].set_visible(False)
    axes[0, 0].spines['top'].set_visible(False)
    axes[0, 0].grid(which='major', axis='y')
    ymax = 10 * (sub_cnt.max() // 10 + 1)
    yticks = range(0, ymax + 10, 10)
    axes[0, 0].set_ylim(0, ymax)
    axes[0, 0].set_yticks(yticks)
    axes[0, 0].set_ylabel('# Scenarios\nwith Flooding')

    # right
    axes[1, 1].barh(range(scenario_cnt.shape[0]), scenario_cnt, color='k')
    axes[1, 1].spines['right'].set_visible(False)
    axes[1, 1].spines['top'].set_visible(False)
    axes[1, 1].grid(which='major', axis='x')
    xmax = 10 * (scenario_cnt.max() // 10 + 1)
    xticks = range(0, xmax + 10, 10)
    axes[1, 1].set_xlim(0, xmax)
    axes[1, 1].set_xticks(xticks)
    axes[1, 1].set_xlabel('# Substations\nFlooded')

    # unused
    axes[0, 1].axis('off')

    # plot
    plt.show()
