import math
from collections import defaultdict
import pandas as pd
from pyomo.environ import *

from .base import AbstractChunk, AbstractModel

df_loadnotretained = pd.read_csv("load_notretained.csv")
set_loadnotretained = set(df_loadnotretained['load'])

class LPNF(AbstractChunk):

    provides = {'N', 'N_G', 'G', 'E', 'EA', 'L', 'NxG', 'ExL', 'G_n', 'L_nm', 'delta_neg',
                'delta_pos', 'p_gen_lo', 'p_gen_hi', 'p_load_hi', 's_flow_hi', 'p', 'p_hat',
                'p_tilde', 'z'}
    requires = set()

    @staticmethod
    def setup_sets(model, **kwargs):
        model.N = Set(initialize=sorted(kwargs['N']))
        model.N_G = Set(initialize=sorted(kwargs['N_G']))
        model.N_D = Set(initialize=sorted(kwargs['N_D']))
        model.G = Set(initialize=sorted(kwargs['G']))
        model.D = Set(initialize=sorted(kwargs['D']))
        model.E = Set(initialize=sorted(kwargs['E']))
        EA = {(n, m) for (n, m) in kwargs['E']} | {(m, n) for (n, m) in kwargs['E']}
        model.EA = Set(initialize=sorted(EA))
        model.L = Set(initialize=sorted(kwargs['L']))
        model.loadnotretained = Set(initialize= set_loadnotretained)
        
        # special sets
        model.NxG = Set(within=model.N * model.G, initialize=sorted(kwargs['NxG']))
        model.NxD = Set(within=model.N * model.D, initialize=sorted(kwargs['NxD']))
        model.ExL = Set(within=model.E * model.L, initialize=sorted(kwargs['ExL']))
        model.G_n = Param(model.N, initialize=kwargs['G_n'], default=set(), within=Any)
        model.D_n = Param(model.N, initialize=kwargs['D_n'], default=set(), within=Any)
        model.L_nm = Param(model.E, initialize=kwargs['L_nm'], default=set(), within=Any)
        model.delta_neg = Param(model.N, initialize=kwargs['delta_neg'], default=set(), within=Any)
        model.delta_pos = Param(model.N, initialize=kwargs['delta_pos'], default=set(), within=Any)

    @staticmethod
    def setup_parameters(model, **kwargs):
        model.p_gen_lo = Param(model.G, initialize=kwargs['p_gen_lo'])
        model.p_gen_hi = Param(model.G, initialize=kwargs['p_gen_hi'])
        model.p_load_hi = Param(model.D, initialize=kwargs['p_load_hi'])
        model.s_flow_hi = Param(model.L, initialize=kwargs['s_flow_hi'])

    @staticmethod
    def setup_variables(model):
        model.p = Var(model.N, model.Omega, domain=Reals)
        model.p_hat = Var(model.G, model.Omega, domain=Reals)
        model.p_tilde = Var(model.L, model.Omega, domain=Reals)
        model.z = Var(model.D, model.Omega, bounds=(0, 1))
        model.zeta = Var(model.NxG, model.Omega, domain=Binary)

    @staticmethod
    def setup_constraints(model):
        #notretained load nodes electrify non-coastal area. Then z=1 for them.
        model.con_loadnotretained = Constraint(model.loadnotretained,model.Omega, rule=LPNF.con_loadnotretained)
        
        model.con_gen_dispatch = Constraint(model.NxG, model.Omega, rule=LPNF.con_gen_dispatch)
        model.con_p_net_flow = Constraint(model.N, model.Omega, rule=LPNF.con_p_net_flow)
        model.con_p_net_injection = Constraint(model.N, model.Omega, rule=LPNF.con_p_net_injection)
        model.con_p_gen_lo = Constraint(model.NxG, model.Omega, rule=LPNF.con_p_gen_lo)
        model.con_p_gen_hi = Constraint(model.NxG, model.Omega, rule=LPNF.con_p_gen_hi)
        model.con_p_flow_lo = Constraint(model.ExL, model.Omega, rule=LPNF.con_p_flow_lo)
        model.con_p_flow_hi = Constraint(model.ExL, model.Omega, rule=LPNF.con_p_flow_hi)

    @staticmethod
    def con_loadnotretained(model, non, omega):
        return model.z[non, omega] == 1
    
    
    @staticmethod
    def con_gen_dispatch(model, n, g, omega):
        return model.zeta[n, g, omega] <= model.alpha[n, omega]

    @staticmethod
    def con_p_net_flow(model, n, omega):
        r"""
        .. math::
           :nowrap:

           \begin{equation*}
           p_{n}^{\omega}
           = \sum_{l \in L_n^-} \tilde{p}_{l}^{\omega}
           - \sum_{l \in L_n^+} \tilde{p}_{l}^{\omega},
           \qquad \forall n \in N, \forall \omega \in \Omega
           \end{equation*}
        """
        flow_out = sum(model.p_tilde[l, omega]
                   for m in model.delta_neg[n]
                   for l in model.L_nm[m, n])
        flow_in = sum(model.p_tilde[l, omega]
                   for m in model.delta_pos[n]
                   for l in model.L_nm[n, m])
        return model.p[n, omega] == flow_out - flow_in

    @staticmethod
    def con_p_net_injection(model, n, omega):
        r"""
        .. math::
           :nowrap:

           \begin{equation*}
           p_{n}^{\omega}
           = \sum_{g \in G_n} \hat{p}_{g}^{\omega}
           - \sum_{d \in D_n} \overline{p}_d z_{d}^{\omega},
           \qquad \forall n \in N, \forall \omega \in \Omega
           \end{equation*}
        """
        net_generation = sum(model.p_hat[g, omega] for g in model.G_n[n])
        net_load = sum(model.p_load_hi[d] * model.z[d, omega] for d in model.D_n[n])
        return model.p[n, omega] == net_generation - net_load

    @staticmethod
    def con_p_gen_lo(model, n, g, omega):
        r"""
        .. math::
           :nowrap:

           \begin{equation*}
           \underline{p}_g^\text{gen} \alpha_{n}^{\omega} \le \hat{p}_{g}^{\omega},
           \qquad \forall (n,m) \in E, \forall l \in L_{nm}, \forall \omega \in \Omega
           \end{equation*}
        """
        return model.p_gen_lo[g] * model.zeta[n, g, omega] <= model.p_hat[g, omega]

    @staticmethod
    def con_p_gen_hi(model, n, g, omega):
        r"""
        .. math::
           :nowrap:

           \begin{equation*}
           \hat{p}_{g}^{\omega} \le \overline{p}_g^\text{gen} \alpha_{n}^{\omega},
           \qquad \forall (n,m) \in E, \forall l \in L_{nm}, \forall \omega \in \Omega
           \end{equation*}
        """
        return model.p_hat[g, omega] <= model.p_gen_hi[g] * model.zeta[n, g, omega]

    @staticmethod
    def con_p_flow_lo(model, n, m, l, omega):
        r"""
        .. math::
           :nowrap:

           \begin{equation*}
           -\overline{s}_l^\text{flow} \beta_{n,m}^{\omega} \le \tilde{p}_{l}^{\omega},
           \qquad \forall (n,m) \in E, \forall l \in L_{nm}, \forall \omega \in \Omega
           \end{equation*}
        """
        return -model.s_flow_hi[l] * model.beta[n, m, omega] <= model.p_tilde[l, omega]

    @staticmethod
    def con_p_flow_hi(model, n, m, l, omega):
        r"""
        .. math::
           :nowrap:

           \begin{equation*}
           \tilde{p}_{l}^{\omega} \le \overline{s}_l^\text{flow} \beta_{n,m}^{\omega},
           \qquad \forall (n,m) \in E, \forall l \in L_{nm}, \forall \omega \in \Omega
           \end{equation*}
        """
        return model.p_tilde[l, omega] <= model.s_flow_hi[l] * model.beta[n, m, omega]


class LPDC(AbstractChunk):

    provides = {*LPNF.provides, 'n_ref', 'b', 'theta'}
    requires = set()

    @staticmethod
    def setup_sets(model, **kwargs):
        LPNF.setup_sets(model, **kwargs)
        # special sets
        model.n_ref = Param(initialize=kwargs['n_ref'])

    @staticmethod
    def setup_parameters(model, **kwargs):
        LPNF.setup_parameters(model, **kwargs)
        model.b = Param(model.L, initialize=kwargs['b'])

    @staticmethod
    def setup_variables(model):
        LPNF.setup_variables(model)
        model.theta = Var(model.N, model.Omega, bounds=(-math.pi, math.pi))

    @staticmethod
    def setup_constraints(model):
        LPNF.setup_constraints(model)
        model.con_ohms_law_gt = Constraint(model.ExL, model.Omega, rule=LPDC.con_ohms_law_gt)
        model.con_ohms_law_lt = Constraint(model.ExL, model.Omega, rule=LPDC.con_ohms_law_lt)
        model.con_ref_phase_angle = Constraint(model.Omega, rule=LPDC.con_ref_phase_angle)

    @staticmethod
    def con_ohms_law_gt(model, n, m, l, omega):
        bigM = 2 * math.pi * -model.b[l]
        rhs = (model.theta[n, omega] - model.theta[m, omega]) * -model.b[l]
        lhs = model.p_tilde[l, omega]
        return rhs >= lhs - bigM * (1 - model.beta[n, m, omega])

    @staticmethod
    def con_ohms_law_lt(model, n, m, l, omega):
        bigM = 2 * math.pi * -model.b[l]
        rhs = (model.theta[n, omega] - model.theta[m, omega]) * -model.b[l]
        lhs = model.p_tilde[l, omega]
        return rhs <= lhs + bigM * (1 - model.beta[n, m, omega])

    @staticmethod
    def con_ref_phase_angle(model, omega):
        return model.theta[model.n_ref, omega] == 0


class LPAC(AbstractChunk):

    provides = {'N', 'N_G', 'G', 'EA', 'L', 'O', 'T', 'n_ref', 'NxG', 'EAxLxO', 'delta', 'G_n',
                'LxO_nm', 'p_gen_lo', 'p_gen_hi', 'q_gen_lo', 'q_gen_hi', 'p_load_hi', 'q_load_hi',
                's_flow_hi', 'b', 'g', 'v', 'v_lo', 'v_hi', 'theta_max', 'p', 'q', 'p_hat',
                'q_hat', 'p_tilde', 'q_tilde', 'theta', 'phi', 'u', 'z'}
    requires = set()

    @staticmethod
    def setup_sets(model, **kwargs):
        model.N = Set(initialize=sorted(kwargs['N']))
        model.N_G = Set(initialize=sorted(kwargs['N_G']))
        model.N_D = Set(initialize=sorted(kwargs['N_D']))
        model.G = Set(initialize=sorted(kwargs['G']))
        model.D = Set(initialize=sorted(kwargs['D']))
        EA = {(n, m) for (n, m) in kwargs['E']} | {(m, n) for (n, m) in kwargs['E']}
        model.EA = Set(initialize=sorted(EA))
        model.L = Set(initialize=sorted(kwargs['L']))
        model.O = Set(initialize=['f', 'b'])
        #model.T = Set(initialize=sorted(kwargs['T']))
        model.T = Set(initialize=sorted({1}))
        # special sets
        model.n_ref = Param(initialize=kwargs['n_ref'])
        model.NxG = Set(within=model.N * model.G, initialize=sorted(kwargs['NxG']))
        model.NxD = Set(within=model.N * model.D, initialize=sorted(kwargs['NxD']))
        EAxLxO = {(n, m, l, 'f') for (n, m, l) in kwargs['ExL']}\
            | {(m, n, l, 'b') for (n, m, l) in kwargs['ExL']}
        model.EAxLxO = Set(within=model.EA * model.L * model.O, initialize=sorted(EAxLxO))
        delta = defaultdict(set)
        for n in kwargs['delta_neg']:
            delta[n] |= kwargs['delta_neg'][n]
        for n in kwargs['delta_pos']:
            delta[n] |= kwargs['delta_pos'][n]
        model.delta = Param(model.N, initialize=delta, default=set(), within=Any)
        model.G_n = Param(model.N, initialize=kwargs['G_n'], default=set(), within=Any)
        model.D_n = Param(model.N, initialize=kwargs['D_n'], default=set(), within=Any)
        LxO_nm = defaultdict(set)
        for (n, m) in kwargs['E']:
            for l in kwargs['L_nm'][n, m]:
                LxO_nm[n, m].add((l, 'f'))
                LxO_nm[m, n].add((l, 'b'))
        model.LxO_nm = Param(model.EA, initialize=LxO_nm, default=set(), within=Any)

    @staticmethod
    def setup_parameters(model, **kwargs):
        model.p_gen_lo = Param(model.G, initialize=kwargs['p_gen_lo'])
        model.p_gen_hi = Param(model.G, initialize=kwargs['p_gen_hi'])
        model.q_gen_lo = Param(model.G, initialize=kwargs['q_gen_lo'])
        model.q_gen_hi = Param(model.G, initialize=kwargs['q_gen_hi'])
        model.p_load_hi = Param(model.D, initialize=kwargs['p_load_hi'])
        model.q_load_hi = Param(model.D, initialize=kwargs['q_load_hi'])
        model.s_flow_hi = Param(model.L, initialize=kwargs['s_flow_hi'])
        model.b = Param(model.L, initialize=kwargs['b'])
        model.g = Param(model.L, initialize=kwargs['g'])
        model.v = Param(model.N, initialize=kwargs['v'])
        model.v_lo = Param(model.N, initialize=kwargs['v_lo'])
        model.v_hi = Param(model.N, initialize=kwargs['v_hi'])
        model.theta_max = Param(initialize=kwargs['theta_max'])

    @staticmethod
    def setup_variables(model):
        model.p = Var(model.N, model.Omega, domain=Reals)
        model.q = Var(model.N, model.Omega, domain=Reals)
        model.p_hat = Var(model.G, model.Omega, domain=Reals)
        model.q_hat = Var(model.G, model.Omega, domain=Reals)
        model.p_tilde = Var(model.L, model.O, model.Omega, domain=Reals)
        model.q_tilde = Var(model.L, model.O, model.Omega, domain=Reals)
        model.theta = Var(model.N, model.Omega, bounds=(-math.pi, math.pi))
        model.phi = Var(model.N, model.Omega)
        model.u = Var(model.EA, model.Omega, bounds=(math.cos(model.theta_max.value), 1))
        model.z = Var(model.D, model.Omega, bounds=(0, 1))
        model.zeta = Var(model.NxG, model.Omega, domain=Binary)

    @staticmethod
    def setup_constraints(model):
        # generation dispatch
        model.con_gen_dispatch = Constraint(model.NxG, model.Omega, rule=LPAC.con_gen_dispatch)
        # active power constraints
        model.con_p_net_flow = Constraint(model.N, model.Omega, rule=LPAC.con_p_net_flow)
        model.con_p_net_injection = Constraint(model.N, model.Omega, rule=LPAC.con_p_net_injection)
        model.con_p_gen_lo = Constraint(model.NxG, model.Omega, rule=LPAC.con_p_gen_lo)
        model.con_p_gen_hi = Constraint(model.NxG, model.Omega, rule=LPAC.con_p_gen_hi)
        model.con_p_ohms_law_gt =\
            Constraint(model.EAxLxO, model.Omega,rule=LPAC.con_p_ohms_law_gt)
        model.con_p_ohms_law_lt =\
            Constraint(model.EAxLxO, model.Omega, rule=LPAC.con_p_ohms_law_lt)
        # temporary
        model.con_p_flow_lo = Constraint(model.EAxLxO, model.Omega, rule=LPAC.con_p_flow_lo)
        model.con_p_flow_hi = Constraint(model.EAxLxO, model.Omega, rule=LPAC.con_p_flow_hi)
        # reactive power constraints
        model.con_q_net_flow = Constraint(model.N, model.Omega, rule=LPAC.con_q_net_flow)
        model.con_q_net_injection = Constraint(model.N, model.Omega, rule=LPAC.con_q_net_injection)
        model.con_q_gen_lo = Constraint(model.NxG, model.Omega, rule=LPAC.con_q_gen_lo)
        model.con_q_gen_hi = Constraint(model.NxG, model.Omega, rule=LPAC.con_q_gen_hi)
        model.con_q_ohms_law_gt =\
            Constraint(model.EAxLxO, model.Omega, rule=LPAC.con_q_ohms_law_gt)
        model.con_q_ohms_law_lt =\
            Constraint(model.EAxLxO, model.Omega, rule=LPAC.con_q_ohms_law_lt)
        # temporary
        model.con_q_flow_lo = Constraint(model.EAxLxO, model.Omega, rule=LPAC.con_q_flow_lo)
        model.con_q_flow_hi = Constraint(model.EAxLxO, model.Omega, rule=LPAC.con_q_flow_hi)
        # additional power flow constraints
        model.con_cos_approx = Constraint(model.EA, model.T, model.Omega, rule=LPAC.con_cos_approx)
        model.con_voltage_lo = Constraint(model.N, model.Omega, rule=LPAC.con_voltage_lo)
        model.con_voltage_hi = Constraint(model.N, model.Omega, rule=LPAC.con_voltage_hi)
        model.con_ref_phase_angle = Constraint(model.Omega, rule=LPAC.con_ref_phase_angle)
        model.con_ref_voltage = Constraint(model.Omega, rule=LPAC.con_ref_voltage)

    @staticmethod
    def con_gen_dispatch(model, n, g, omega):
        return model.zeta[n, g, omega] <= model.alpha[n, omega]

    @staticmethod
    def con_p_net_flow(model, n, omega):
        sum1 = sum(model.p_tilde[l, o, omega]
                   for m in model.delta[n]
                   for (l, o) in model.LxO_nm[n, m])
        return model.p[n, omega] == sum1

    @staticmethod
    def con_p_net_injection(model, n, omega):
        net_generation = sum(model.p_hat[g, omega] for g in model.G_n[n])
        net_load = sum(model.p_load_hi[d] * model.z[d, omega] for d in model.D_n[n])
        return model.p[n, omega] == net_generation - net_load

    @staticmethod
    def con_p_gen_lo(model, n, g, omega):
        return model.p_gen_lo[g] * model.zeta[n, g, omega] <= model.p_hat[g, omega]

    @staticmethod
    def con_p_gen_hi(model, n, g, omega):
        return model.p_hat[g, omega] <= model.p_gen_hi[g] * model.zeta[n, g, omega]

    @staticmethod
    def con_p_ohms_law_gt(model, n, m, l, o, omega):
        v_n, v_m = model.v[n], model.v[m]
        b, g = model.b[l], model.g[l]
        theta_n, theta_m = model.theta[n, omega], model.theta[m, omega]
        u = model.u[n, m, omega]
        term1 = v_n ** 2 * g
        term2 = -1 * v_n * v_m * (g * u + b * (theta_n - theta_m))
        bigM = 1e6
        toggle = bigM * (1 - model.beta[n, m, omega])
        return model.p_tilde[l, o, omega] >= term1 + term2 - toggle

    @staticmethod
    def con_p_ohms_law_lt(model, n, m, l, o, omega):
        v_n, v_m = model.v[n], model.v[m]
        b, g = model.b[l], model.g[l]
        theta_n, theta_m = model.theta[n, omega], model.theta[m, omega]
        u = model.u[n, m, omega]
        term1 = v_n ** 2 * g
        term2 = -1 * v_n * v_m * (g * u + b * (theta_n - theta_m))
        bigM = 1e6
        toggle = bigM * (1 - model.beta[n, m, omega])
        return model.p_tilde[l, o, omega] <= term1 + term2 + toggle

    # temporary
    @staticmethod
    def con_p_flow_lo(model, n, m, l, o, omega):
        return -model.s_flow_hi[l] * model.beta[n, m, omega] <= model.p_tilde[l, o, omega]

    # temporary
    @staticmethod
    def con_p_flow_hi(model, n, m, l, o, omega):
        return model.p_tilde[l, o, omega] <= model.s_flow_hi[l] * model.beta[n, m, omega]

    @staticmethod
    def con_q_net_flow(model, n, omega):
        sum1 = sum(model.q_tilde[l, o, omega]
                   for m in model.delta[n]
                   for (l, o) in model.LxO_nm[n, m])
        return model.q[n, omega] == sum1

    @staticmethod
    def con_q_net_injection(model, n, omega):
        net_generation = sum(model.q_hat[g, omega] for g in model.G_n[n])
        net_load = sum(model.q_load_hi[d] * model.z[d, omega] for d in model.D_n[n])
        return model.q[n, omega] == net_generation - net_load

    @staticmethod
    def con_q_gen_lo(model, n, g, omega):
        return model.q_gen_lo[g] * model.zeta[n, g, omega] <= model.q_hat[g, omega]

    @staticmethod
    def con_q_gen_hi(model, n, g, omega):
        return model.q_hat[g, omega] <= model.q_gen_hi[g] * model.zeta[n, g, omega]

    @staticmethod
    def con_q_ohms_law_gt(model, n, m, l, o, omega):
        v_n, v_m = model.v[n], model.v[m]
        b, g = model.b[l], model.g[l]
        theta_n, theta_m = model.theta[n, omega], model.theta[m, omega]
        phi_n, phi_m = model.phi[n, omega], model.phi[m, omega]
        u = model.u[n, m, omega]
        term1 = -1 * v_n ** 2 * b - v_n * v_m * (g * (theta_n - theta_m) - b * u)
        term2 = -1 * v_n * b * (phi_n - phi_m)
        term3 = -1 * (v_n - v_m) * b * phi_n
        bigM = 1e6
        toggle = bigM * (1 - model.beta[n, m, omega])
        return model.q_tilde[l, o, omega] >= term1 + term2 + term3 - toggle

    @staticmethod
    def con_q_ohms_law_lt(model, n, m, l, o, omega):
        v_n, v_m = model.v[n], model.v[m]
        b, g = model.b[l], model.g[l]
        theta_n, theta_m = model.theta[n, omega], model.theta[m, omega]
        phi_n, phi_m = model.phi[n, omega], model.phi[m, omega]
        u = model.u[n, m, omega]
        term1 = -1 * v_n ** 2 * b - v_n * v_m * (g * (theta_n - theta_m) - b * u)
        term2 = -1 * v_n * b * (phi_n - phi_m)
        term3 = -1 * (v_n - v_m) * b * phi_n
        bigM = 1e6
        toggle = bigM * (1 - model.beta[n, m, omega])
        return model.q_tilde[l, o, omega] <= term1 + term2 + term3 + toggle

    # temporary
    @staticmethod
    def con_q_flow_lo(model, n, m, l, o, omega):
        return -model.s_flow_hi[l] * model.beta[n, m, omega] <= model.q_tilde[l, o, omega]

    # temporary
    @staticmethod
    def con_q_flow_hi(model, n, m, l, o, omega):
        return model.q_tilde[l, o, omega] <= model.s_flow_hi[l] * model.beta[n, m, omega]

    @staticmethod
    def con_cos_approx(model, n, m, t, omega):
        theta = model.theta[n, omega] - model.theta[m, omega]
        bigM = 1e6
        toggle = bigM * (1 - model.beta[n, m, omega])
        # outer approximation (based on point-slope form of line)
        if False:
            xm = -model.theta_max + 2 * model.theta_max * t / (len(model.T) - 1)
            ym = math.cos(xm)
            slope = -math.sin(xm)
            return model.u[n, m, omega] <= slope * (theta - xm) + ym
        # inner approximation (based on two-point form of line)
        elif False:
            xl = -model.theta_max + 2 * model.theta_max * t / len(model.T)
            xr = -model.theta_max + 2 * model.theta_max * (t + 1) / len(model.T)
            yl = math.cos(xl)
            yr = math.cos(xr)
            return model.u[n, m, omega] <= (yl - yr) / (xl - xr) * (theta - xr) + yr
        # cosine of a small angle is approximately 1
        else:
            return model.u[n, m, omega] == 1

    @staticmethod
    def con_voltage_lo(model, n, omega):
        return model.v_lo[n] <= model.v[n] + model.phi[n, omega]

    @staticmethod
    def con_voltage_hi(model, n, omega):
        return model.v[n] + model.phi[n, omega] <= model.v_hi[n]

    @staticmethod
    def con_ref_phase_angle(model, omega):
        return model.theta[model.n_ref, omega] == 0

    @staticmethod
    def con_ref_voltage(model, omega):
        return model.phi[model.n_ref, omega] == 0

