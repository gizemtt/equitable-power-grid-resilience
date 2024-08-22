from pyomo.environ import *
import os
from .base import *
from .powerflow import *

import pandas as pd 
import numpy as np


################################################################### 
#JUSTICE AS A FIRST STAGE
disadv_justice40 = pd.read_csv("SubSVIandDisadCommunities.csv", index_col=0)
percent_disadvantaged = dict()
for i in disadv_justice40.index:
    percent_disadvantaged[i] = float(disadv_justice40.loc[i]['Pvulnerability'])
###################################################################        

df_load = pd.read_csv("loadnode.csv")
svi_region = pd.read_csv("assignment.csv")
set_census = set(svi_region['Census'])
set_d = set(svi_region['d'])

number_tracts = len(set_census)

pop = dict()
power =dict()
svi= dict()
for i in svi_region.index:
    r = int(svi_region.loc[i]['Census'])
    svi[r]= float(svi_region.loc[i]['SVI'])  # SVI ,  Housing Type & Transportation  Household Composition & Disability   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    pop[r]= float(svi_region.loc[i]['pop'])
    d = int(svi_region[svi_region['Census']==r]['d'])
    power[r]= float(df_load[df_load['d']==d]['power']) * float(svi_region[svi_region['Census']==r]['pop']) / svi_region[svi_region['d']==d]['pop'].sum()

vulnerable_tracts = set([key for key, value in svi.items() if 0.5 < value])
number_tracts= len(vulnerable_tracts)
  
###################################################################     
#Logistic Function to map SVI's to Probability of well being loss
###################################################################  
mean_x = sum(svi.values())/len(svi.values())
min_y = 0.1 #0.1
max_y = 1
k= 14
a= 1.4   # 5 # 1.4

f = lambda xx: min_y + (max_y-min_y) * (1 / (1 + np.exp(-k*(xx-mean_x))))**a

######################################################################################     
#Exponential Function to map SVI's to well being loss duration given there is a loss
######################################################################################

cc= 0
bb= 5
aa= 0.068
g = lambda yy: (aa * np.exp(bb*yy)) + cc


##################################################################################################################### 

class ShortTermBridge(AbstractChunk):

    provides = {'Omega', 'k_of_n', 'probability', 'xi', 'alpha', 'beta'}
    requires = {'N', 'EA', 'K', 'R', 'x'}

    @staticmethod
    def setup_sets(model, **kwargs):
        model.Omega = Set(initialize=sorted(kwargs['Omega']))
        # special sets
        model.k_of_n = Param(model.N, initialize=kwargs['k_of_n'])
        model.NxR = Set(initialize=[(n, r) for n in model.N for r in model.R[model.k_of_n[n]]])
        
        model.census = Set(initialize= set_census)
        model.disadv_census = Set(initialize= set(disadv_census.index))
        model.vulnerable_tracts = Set(initialize= vulnerable_tracts)

    @staticmethod
    def setup_parameters(model, **kwargs):
        model.probability = Param(model.Omega, initialize=kwargs['probability'])
        model.xi = Param(model.KxR, model.Omega, initialize=kwargs['xi'], default=0)
        
        model.weight = Param(initialize=1000000000000000, mutable=True)
        model.JusticeRatio = Param(initialize=0, mutable=True)

    @staticmethod
    def setup_variables(model):
        model.alpha = Var(model.N, model.Omega, domain=Binary)
        model.beta = Var(model.EA, model.Omega, domain=Binary)
        model.gamma = Var(model.Omega, domain=NonNegativeReals)

    @staticmethod
    def setup_constraints(model):
        model.con_def_alpha_gt =\
            Constraint(model.N, model.Omega, rule=ShortTermBridge.con_def_alpha_gt)
        model.con_def_alpha_lt =\
            Constraint(model.NxR, model.Omega, rule=ShortTermBridge.con_def_alpha_lt)
        model.con_def_beta_gt =\
            Constraint(model.EA, model.Omega, rule=ShortTermBridge.con_def_beta_gt)
        model.con_def_beta_lt_f =\
            Constraint(model.EA, model.Omega, rule=ShortTermBridge.con_def_beta_lt_f)
        model.con_def_beta_lt_t =\
            Constraint(model.EA, model.Omega, rule=ShortTermBridge.con_def_beta_lt_t)
        model.con_def_gamma =\
            Constraint(model.Omega, rule=ShortTermBridge.con_def_gamma)
    

    @staticmethod
    def con_def_alpha_gt(model, n, omega):
        k = model.k_of_n[n]
        sum1 = sum(1 - model.xi[k, r, omega] * (1 - model.x[k, r]) for r in model.R[k])
        return model.alpha[n, omega] >= sum1 - len(model.R[k]) + 1

    @staticmethod
    def con_def_alpha_lt(model, n, r, omega):
        k = model.k_of_n[n]
        return model.alpha[n, omega] <= 1 - model.xi[k, r, omega] * (1 - model.x[k, r])

    @staticmethod
    def con_def_beta_gt(model, n, m, omega):
        return model.beta[n, m, omega] >= model.alpha[n, omega] + model.alpha[m, omega] - 1

    @staticmethod
    def con_def_beta_lt_f(model, n, m, omega):
        return model.beta[n, m, omega] <= model.alpha[n, omega]

    @staticmethod
    def con_def_beta_lt_t(model, n, m, omega):
        return model.beta[n, m, omega] <= model.alpha[m, omega]

    @staticmethod
    def con_def_gamma(model, omega):
        return model.gamma[omega] == sum(model.p_load_hi[d] * (1 - model.z[d, omega]) for d in model.D)


class ShortTermMinShedConstraints(AbstractChunk):

    provides = {'K', 'R', 'c', 'f', 'r_max', 'x'}
    requires = set()

    @staticmethod
    def setup_sets(model, **kwargs):
        model.K = Set(initialize=sorted(kwargs['K']))
        model.R = Set(model.K, initialize=kwargs['R'])
        model.KxR = Set(initialize=[(k, r) for k in model.R for r in model.R[k]])
     

    @staticmethod
    def setup_parameters(model, **kwargs):
        model.c = Param(model.KxR, initialize=kwargs['c'], default=0)
        model.f = Param(initialize=kwargs['f'], mutable=True)
        model.r_hat = Param(model.K, initialize=kwargs['r_hat'])

    @staticmethod
    def setup_variables(model):
        model.x = Var(model.KxR, domain=Binary)

    @staticmethod
    def setup_constraints(model):
        model.con_resource_hi = Constraint(rule=ShortTermMinShedConstraints.con_resource_hi)
        model.con_inevitable_damage = Constraint(model.K, rule=ShortTermMinShedConstraints.con_inevitable_damage)
        model.con_incremental = Constraint(model.KxR, rule=ShortTermMinShedConstraints.con_incremental)
        
        model.con_Justice = Constraint(rule=ShortTermMinShedConstraints.con_Justice)

    @staticmethod
    def con_Justice(model):
        return sum(model.c[k, r] * model.x[k, r] for k, r in model.KxR) * model.JusticeRatio <= sum(percent_disadvantaged[k] * model.c[k, r] * model.x[k, r] for k, r in model.KxR)
      

    @staticmethod
    def con_resource_hi(model):
        return sum(model.c[k, r] * model.x[k, r] for k, r in model.KxR) <= model.f

    @staticmethod
    def con_inevitable_damage(model, k):
        return model.x[k, model.r_hat[k]] == 0

    @staticmethod
    def con_incremental(model, k, r):
        if r == model.r_hat[k]:
            return Constraint.Skip
        else:
            return model.x[k, r+1] <= model.x[k, r]


class StochasticShortTermMinShedObjective(AbstractChunk):

    provides = set()
    requires = set()

    def setup_objectives(model):
        model.obj_min_expected_shed =\
            Objective(sense=minimize, rule=StochasticShortTermMinShedObjective.obj_min_expected_shed)

    @staticmethod
    def obj_min_expected_shed(model):
        return  sum(model.probability[omega] * model.gamma[omega] for omega in model.Omega)
        
        # Expected number of people
        #return  sum(model.probability[omega] * f(svi[r]) * (1 - model.z[int(svi_region.loc[svi_region.Census==r]['d']), omega])  * pop[r] for omega in model.Omega for r in model.census) 
        
        # EIP Objective Function- Composite - Expected number of people
        #return  model.weight * sum(model.probability[omega] * model.gamma[omega] for omega in model.Omega) + sum(model.probability[omega] * f(svi[r]) * (1 - model.z[int(svi_region.loc[svi_region.Census==r]['d']), omega])  * pop[r] for omega in model.Omega for r in model.census) 
        
        # Composite- Expected number of people in the vulnerable regions (SVI>0.5)
        #return  model.weight * sum(model.probability[omega] * model.gamma[omega] for omega in model.Omega) + sum(model.probability[omega] * f(svi[r]) * (1 - model.z[int(svi_region.loc[svi_region.Census==r]['d']), omega])  * pop[r] for omega in model.Omega for r in model.vulnerable_tracts) 

        # EID Objective Function- Composite - Expected Loss Duration
        #return  model.weight * sum(model.probability[omega] * model.gamma[omega] for omega in model.Omega) + 12*30* sum(model.probability[omega] * g(svi[r]) * f(svi[r]) * (1 - model.z[int(svi_region.loc[svi_region.Census==r]['d']), omega]) * pop[r]  for omega in model.Omega for r in model.census) / sum(pop.values())     
        


class RobustShortTermMinShedObjective(AbstractChunk):

    provides = set()
    requires = set()

    def setup_variables(model):
        model.gamma_max = Var(domain=Reals)

    def setup_constraints(model):
        model.con_def_gamma_max = Constraint(model.Omega, rule=RobustShortTermMinShedObjective.con_def_gamma_max)

    def setup_objectives(model):
        model.obj_min_max_shed =\
            Objective(sense=minimize, rule=RobustShortTermMinShedObjective.obj_min_max_shed)

    @staticmethod
    def con_def_gamma_max(model, omega):
        return model.gamma_max >= model.gamma[omega]

    @staticmethod
    def obj_min_max_shed(model):
        return model.gamma_max


class CVaRShortTermMinShedObjective(AbstractChunk):

    provides = set()
    requires = set()

    def setup_parameters(model, **kwargs):
        # smaller epsilon -> more like EV
        model.epsilon = Param(initialize=kwargs['epsilon'])

    def setup_variables(model):
        model.gamma_plus = Var(model.Omega, domain=NonNegativeReals)
        model.cvar_scale = Var(domain=Reals)

    def setup_constraints(model):
        model.con_gamma_plus = Constraint(model.Omega, rule=CVaRShortTermMinShedObjective.con_gamma_plus)

    def setup_objectives(model):
        model.obj_min_cvar_shed =\
            Objective(sense=minimize, rule=CVaRShortTermMinShedObjective.obj_min_cvar_shed)

    @staticmethod
    def con_gamma_plus(model, omega):
        return model.gamma_plus[omega] >= model.gamma[omega] - model.cvar_scale

    @staticmethod
    def obj_min_cvar_shed(model):
        sum1 = sum(model.probability[omega] * model.gamma_plus[omega] for omega in model.Omega)
        return model.cvar_scale + 1 / (1 - model.epsilon) * sum1


class ShortTermMinBudget(AbstractChunk):

    provides = {'K', 'R', 'c', 'f', 'r_max', 'x'}
    requires = set()

    @staticmethod
    def setup_sets(model, **kwargs):
        model.K = Set(initialize=sorted(kwargs['K']))
        model.R = Set(initialize=sorted(kwargs['R']))

    @staticmethod
    def setup_parameters(model, **kwargs):
        model.c = Param(model.K, model.R, initialize=kwargs['c'])
        model.r_max = Param(initialize=kwargs['r_max'])

    @staticmethod
    def setup_variables(model):
        model.f = Var(domain=NonNegativeReals)
        model.x = Var(model.K, model.R, domain=Binary)

    @staticmethod
    def setup_constraints(model):
        model.con_resource_hi = Constraint(rule=ShortTermMinShed.con_resource_hi)
        model.con_inevitable_damage = Constraint(model.K, rule=ShortTermMinShed.con_inevitable_damage)
        model.con_incremental = Constraint(model.K, model.R, rule=ShortTermMinShed.con_incremental)

    @staticmethod
    def con_resource_hi(model):
        return sum(model.c[k, r] * model.x[k, r] for k in model.K for r in model.R) <= model.f

    @staticmethod
    def con_inevitable_damage(model, k):
        return model.x[k, model.r_max] == 0

    @staticmethod
    def con_incremental(model, k, r):
        if r == model.r_max:
            return Constraint.Skip
        else:
            return model.x[k, r+1] <= model.x[k, r]


class StochasticShortTermMinShedLPNF(AbstractModel):

    chunks = [ShortTermMinShedConstraints,
              LPNF,
              ShortTermBridge,
              StochasticShortTermMinShedObjective]


class RobustShortTermMinShedLPNF(AbstractModel):

    chunks = [ShortTermMinShedConstraints,
              LPNF,
              ShortTermBridge,
              RobustShortTermMinShedObjective]


class CVaRShortTermMinShedLPNF(AbstractModel):

    chunks = [ShortTermMinShedConstraints,
              LPNF,
              ShortTermBridge,
              CVaRShortTermMinShedObjective]


class ShortTermMinBudgetLPNF(AbstractModel):

    chunks = [ShortTermMinBudget, LPNF, ShortTermBridge]

    def setup(self, model, **kwargs):
        AbstractModel.setup(self, model, **kwargs)
        model.obj_min_budget = Objective(sense=minimize, expr=model.f)


class StochasticShortTermMinShedLPDC(AbstractModel):

    chunks = [ShortTermMinShedConstraints,
              LPDC,
              ShortTermBridge,
              StochasticShortTermMinShedObjective]


class RobustShortTermMinShedLPDC(AbstractModel):

    chunks = [ShortTermMinShedConstraints,
              LPDC,
              ShortTermBridge,
              RobustShortTermMinShedObjective]


class CVaRShortTermMinShedLPDC(AbstractModel):

    chunks = [ShortTermMinShedConstraints,
              LPDC,
              ShortTermBridge,
              CVaRShortTermMinShedObjective]


class ShortTermMinBudgetLPDC(AbstractModel):

    chunks = [ShortTermMinBudget, LPDC, ShortTermBridge]

    def setup(self, model, **kwargs):
        AbstractModel.setup(self, model, **kwargs)
        model.obj_min_budget = Objective(sense=minimize, expr=model.f)


class StochasticShortTermMinShedLPAC(AbstractModel):

    chunks = [ShortTermMinShedConstraints,
              LPAC,
              ShortTermBridge,
              StochasticShortTermMinShedObjective]

    def setup(self, model, **kwargs):
        AbstractModel.setup(self, model, **kwargs)
        model.obj_max_cosine_sum =\
            Objective(sense=maximize, rule=self.obj_max_cosine_sum)
        model.obj_max_cosine_sum.deactivate()

    @staticmethod
    def obj_max_cosine_sum(model):
        return sum(model.probability[omega] * model.u[n, m, omega]
                   for (n, m) in model.EA for omega in model.Omega)


class RobustShortTermMinShedLPAC(AbstractModel):

    chunks = [ShortTermMinShedConstraints,
              LPAC,
              ShortTermBridge,
              RobustShortTermMinShedObjective]

    def setup(self, model, **kwargs):
        AbstractModel.setup(self, model, **kwargs)
        model.obj_max_cosine_sum =\
            Objective(sense=maximize, rule=self.obj_max_cosine_sum)
        model.obj_max_cosine_sum.deactivate()

    @staticmethod
    def obj_max_cosine_sum(model):
        return sum(model.probability[omega] * model.u[n, m, omega]
                   for (n, m) in model.EA for omega in model.Omega)


class CVaRShortTermMinShedLPAC(AbstractModel):

    chunks = [ShortTermMinShedConstraints,
              LPAC,
              ShortTermBridge,
              CVaRShortTermMinShedObjective]

    def setup(self, model, **kwargs):
        AbstractModel.setup(self, model, **kwargs)
        model.obj_max_cosine_sum =\
            Objective(sense=maximize, rule=self.obj_max_cosine_sum)
        model.obj_max_cosine_sum.deactivate()

    @staticmethod
    def obj_max_cosine_sum(model):
        return sum(model.probability[omega] * model.u[n, m, omega]
                   for (n, m) in model.EA for omega in model.Omega)


class ShortTermMinBudgetLPAC(AbstractModel):

    chunks = [ShortTermMinBudget, LPAC, ShortTermBridge]

    def setup(self, model, **kwargs):
        AbstractModel.setup(self, model, **kwargs)
        model.obj_min_budget = Objective(sense=minimize, expr=model.f)
        model.obj_max_cosine_sum =\
            Objective(sense=maximize, rule=self.obj_max_cosine_sum)
        model.obj_max_cosine_sum.deactivate()

    @staticmethod
    def obj_max_cosine_sum(model):
        return sum(model.probability[omega] * model.u[n, m, omega]
                   for (n, m) in model.EA for omega in model.Omega)
