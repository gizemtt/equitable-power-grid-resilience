import os
import re

import pandas as pd

# These parameters all come from the MATPOWER 7.0 Manual, Appendix B.

COLUMNS_BUS = {
    'BUS_I': 'bus number (positive integer)',
    'BUS_TYPE': 'bus type (1 = PQ, 2 = PV, 3 = ref, 4 = isolated',
    'PD': 'real power demand (MW)',
    'QD': 'reactive power demand (MVAr)',
    'GS': 'shunt conductance (MW demanded at V = 1.0 p.u.',
    'BS': 'shunt susceptance (MVAr demanded at V = 1.0 p.u.',
    'BUS_AREA': 'area number (positive integer)',
    'VM': 'voltage magnitude (p.u.)',
    'VA': 'voltage angle (degrees)',
    'BASE_KV': 'base voltage (kV)',
    'ZONE': 'loss zone (positive integer)',
    'VMAX': 'maximum voltage magnitude (p.u.)',
    'VMIN': 'minimum voltage magnitude (p.u.)',
    'LAM_P': 'Lagrange multiplier on real power mismatch (u/MW)',
    'LAM_Q': 'Lagrange multiplier on reactive power mistmatch (u/MVAr)',
    'MU_VMAX': 'Kuhn-Tucker multiplier on upper voltage limit (u/p.u.)',
    'MU_VMIN': 'Kuhn-Tucker multiplier on lower voltage limit (u/p.u.)',
}

DTYPES_BUS = {
    'BUS_I': int,
    'BUS_TYPE': int,
    'PD': float,
    'QD': float,
    'GS': float,
    'BS': float,
    'BUS_AREA': int,
    'VM': float,
    'VA': float,
    'BASE_KV': float,
    'ZONE': int,
    'VMAX': float,
    'VMIN': float,
    'LAM_P': float,
    'LAM_Q': float,
    'MU_VMAX': float,
    'MU_VMIN': float,
}

COLUMNS_GEN = {
    'GEN_BUS': 'bus number',
    'PG': 'real power output (MW)',
    'QG': 'reactive power output (MVAr)',
    'QMAX': 'maximum reactive power output (MVAr)',
    'QMIN': 'minimum reactive power output (MVAr)',
    'VG': 'voltage magnitude setpoint (p.u.)',
    'MBASE': 'total MVA base of machine, defaults to baseMVA',
    'GEN_STATUS': 'machine status, >0 = machine in-service, '
                  '<=0 = machine out-of-service',
    'PMAX': 'maximum real power output (MW)',
    'PMIN': 'minimum real power output (MW)',
    'PC1': 'lower real power output of PQ capability curve (MW)',
    'PC2': 'upper real power output of PQ capability curve (MW)',
    'QC1MIN': 'minimum reactive power output at PC1 (MVAr)',
    'QC1MAX': 'maximum reactive power output at PC1 (MVAr)',
    'QC2MIN': 'minimum reactive power output at PC2 (MVAr)',
    'QC2MAX': 'maximum reactive power output at PC2 (MVAr)',
    'RAMP_AGC': 'ramp rate for load foloowing/AGC (MW/min)',
    'RAMP_10': 'ramp rate for 10 minute reserves (MW)',
    'RAMP_30': 'ramp rate for 30 minute reserves (MW)',
    'RAMP_Q': 'ramp rate for reactive power (2 sec timescale) (MVAr/min)',
    'APF': 'area participation factor',
    'MU_PMAX': 'Kuhn-Tucker multiplier on upper Pg limit (u/MW)',
    'MU_PMIN': 'Kuhn-Tucker multiplier on lower Pg limit (u/MW)',
    'MU_QMAX': 'Kuhn-Tucker multiplier on upper Qg limit (u/MVAr)',
    'MU_QMIN': 'Kuhn-Tucker multiplier on lower Qg limit (u/MVAr)',
}

DTYPES_GEN = {
    'GEN_BUS': int,
    'PG': float,
    'QG': float,
    'QMAX': float,
    'QMIN': float,
    'VG': float,
    'MBASE': float,
    'GEN_STATUS': float,
    'PMAX': float,
    'PMIN': float,
    'PC1': float,
    'PC2': float,
    'QC1MIN': float,
    'QC1MAX': float,
    'QC2MIN': float,
    'QC2MAX': float,
    'RAMP_AGC': float,
    'RAMP_10': float,
    'RAMP_30': float,
    'RAMP_Q': float,
    'APF': float,
    'MU_PMAX': float,
    'MU_PMIN': float,
    'MU_QMAX': float,
    'MU_QMIN': float,
}

COLUMNS_BRANCH = {
    'F_BUS': '"from" bus number',
    'T_BUS': '"to" bus number',
    'BR_R': 'resistance (p.u.)',
    'BR_X': 'reactance (p.u.)',
    'BR_B': 'total line charging susceptance (p.u.)',
    'RATE_A': 'MVA rating A (long term rating), set to 0 for unlimited',
    'RATE_B': 'MVA rating B (medium term rating), set to 0 for unlimited',
    'RATE_C': 'MVA rating C (emergency rating), set to 0 for unlimited',
    'TAP': 'transformer off nominal turns ratio, if non-zero (taps at "from" '
           'bus, impedance at "to" bus, i.e. if r = x = b = 0, '
           'tap = |Vf|/|Vt|; tap = 0 used to indicate transmission line '
           'rather than transformer, i.e. mathematically equivalent to '
           'transformer with tap = 1',
    'SHIFT': 'transformer phase shift angle (degrees), positive => delay',
    'BR_STATUS': 'initial branch status, 1 = in-service, 0 = out-of-service',
    'ANGMIN': 'minimum angle difference, theta_f - theta_t (degrees)',
    'ANGMAX': 'maximum angle difference, theta_f - theta_t (degrees)',
    'PF': 'real power inject at "from" bus end (MW)',
    'QF': 'reactive power injected at "from" bus end (MVAr)',
    'PT': 'real power injected at "to" bus end (MW)',
    'QT': 'reactive power injected at "to" bus end (MVar)',
    'MU_SF': 'Kuhn-Tucker multiplier on MVA limit at "from" bus (u/MVA)',
    'MU_ST': 'Kuhn-Tucker multiplier on MVA limit at "to" bus (u/MVA)',
    'MU_ANGMIN': 'Kuhn-Tucker multiplier lower angle difference limit '
                 '(u/degree)',
    'MU_ANGMAX': 'Kuhn-Tucker multiplier upper angle difference limit '
                 '(u/degree)',
}

DTYPES_BRANCH = {
    'F_BUS': int,
    'T_BUS': int,
    'BR_R': float,
    'BR_X': float,
    'BR_B': float,
    'RATE_A': float,
    'RATE_B': float,
    'RATE_C': float,
    'TAP': float,
    'SHIFT': float,
    'BR_STATUS': int,
    'ANGMIN': float,
    'ANGMAX': float,
    'PF': float,
    'QF': float,
    'PT': float,
    'QT': float,
    'MU_SF': float,
    'MU_ST': float,
    'MU_ANGMIN': float,
    'MU_ANGMAX': float,
}

COLUMNS_GENCOST = {
    'MODEL': 'cost model (1 = piecewise linear, 2 = polynomial)',
    'STARTUP': 'startup cost in US dollars',
    'SHUTDOWN': 'shutdown cost in US dollars',
    'NCOST': 'number N = n + 1 of data points defining an n-segment piecewise '
             'linear cost function, or of coefficients defining an n-th order '
             'polynomial cost function',
}

DTYPES_GENCOST = {
    'MODEL': int,
    'STARTUP': float,
    'SHUTDOWN': float,
    'NCOST': float,
}


def load_mpc(fh):
    """
    Generic function that takes a file handle for an .m file and returns a
    dictionary of tables extracted from that file as well as a dictionary of
    values for various other parameters (e.g. base MVA).
    """
    record = False
    tables = {}
    values = {}
    for line in fh.readlines():
        if isinstance(line, bytes):
            line = line.decode()
        if line.startswith('mpc.'):
            name = line.split(' ')[0].split('.')[1]
            record = True
            data = []
            if line.endswith('[\n'):
                columns = re.split('\s+', last_line.strip('%').strip())
            elif line.endswith('{\n'):
                columns = [name]
            else:
                values[name] = line.strip().strip(';').split('=')[1]
                record = False
        elif line.endswith('];\n') or line.endswith('};\n'):
            record = False
            tables[name] = pd.DataFrame(data=data)
            for col in tables[name].columns:
                if tables[name][col].str.contains('\'').any():
                    tables[name][col] = tables[name][col].astype(str)
                elif tables[name][col].str.contains('\.').any():
                    tables[name][col] = tables[name][col].astype(float)
                else:
                    tables[name][col] = tables[name][col].astype(int)
        elif record:
            row = []
            token = ''
            quote = False
            for char in line.strip().strip(';'):
                if char in (' ', '\t'):
                    if not quote and not token == '':
                        row.append(token)
                        token = ''
                    elif quote:
                        token += char
                    else:
                        token = ''
                elif char == '\'':
                    token += char
                    quote = not quote
                else:
                    token += char
            if not quote and not token == '':
                row.append(token)
            data.append(row)
        last_line = line
    _apply_table_formats(tables)
    _apply_value_formats(values)
    return tables, values


def _apply_table_formats(tables):
    if 'bus' in tables:
        num_cols = len(tables['bus'].columns)
        tables['bus'].columns = list(COLUMNS_BUS.keys())[:num_cols]
        for k, v in DTYPES_BUS.items():
            if k in tables['bus']:
                tables['bus'][k] = tables['bus'][k].astype(v)
    if 'branch' in tables:
        num_cols = len(tables['branch'].columns)
        tables['branch'].columns = list(COLUMNS_BRANCH.keys())[:num_cols]
        for k, v in DTYPES_BRANCH.items():
            if k in tables['branch']:
                tables['branch'][k] = tables['branch'][k].astype(v)
    if 'gen' in tables:
        num_cols = len(tables['gen'].columns)
        tables['gen'].columns = list(COLUMNS_GEN.keys())[:num_cols]
        for k, v in DTYPES_GEN.items():
            if k in tables['gen']:
                tables['gen'][k] = tables['gen'][k].astype(v)
    if 'gencost' in tables:
        tables['gencost'].columns = [
            *COLUMNS_GENCOST.keys(),
            *range(tables['gencost'].shape[1] - len(COLUMNS_GENCOST))
        ]
        for k, v in DTYPES_GENCOST.items():
            tables['gencost'][k] = tables['gencost'][k].astype(v)


def _apply_value_formats(values):
    if 'baseMVA' in values:
        values['baseMVA'] = float(values['baseMVA'].replace('\'', '').strip())
    if 'version' in values:
        values['version'] = int(values['version'].replace('\'', '').strip())
