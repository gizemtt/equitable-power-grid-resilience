import re

import pandas as pd


SUBSTATION_CARD = {
    0: 'id',
    1: 'label',
    2: 'latitude',
    3: 'longitude',
}


BUS_CARD = {
    0: 'id',
    1: 'label',
    2: 'kv',
    # 3,4
    # :
    5: 'ty',
    6: 'vsched',
    7: 'volt',
    8: 'angle',
    9: 'ar',
    10: 'zone',
    11: 'vmax',
    12: 'vmin',
    13: 'date_in',
    14: 'date_out',
    15: 'pid',
    16: 'L',
    17: 'own',
    18: 'st',
    19: 'latitude',
    20: 'longitude',
    21: 'island',
    22: 'sdmon',
    23: 'vmax1',
    24: 'vmin1',
    25: 'dvmax',
    26: 'subst',
}


BRANCH_CARD = {
    0: 'id_f',
    1: 'label_f',
    2: 'kv_f',
    3: 'id_t',
    4: 'label_t',
    5: 'kv_t',
    # 6, 7, 8
    # :
    9: 'st',
    10: 'resist',
    11: 'react',
    12: 'charge',
    13: 'rate1',
    14: 'rate2',
    15: 'rate3',
    16: 'rate4',
    17: 'aloss',
    19: 'lngth',
}


TRANSFORMER_CARD = {
    0: 'id_f',
    1: 'label_f',
    2: 'kv_f',
    3: 'id_t',
    4: 'label_t',
    5: 'kv_t',
    # 6, 7
    # :
    # 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
    20: 'ar',
    21: 'zone',
    22: 'tbase',
    23: 'ps_r',
    24: 'ps_x',
    25: 'pt_r',
    26: 'pt_x',
    27: 'ts_r',
    28: 'ts_x',
}


GENERATOR_CARD = {
    0: 'id',
    1: 'label',
    2: 'kv',
    # 3, 4
    # :
    # 5, 6, 7, 8
    9: 'prf',
    10: 'qrf',
    11: 'ar',
    12: 'zone',
    13: 'pgen',
    14: 'pmax',
    15: 'pmin',
    16: 'qgen',
    17: 'qmax',
    18: 'qmin',
    19: 'mbase',
    20: 'cmp_r',
    21: 'cmp_x',
    22: 'gen_r',
    23: 'gen_x',
    # 24, 25, 26, 27, 28, 29
    30: 'date_in',
    31: 'date_out',
    32: 'pid',
    33: 'N',
}


LOAD_CARD = {
    0: 'id',
    1: 'label',
    2: 'kv',
    # 3, 4
    # :
    5: 'st',
    6: 'mw',
    7: 'mvar',
    8: 'mw_i',
    9: 'mvar_i',
    10: 'mw_z',
    11: 'mvar_z',
    12: 'ar',
    13: 'zone',
    14: 'date_in',
    15: 'date_out',
    16: 'pid',
    17: 'N',
    18: 'own',
    19: 'M',
    20: 'nonc',
    21: 'thr_bus',
    22: 'flg',
}


TABLE_CARDS = {
    'substation': SUBSTATION_CARD,
    'bus': BUS_CARD,
    'branch': BRANCH_CARD,
    'transformer': TRANSFORMER_CARD,
    'generator': GENERATOR_CARD,
    'load': LOAD_CARD,
}


KEYVAL_CARDS = {
    'title': (),
    'comments': (),
    'solution parameters': (),
}


def _tokenize(line):
    modline = ''
    quote = False
    for c in line:
        if c == r'"':
            quote = not quote
        elif c == ' ' and quote:
            modline += '_'
        elif c in (':', '/'):
            continue
        else:
            modline += c
    return [x.replace('_', ' ').strip() for x in modline.split(' ') if x != '']


def _df_auto_dtype(df):
    for col in df.columns:
        regex_int = r'^[-+]?\d+$'
        regex_float = r'^[-+]?\d+\.\d*(?:[eE][-+]?\d*)?$'
        if df[col].apply(lambda x: re.findall(regex_int, x)).all():
            df[col] = df[col].astype(int)
        elif df[col].apply(lambda x: re.findall(regex_float, x)).all():
            df[col] = df[col].astype(float)
        else:
            df[col] = df[col].astype(str)


def _load_epc_helper(filename):
    with open(filename) as fh:
        card = None
        linedata = ''
        for line in fh:
            cline = line.strip()
            pattern = '|'.join('^{} data'.format(card) for card in TABLE_CARDS)
            hits = re.findall(pattern, cline)
            if hits:
                if card is not None:
                    df = pd.DataFrame(data=data)
                    _df_auto_dtype(df)
                    yield card.replace(' data', ''), df
                card = hits[0].replace(' data', '')
                columns = TABLE_CARDS[card].keys()
                objs = TABLE_CARDS[card].values()
                data = []
            elif 'data' in cline:
                if card is not None:
                    df = pd.DataFrame(data=data)
                    _df_auto_dtype(df)
                    yield card.replace(' data', ''), df
                card = None
            elif cline.startswith('!'):
                card = None
            elif cline.startswith('#'):
                continue
            elif cline.startswith('end'):
                if card is not None:
                    df = pd.DataFrame(data=data)
                    _df_auto_dtype(df)
                    yield card.replace(' data', ''), df
                    break
            elif cline.endswith('/'):
                linedata += cline.replace('/', '')
            elif card is not None:
                linedata += cline.strip()
                data.append(_tokenize(linedata))
                linedata = ''


def load_epc(filename):
    tables = {}
    for card, df in _load_epc_helper(filename):
        col_map = TABLE_CARDS[card]
        tables[card] = df[col_map.keys()].rename(col_map, axis=1)
    return tables
