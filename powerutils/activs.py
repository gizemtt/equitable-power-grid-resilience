import io
import os
import re
from collections import defaultdict
from zipfile import ZipFile

import pandas as pd

from .matpower import load_mpc


def _get_tables_from_aux_file(fh):
    """
    Generic function that takes a file handle for an .aux file and returns a
    dictionary of tables extracted from that file.
    """

    skip_tables = ['PWCaseInformation',
                   'Contingency',
                   'ContingencyElement']
    table_counter = defaultdict(int)
    tables = {}
    line = 'line'
    while line:
        #line = fh.readline().decode()
        line = fh.readline()
        while not line.startswith('DATA') and not line == '':
            #line = fh.readline().decode()
            line = fh.readline()
        if line == '':
            break
        card = line.strip()
        while ')' not in card:
            #line = fh.readline().decode().strip()
            line = fh.readline().strip()
            card += line
        PATTERN = '\((.*),\s*\[(.*)\].*\)'
        hits = re.search(PATTERN, card).groups()
        table = hits[0].strip()
        columns = [col.strip() for col in hits[1].split(',')]
        if table not in skip_tables:
            #burn = fh.readline().decode()
            burn = fh.readline()
            data = []
            while True:
                #line = fh.readline().decode()
                line = fh.readline()
                if line.startswith('}'):
                    break
                row = []
                token = ''
                quote = False
                for char in line.strip():
                    if char == ' ':
                        if not quote and not token == '':
                            row.append(token)
                            token = ''
                        elif quote:
                            token += char
                        else:
                            token = ''
                    elif char == '"':
                        token += char
                        quote = not quote
                    else:
                        token += char
                if not quote and not token == '':
                    row.append(token)
                data.append(row)
            df = pd.DataFrame(data=data, columns=columns)
            for column in columns:
                value = df.iloc[0][column]
                if '"' in value:
                    df[column] = df[column].str.strip('"')
                    df[column] = df[column].str.strip()
                elif '.' in value:
                    df[column] = df[column].astype(float)
                else:
                    df[column] = df[column].astype(int)
            table_name = table.lower() + str(table_counter[table.lower()])
            table_counter[table.lower()] += 1
            tables[table_name] = df
    return tables


def extract_aux_ACTIVSg2000(zip_filename, aux_filename, encoding=None, purgena=True, fix=True):
    """
    Extracts the .aux file inside the given ACTIVS archive. The default path
    to the .aux file is used, but this may be reconfigured (e.g. if the archive
    was unpacked, modified, and repacked). Additionally, manual fixes are
    applied to the data by default, but this feature may be disabled.
    """

    with ZipFile(zip_filename).open(aux_filename) as fh:
        tables = _get_tables_from_aux_file(io.TextIOWrapper(fh, encoding=encoding))

    # rename tables
    for name in list(tables.keys()):
        if name == 'branch1':
            tables['transformer'] = tables.pop(name)
        else:
            tables[name[:-1]] = tables.pop(name)

    # purge empty columns
    if purgena:
        for name, table in tables.items():
            drop_cols = []
            for col in table:
                if table[col].apply(lambda v: v == '').all():
                    drop_cols.append(col)
                elif table[col].isna().all():
                    drop_cols.append(col)
            tables[name] = table.drop(drop_cols, axis=1)

    # fix a transformer that was probably meant to be a line
    if fix:
        oldrow = tables['transformer'].loc[655]
        tables['transformer'].drop(655, inplace=True)
        newrow = tables['branch'].loc[(tables['branch']['BusNum'] ==\
                                       oldrow['BusNum']) &\
                                      (tables['branch']['BusNum:1'] ==\
                                       oldrow['BusNum:1'])].iloc[0]
        newrow['LineCircuit'] = 1
        tables['branch'] = tables['branch'].append(newrow)

    return tables


def extract_m_ACTIVSg2000(zip_filename, m_filename=None, fix=True):
    """
    Extracts the .m file inside the given ACTIVS archive. The default path
    to the .m file is used, but this may be reconfigured (e.g. if the archive
    was unpacked, modified, and repacked). Additionally, manual fixes are
    applied to the data by the default, but this feature may be disabled.
    """

    if m_filename is None:
        m_filename = os.path.join('ACTIVSg2000',
                                  'ACTIVSg2000',
                                  'case_ACTIVSg2000.m')

    with ZipFile(zip_filename) as zfh:
        with zfh.open(m_filename) as fh:
            tables, values = load_mpc(fh)

    # fix a transformer that was probably meant to be a line
    if fix:
        tables['branch'].loc[2619, 'ratio'] = 0.00000

    return tables, values

