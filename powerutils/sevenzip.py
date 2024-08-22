import subprocess

SEVENZIP = '7za'

def list(archive):

    cmd = [SEVENZIP, 'l', archive]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()

    if err:
        raise RuntimeError('Error: Unable to list files in 7z archive.')

    filenames = []
    active = False
    for line in out.decode().split('\n'):
        if line.startswith('-' * 19):
            if active:
                break
            else:
                active = True
        elif active:
            tokens = line.split(' ')
            # third token is string of attributes
            # first attribute is directory and we want to skip those
            if tokens[2][0] != 'D':
                filenames.append(line.split(' ')[-1])
        else:
            continue

    return filenames


def extract(archive, *flags):
    
    cmd = [SEVENZIP, 'x', archive, *flags]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()

    if err:
        raise RuntimeError('Error: Unable to extract files from 7z archive.')
