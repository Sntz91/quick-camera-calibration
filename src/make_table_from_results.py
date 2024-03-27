import os
import numpy as np
import json

errs = []
vars_ = []
stds = []
sample_sizes = []
m = 100 # cm
dirnames = []
distances = []
for subdir in os.listdir('out'):
    if subdir == '.DS_Store':
        continue
    fname_result = f'out/{subdir}/summary.txt'
    dirnames.append(subdir)
    with open(fname_result, 'r') as file:
        for line in file: #only one line
            key, value = line.split(';')
            if key == 'mean_distance':
                errs.append(float(value)*m)
            elif key == 'variance': 
                vars_.append(float(value)*m)
            elif key == 'std':
                stds.append(float(value)*m)
            elif key == 'sample_size': 
                sample_sizes.append(int(value))
            elif key == 'distances':
                distances.append(value)
for dirname, err, var, std, sample_size, dists in sorted(zip(dirnames, errs, vars_, stds, sample_sizes, distances)):
    dists = dists.replace("'", '"')
    ddict = json.loads(dists)
    for key, value in ddict.items():
        ddict[key] = round(value*100, 2)
    print(f'| {dirname} | {err:.2f} | {var:.2f} | {std:.2f} | {sample_size} | {ddict} |')
