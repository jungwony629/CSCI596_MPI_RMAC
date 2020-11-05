import math
import json
import os

import sys

os.environ['KERAS_BACKEND'] = 'theano'

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from rmac import execute
import time
import os
import numpy

# ABSOULTE_PATH = '/data/VBS/keyframes/'
ABSOULTE_PATH = './keyframes/'

keyframes_dirs = os.listdir(ABSOULTE_PATH)
start = time.time()
results = {}
#1082462
with open("./all_paths_thumbnails.json", 'r') as outfile:
    imgs = json.load(outfile)

TOTAL = len(imgs)
ATOM_LENGTH = math.ceil(TOTAL / size)
FROM = int(ATOM_LENGTH * rank)

print("=====START FROM:", FROM, '=====', "FROM:" + str(rank))

# shot00001_100_RKF.png
for idx, img in enumerate(imgs, start=FROM):
    if rank + 1 != size:
        if idx > int(ATOM_LENGTH * (rank + 1)):
            break

    # shot00001_100_RKF.png
    if ABSOULTE_PATH == '/data/VBS/keyframes/':
        els = img.split('.png')
        img = els[0] + '_RKF.png'

    RMAC = execute(ABSOULTE_PATH + "/" + img, idx, rank)
    data = []
    for i in numpy.nditer(RMAC):
        data.append(float(i))
    results[img] = data
    RMAC = None

    if len(results.keys()) % 100 == 0:
        with open('./results/' + str(FROM) + '_' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '.json', 'w') as outfile:
            json.dump(results, outfile)

        FROM += len(results.keys())
        results = {}
        end = time.time()
        with open('./results/process.txt', 'a') as outfile:
            currentTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            outfile.write(currentTime + '\t' + 'Total:' + str(end - start) + '\t' + 'Per seconds:' + str((end - start) / 100) + '\n')
        print(currentTime, 'Total:' + str(end - start), 'Per seconds:' + str((end - start) / 100))
        start = time.time()

if len(results.keys()) > 0:
    with open('./results/' + str(FROM) + '_' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '.json', 'w') as outfile:
        json.dump(results, outfile)
