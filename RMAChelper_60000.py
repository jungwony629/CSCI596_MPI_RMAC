import sys
import json
import os

os.environ['KERAS_BACKEND'] = 'theano'

from rmac import execute
import time
import os
import numpy

# ABSOULTE_PATH = '/data/VBS/keyframes/'
ABSOULTE_PATH = './keyframes/'
keyframes_dirs = os.listdir(ABSOULTE_PATH)
start = time.time()
results = {}
FROM = 0
if len(sys.argv) > 1:
    FROM = int(sys.argv[2])

print("=====START FROM:", FROM, '=====')

count = 0
for idx, dir in enumerate(keyframes_dirs):
    imgs = os.listdir(ABSOULTE_PATH + dir)
    for img in imgs:
        if img.endswith(".png"):
            count += 1
            if count < FROM:
                continue

            RMAC = execute(ABSOULTE_PATH + dir + "/" + img, count)
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

with open('./results/' + str(FROM) + '_' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '.json', 'w') as outfile:
    json.dump(results, outfile)
