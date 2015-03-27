'''
Created on Oct 8, 2014

@author: d1farre
'''
FILE_NAME = "number_stream.txt"
from collections import defaultdict
from random import randrange

def readTokens(file_name):
    with open(file_name) as infile:
        for line in infile:
            yield int(line)

def update_moment(samples, index, x, s, n):
    if x in index.keys():
        for pos in index[x]:
            element, count = samples[pos]
            samples[pos] = (element, count + 1)
    if n <= s:
        samples[n-1] = (x, 1)
        if x in index.keys():
            index[x].append(n-1)
        else:
            index[x] = [n-1]
    else:
        j = randrange(0,n)
        if j < s:
            element, count = samples[j]
            index[element].remove(j)
            samples[j] = (x, 1)
            if x in index.keys():
                index[x].append(j)
            else:
                index[x] = [j]

def computeKthMoment(stream, end_timestamp, k):
    n = 0
    index = defaultdict(list)
    s = end_timestamp / 5
    samples = [(0, 0)] * s
    for x in stream:
        n += 1
        update_moment(samples, index, x, s, n)
        if n == end_timestamp:
            break

    estimates = []
    for (element, count) in samples:
        estimates.append(end_timestamp * (count**k - (count - 1)**k))

    print "Average = %f" % (sum(estimates, 0.0) / len(estimates))

    estimates = sorted(estimates, key = int)
    # print estimates

    if s % 2 == 1:
        print "Median value = %f" % (estimates[(s + 1) / 2 - 1])
    else:
        print "Median value = %f" % ((estimates[s / 2 - 1] + estimates[(s + 2) / 2 - 1]) / 2.0)

if __name__ == "__main__":
    stream = readTokens(FILE_NAME)
    n = 15
    k = 2
    computeKthMoment(stream, n, k)
    # computeKthMoment([1,2,3,2,4,1,3,4,1,2,4,3,1,1,2], 15, 2)




