'''
Created on Oct 8, 2014

@author: d1farre
'''

FILE_NAME = "number_stream.txt"
from collections import defaultdict
import random
import math

def readTokens(file_name):
    with open(file_name) as infile:
        for line in infile:
            yield line

class HashFunc(object):
    '''universal hash function: ((a*x) % p) % n'''
    def __init__(self, buckets, dim=1):
        self.n = buckets
        self.p = self.prime(buckets)
        self.a = [random.randint(1,self.p-1) for x in range(dim+1)]
    def __call__(self, x):
        if type(x) is list:
            x = x + [1]
        else:
            x = [x, 1]
        return (self.dot(x, self.a) % self.p) % self.n
    def dot(self, x, y):
        return sum([x[i]*y[i] for i in range(len(x))])
    def prime(self, p):
        #p must be larger than the number of buckets
        p*=2
        while not self.is_prime(p):
            p+=1
        return p
    def is_prime(self, num):
        for j in range(2,int(math.sqrt(num)+1)):
            if (num % j) == 0:
                return False
        return True

lookup = [0, 0, 1, 26, 2, 23, 27, 0, 3, 16, 24, 30, 28, 11, 0, 13, 4,
  7, 17, 0, 25, 22, 31, 15, 29, 10, 12, 6, 0, 21, 14, 9, 5, 20, 8, 19, 18]

def countTrailingZeros(x):
    return lookup[(-x & x) % 37]

def countDistinctItems(stream, n):
    k = int(math.ceil(math.log(n, 2)))
    s = 10
    m = k * s
    R = [-1]*m
    H = []
    for i in xrange(0, m):
        f = HashFunc(n*2)
        H.append(f)
    for x in stream:
        for i in xrange(0, m):
            h = H[i].__call__(x)
            R[i] = max(R[i], countTrailingZeros(h))

    averages = []
    for i in xrange(0, 10):
        group = [2**x for x in R[i*k:(i+1)*k]]
        averages.append(sum(group, 0.0) / len(group))
    sorted_averages = sorted(averages, key = int)
    ans = 0
    if s % 2 == 1:
        ans = sorted_averages[(s + 1) / 2 - 1]
    else:
        ans = (sorted_averages[s / 2 - 1] + sorted_averages[(s + 2) / 2 - 1]) / 2.0
    print "No. of distinct elements = %d" % (ans + 0.5)

if __name__ == "__main__":
    # stream = readToken(FILE_NAME)
    countDistinctItems([1,2,3,2,4,1,3,4,1,2,4,3,1,1,2], 4)
