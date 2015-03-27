'''
Created on Oct 8, 2014

@author: d1farre
'''
NUMBERS = "numbers.txt"
IN_FILE = "bf_input.txt"

from collections import defaultdict
from random import randrange
from bitarray import bitarray
import math
import random

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

class BloomFilter(object):
    def __init__(self, bit_array, hash_funcs):
        self.bit_array = bit_array
        self.hash_funcs = hash_funcs

    def apply(self, stream):
        positives = []
        negatives = []
        for x in stream:
            present = True
            for hash_func in self.hash_funcs:
                if present and self.bit_array[hash_func.__call__(x)] == 1:
                    continue
                else:
                    negatives.append(x)
                    present = False
                    break
            if present == True:
                positives.append(x)
        return positives, negatives


def readTokens(file_name):
    tokens = []
    with open(file_name) as infile:
        for line in infile:
            tokens.append(int(line))
    return tokens

def constructBloomFilter(stream, size, m):
    bit_array = bitarray(size)
    bit_array.setall(False)
    H = []
    for i in xrange(0, m):
        H.append(HashFunc(size))
    for x in stream:
        for hash_func in H:
            bit_array[hash_func.__call__(x)] = True
    return BloomFilter(bit_array, H)

def printAccuracy(numbers, stream, positives, negatives):
    numbers_dict = {}
    for x in numbers:
        numbers_dict[x] = True
    intersection_dict = {}
    for x in stream:
        if x in numbers_dict.keys():
            intersection_dict[x] = True

    false_positives = [i for i in positives if i not in intersection_dict.keys()]
    false_negatives = [i for i in negatives if i in intersection_dict.keys()]
    print "False positive rate = %f" % (len(false_positives)*100.0/len(stream))
    print "False negative rate = %f" % (len(false_negatives)*100.0/len(stream))

if __name__ == "__main__":
    numbers = readTokens(NUMBERS)
    size = len(numbers)*10 # Size of the bit array
    m = 10                 # No. of hash functions
    bloom_filter = constructBloomFilter(numbers, size, m)
    stream = readTokens(IN_FILE)
    positives, negatives = bloom_filter.apply(stream)
    print positives
    print negatives
    printAccuracy(numbers, stream, positives, negatives)

if __name__ == "test":
    numbers = readTokens(NUMBERS)
    size = len(numbers)*10 # Size of the bit array
    m = 10                 # No. of hash functions
    bloom_filter = constructBloomFilter(numbers, size, m)
    stream = readTokens(IN_FILE)
    positives, negatives = bloom_filter.apply(stream)
    printAccuracy(numbers, stream, positives, negatives)
