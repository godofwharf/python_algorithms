'''
Created on Oct 8, 2014

@author: d1farre
'''
import math
GRAPH ="graph.txt"

def delta(a, b):
    return sum([math.fabs(x - y) for x, y in zip(a, b)])

def dotproduct(u, v):
    return sum([x * y for x, y in zip(u,v)])

def multiply(M, v):
    n = len(M)
    return [dotproduct(M[i], v) for i in xrange(0, n)]

''' computePageRank takes in an adjacency matrix of a graph and a beta parameter
and returns the PageRank vector for the nodes in the input graph '''
def computePageRank(M, beta = 0.85):
    ''' PageRank equation is r(n) = A.r(n-1), n >= 1
    We use Power Iteration method to solve the above equation for r (stationary distribution)
    To avoid spider traps and dead ends, A needs to be column stochastic and aperiodic.
    To ensure this, we add teleport links from each node to every other node in the
    graph and the probability of a random surfer traversing this link is (1-beta)/no_of_nodes
    '''
    A = addTeleportLinks(M, beta)
    r_prev = [1.0 / len(M)] * len(M)
    r_cur = [0.0] * len(M)
    itr = 0
    while True:
        r_cur = multiply(A, r_prev)
        itr += 1
        if delta(r_cur, r_prev) < 1e-6:
            break
        r_prev = r_cur
    print "No of iterations = %d" % itr
    return r_cur

''' same as computePageRank but uses less memory for sparse graphs '''
def computePageRank_Optimized(M, beta = 0.85):
    r_prev = [1.0 / len(M)] * len(M)
    r_cur = [0.0] * len(M)
    itr = 0
    while True:
        r_temp = [0.0] * len(M)
        for i in xrange(len(M)):
            for j in xrange(len(M[i])):
                r_temp[M[i][j]] += beta * (r_prev[i] / len(M[i]))
        S = sum(r_temp)
        for i in xrange(len(r_temp)):
            r_cur[i] = r_temp[i] + (1 - S) / len(M)
        itr += 1
        if delta(r_cur, r_prev) < 1e-6:
            break
        r_prev = list(r_cur)
    print "No of iterations = %d" % itr
    return r_cur

def addTeleportLinks(M, beta):
    for i in xrange(len(M)):
        for j in xrange(len(M[i])):
            M[i][j] = beta * M[i][j] + (1 - beta) / len(M)
    return M

def readGraphList(fileName):
    M = []
    with open(fileName) as file:
        line_cnt = 0
        for line in file:
            if line_cnt > 0:
                tokens = line.split(' ')[1:]
                row = []
                for token in tokens:
                    row.append(int(token))
                M.append(row)
            line_cnt += 1
    return M

def readGraphMatrix(fileName):
    out_degree = []
    M = []
    N = 0
    with open(fileName) as file:
        line_cnt = 0
        for line in file:
            if line_cnt == 0:
                N = int(line)
                M = [[0.0] * N for i in xrange(N)]
                out_degree = [0] * N
            else:
                tokens = line.split(' ')[1:]
                for token in tokens:
                    i = line_cnt - 1
                    j = int(token)
                    M[j][i] = 1.0
                    out_degree[i] += 1
            line_cnt += 1
    for i in xrange(N):
        for j in xrange(N):
            M[i][j] /= out_degree[j]
    return M

def run_tests():
    M = readGraphMatrix(GRAPH)
    N = readGraphList(GRAPH)
    pr1 = computePageRank(M)
    pr2 = computePageRank_Optimized(N)
    print pr1, pr2
    matches = [1 for i, j in zip(pr1, pr2) if i == j]
    if matches == len(pr1):
        print "Success"
    else:
        print "Failure"

if __name__ == "__main__":
    run_tests()
    M = readGraphMatrix(GRAPH)
    pageRanks = computePageRank(M)
    print pageRanks



