'''
Created on Oct 8, 2014

@author: d1farre
'''
input_file = "cf_input.txt"

def readUtilityMatrx(fileName):
    with open(fileName) as file:
        # First line has dimensions of the utility matrix (m,n)
        line_cnt = 0
        m = 0
        n = 0
        U = []
        for line in file:
            if line_cnt == 0:
                tokens = line.split(' ')
                m = int(tokens[0])
                n = int(tokens[1])
                U = [[0.0]*n for i in xrange(m)]
            else:
                tokens = line.split(' ')
                for j in xrange(len(tokens)):
                    U[line_cnt-1][j] = int(tokens[j])
            line_cnt += 1
    return U
if __name__ == "__main__":
    U = readUtilityMatrix(input_file)

