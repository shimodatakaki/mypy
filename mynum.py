"""

"""

def nsplit(x, n):
    """
    split x into n parts
    :param x:
    :param n:
    :return:
    """
    m = int( len(x)/n )
    return [ x[m*i:m*(i+1)] for i in range(n) ]

if __name__=="__main__":
    import numpy as np
    a = np.ones(100)
    print( nsplit(a, 2) )