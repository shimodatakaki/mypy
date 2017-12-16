"""
Example2: mycvxopt test
"""
import mycvxopt
import numpy as np

def test():
    """

    :return:
    """

    """
    Linear Quadratic Programming Example
    """
    M = np.array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
    P = np.dot(M.T, M)
    q = np.dot(np.array([3., 2., 3.]), M).reshape((3,))
    if True:
        G = np.array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
        h = np.array([3., 2., -2.]).reshape((3,))
    else:
        G = None
        h = None
    print(mycvxopt.solve("qp", [P, q], G=G, h=h))

    """
    Linear Programming
    """
    c = np.array([-4., -5.])
    G = np.array([[2., 1., -1., 0.], [1., 2., 0., -1.]]).reshape((4, 2))
    h = np.array([3., 3., 0., 0.])
    print(mycvxopt.solve("lp", [c], G=G, h=h))

    """
    Second Order Cone Programming
    min x1 + x2
    s.t. ||x||2 <= 1 eqiv. x in circle with radius=1
    :return:
    """
    A0 = np.array([[1., 0], [0, 1]])
    A0.reshape((2, 2))
    b0 = np.array([0., 0])
    b0.reshape((2, 1))
    c0 = np.array([0., 0])
    c0.reshape((2, 1))
    d0 = np.array([1.])

    gq0, hq0 = mycvxopt.qc2socp(A0, b0, c0, d0)

    Gq = [gq0]
    hq = [hq0]

    c = np.array([1., 1])
    print(mycvxopt.solve("socp", [c], Gql=Gq, hql=hq))

    """
    Semi-Definite Progamming
    """
    c = np.array([0., 0.])

    hs0 = [[1., 0.], [0., 1.]]
    Gs0 = [[0., 1, 1, 0], [0., 0, 0, 0]]

    Gs = [Gs0]
    hs = [hs0]

    Gl = np.eye(2) * -1
    hl = - 0. * np.ones(2)

    sol = mycvxopt.solve("sdp", [c], G=Gl, h=hl, Gsl=Gs, hsl=hs)
    print(sol)

if __name__ == "__main__":
    test()

