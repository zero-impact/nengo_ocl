import numpy as np
from nengo_ocl.sim_npy import ragged_gather_gemv
from nengo_ocl.sim_npy import RaggedArray as RA
from nengo_ocl.sim_ocl import RaggedArray as CLRA
from plan import Prog

from gemv_batched import *

import pyopencl as cl
ctx = cl.create_some_context()

def test_basic():
    A = RA([ [0.1, .2, .3, .4], [.5, .6]])
    Ms = [2, 1]
    Ns = [2, 2]
    X = RA([ [3, 5] ])

    X_js = RA([[0], [0]])
    A_js = RA([[1], [0]])

    Y = RA([[0.0], [2, 3],])

    print ragged_gather_gemv(Ms, Ns, .5, A, A_js, X, X_js, .1, Y)
    result1 = Y.buf

    queue = cl.CommandQueue(ctx)

    A = CLRA(queue, [[0.1, .2, .3, .4], [.5, .6]])
    Ms = [2, 1]
    Ns = [2, 2]
    X = CLRA(queue, [[3, 5]])

    X_js = CLRA(queue, [[0], [0]])
    A_js = CLRA(queue, [[1], [0]])

    Y = CLRA(queue, [[0.0], [2, 3],])


    plan = plan_ragged_gather_gemv(queue, Ms, Ns, .5, A, A_js, X, X_js, .1, Y)
    plan()
    result2 = Y.buf.get()
    assert np.allclose(result1, result2)
    
def test_basic2():
    L = 512  # -- length of each vector
    N = 1000 # -- number of vectors
    
    A = RA([np.random.randn(L) for i in range(N)])
    Ms = [1] * N
    Ns = [L] * N
    X = RA([np.random.randn(L) for i in range(N)])

    X_js = RA( [[i] for i in range(N) + range(N)])
    A_js = RA( [[i] for i in range(N) + list(reversed(range(N)))])

    Y = RA([[1.0] for i in range(2 * N)])

    print ragged_gather_gemv(Ms, Ns, .5, A, A_js, X, X_js, .1, Y, use_raw_fn = False)
    result1 = Y.buf




def test_reduction_speed():
    queue = cl.CommandQueue(
        ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    np.random.seed(50)
    
    L = 512  # -- length of each vector
    N = 1000 # -- number of vectors

    Arows = [1] * N
    Acols = [L] * N
    A = CLRA(queue, [np.random.randn(L) for i in range(N)]) #adjacency matrix for synapse weights. Row is a soma, column soma
    X = CLRA(queue, [np.random.randn(L) for i in range(N)]) #X potential on axons. X*A = synapse strength times current strength. Dot product from row of A to row of X to get total current that goes into a soma.
    #List of lists of size 1.
    #Xjs can be different lengths too!

    X_js = CLRA(queue, [[i] for i in range(N) + range(N)])
    A_js = CLRA(queue, [[i] for i in range(N) + list(reversed(range(N)))])

    Y = CLRA(queue, [[1.0] for i in range(2 * N)])

    plan = new_plan_ragged_gather_gemv(queue, Arows, Acols,
                                   1.0, A, A_js, X, X_js,
                                   0.0, Y)
    for i in xrange(10):
        plan(profiling=True)

    print 'n_calls         ', plan.n_calls
    print 'queued -> submit', plan.atime
    print 'submit -> start ', plan.btime
    print 'start -> end    ', plan.ctime
    print Y.buf
    print len(Y)
    
def my_test():
    print "Modified Kernel: "
    print
    
    queue = cl.CommandQueue(
        ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    np.random.seed(50)
    
    L = 512  # -- length of each vector
    N = 1000 # -- number of vectors

    Arows = [1] * N
    Acols = [L] * N
    A = CLRA(queue, [np.random.randn(L) for i in range(N)]) #adjacency matrix for synapse weights. Row is a soma, column soma
    X = CLRA(queue, [np.random.randn(L) for i in range(N)]) #X potential on axons. X*A = synapse strength times current strength. Dot product from row of A to row of X to get total current that goes into a soma.
    #List of lists of size 1.
    #Xjs can be different lengths too!

    X_js = CLRA(queue, [[i] for i in range(N) + range(N)])
    A_js = CLRA(queue, [[i] for i in range(N) + list(reversed(range(N)))])

    Y = CLRA(queue, [[1.0] for i in range(2 * N)])
    
    R = CLRA(queue, [np.zeros(L) for i in range(2 * N)])
    
    #plan = new_plan_ragged_gather_gemv(queue, Arows, Acols,
    #                               1.0, A, A_js, X, X_js,
    #                               0.0, Y)
    
    plans = []
    plans.append(plan_multiply(queue, Arows, Acols,A, A_js, X, X_js, Y, R))
    plans.append(plan_reduce(queue, Arows, Acols, 1.0, A, A_js, X, X_js, 0.0, Y, R))
    
    prog = Prog(plans)
    for i in xrange(10):
        prog(profiling=True)
    #plans[0](profiling=True)
    #plans[1](profiling=True)
    
    #print R.buf
    #print Y.buf
    #print len(Y)
    
    print 'n_calls         ', prog.plans[0].n_calls + prog.plans[0].n_calls
    print 'queued -> submit', prog.plans[0].atime + prog.plans[0].atime 
    print 'submit -> start ', prog.plans[0].btime + prog.plans[0].btime 
    print 'start -> end    ', prog.plans[0].ctime  + prog.plans[0].ctime 
    print Y.buf
    
if __name__ == "__main__":
    test_reduction_speed()
    my_test()
    

