import numpy as np
import pyopencl as cl
import math
from plan import Plan
from mako.template import Template
from nengo_ocl.ocl.array import to_device

#Parallel version of plan_ragged_gather_gemv
#Should replicate all functionality except iterating over the "mm"s
def plan_parallel_ragged_gather_gemv2(queue, Ms, Ns, alpha, A, A_js, X, X_js,
                       beta, Y, group_size = 32, Y_in=None, tag=None):
    """
    """
    
    # TODO: if alpha or beta is a float
    #       then render it into the kernel text.
    try:
        float(alpha)
        alpha = [alpha] * len(Y)
    except TypeError:
        pass

    try:
        float(beta)
        beta = [beta] * len(Y)
    except TypeError:
        pass

    cl_alpha = to_device(queue, np.asarray(alpha, Y.buf.dtype))
    cl_beta = to_device(queue, np.asarray(beta, Y.buf.dtype))
    

    if Y_in is None:
        Y_in = Y

    # XXX check for e.g. all Ns being the same thing
    #     especially all Ns == 1
    cl_Ns = to_device(queue, np.asarray(Ns, 'int32'))

    # XXX check that all the ints are ints not longs
    textconf = {
        'type_alpha': cl_alpha.ocldtype,
        'type_beta': cl_beta.ocldtype,
        'type_A': A.cl_buf.ocldtype,
        'type_X': X.cl_buf.ocldtype,
        'type_Y': Y.cl_buf.ocldtype,
        'y_len': len(Y),
        'lsize': group_size,
    }

    text = """
        __kernel void fn(
            const __global int *Ns,
            const __global ${type_alpha} * alphas,
            const __global int *A_starts,
            const __global ${type_A} *A_data,
            const __global int *A_js_starts,
            const __global int *A_js_lens,
            const __global int *A_js_data,
            const __global int *X_starts,
            const __global ${type_X} *X_data,
            const __global int *X_js_starts,
            const __global int *X_js_data,
            const __global ${type_beta} * betas,
            const __global int *Y_in_starts,
            const __global ${type_Y} *Y_in_data,
            const __global int *Y_starts,
            const __global int *Y_lens,
            __global ${type_Y} *Y_data)
        {
            //const int mm = get_global_id(1); //TODO
            
            
            __local ${type_Y} partialDotProduct[${lsize}]; //Scratch space for the dot products

            //Y is divided into groups of size group_size. Each work-item does enough dot-products to cover one of the groups
            for (uint yi = get_group_id(0); yi < ${y_len}; yi += get_num_groups(0)) {

                const __global int* X_js_row = X_js_data + X_js_starts[yi];
                const __global int* A_js_row = A_js_data + A_js_starts[yi];
                
                const ${type_alpha} alpha = alphas[yi];
                const ${type_beta} beta = betas[yi];
                
                int y_offset = Y_starts[yi];
                int y_in_offset = Y_in_starts[yi];
                Y_data[y_offset] = beta * Y_in_data[y_in_offset];
                
                float sum = 0; 
                int n_dot_products = A_js_lens[yi]; //Do all of xjs dot products at same time
                for(int j = 0; j < n_dot_products; j++) {
                    int x_ji = X_js_row[j];
                    int a_ji = A_js_row[j];
                    int N_i = Ns[a_ji];
                              
                    const __global ${type_A}* A_row = A_data + A_starts[a_ji]; //Get the rows for the product
                    const __global ${type_X}* X_row = X_data + X_starts[x_ji];

                    //Each work item will do some fraction of the multiplications and store the result locally
                    for (uint x = get_local_id(0); x < N_i; x += get_local_size(0)) {
                        sum += A_row[x] * X_row[x]; 
                    }
                }
                partialDotProduct[get_local_id(0)] = sum;
                
                //Parallel reduction of locally stored sums
                for (uint stride = 1; stride < get_local_size(0); stride *= 2) { 
                    barrier(CLK_LOCAL_MEM_FENCE); 
                    
                    uint index = 2 * stride * get_local_id(0);
                    if (index < get_local_size(0)) { 
                        partialDotProduct[index] += partialDotProduct[index + stride]; 
                    } 
                } 
                
                //Multiply by alpha and store the result.
                if (get_local_id(0) == 0) {
                    Y_data[yi] += alpha * partialDotProduct[0];
                    barrier(CLK_LOCAL_MEM_FENCE); 
                }   
            }
        }
        """

    text = Template(text, output_encoding='ascii').render(**textconf)
    
    #Make the global size the closest multiple of the group size (ceiling)
    y_size = int(math.ceil(len(Y) / float(group_size))) * group_size
    gsize = (y_size,)
    lsize = (group_size,)
    
    _fn = cl.Program(queue.context, text).build().fn
    full_args = (cl_Ns,
                 cl_alpha,
                 A.cl_starts,
                 A.cl_buf,
                 A_js.cl_starts,
                 A_js.cl_lens,
                 A_js.cl_buf,
                 X.cl_starts,
                 X.cl_buf,
                 X_js.cl_starts,
                 X_js.cl_buf,
                 cl_beta,
                 Y_in.cl_starts,
                 Y_in.cl_buf,
                 Y.cl_starts,
                 Y.cl_lens,
                 Y.cl_buf,
                )

    _fn.set_args(*[arr.data for arr in full_args])
    
    rval = Plan(queue, _fn, gsize, lsize,
                name='ref_parallel_ragged_gather_gemv',
                tag=tag,
               )
    # prevent garbage-collection
    rval.alpha = cl_alpha
    rval.beta = cl_beta
    rval.Ns = cl_Ns
    return rval

#Original gemv kernel
def plan_ragged_gather_gemv(queue, Ms, Ns, alpha, A, A_js, X, X_js,
                       beta, Y, Y_in=None, tag=None):
    """
    """
    # TODO: if alpha or beta is a float
    #       then render it into the kernel text.
    try:
        float(alpha)
        alpha = [alpha] * len(Y)
    except TypeError:
        pass

    try:
        float(beta)
        beta = [beta] * len(Y)
    except TypeError:
        pass

    cl_alpha = to_device(queue, np.asarray(alpha, Y.buf.dtype))
    cl_beta = to_device(queue, np.asarray(beta, Y.buf.dtype))

    if Y_in is None:
        Y_in = Y

    # XXX check for e.g. all Ns being the same thing
    #     especially all Ns == 1
    cl_Ns = to_device(queue, np.asarray(Ns, 'int32'))

    # XXX check that all the ints are ints not longs
    textconf = {
        'type_alpha': cl_alpha.ocldtype,
        'type_beta': cl_beta.ocldtype,
        'type_A': A.cl_buf.ocldtype,
        'type_X': X.cl_buf.ocldtype,
        'type_Y': Y.cl_buf.ocldtype,
    }

    text = """
        __kernel void fn(
            __global int *Ns,
            __global ${type_alpha} * alphas,
            __global int *A_starts,
            __global ${type_A} *A_data,
            __global int *A_js_starts,
            __global int *A_js_lens,
            __global int *A_js_data,
            __global int *X_starts,
            __global ${type_X} *X_data,
            __global int *X_js_starts,
            __global int *X_js_data,
            __global ${type_beta} * betas,
            __global int *Y_in_starts,
            __global ${type_Y} *Y_in_data,
            __global int *Y_starts,
            __global int *Y_lens,
            __global ${type_Y} *Y_data)
        {
            const int mm = get_global_id(0);
            const int bb = get_global_id(1);
            const int M = Y_lens[bb];
            if (mm < M)
            {
                const ${type_alpha} alpha = alphas[bb];
                const ${type_beta} beta = betas[bb];

                int n_dot_products = A_js_lens[bb];
                int y_offset = Y_starts[bb];
                int y_in_offset = Y_in_starts[bb];

                X_js_data += X_js_starts[bb];
                A_js_data += A_js_starts[bb];

                Y_data[y_offset + mm] = beta * Y_in_data[y_in_offset + mm];

                for (int ii = 0; ii < n_dot_products; ++ii)
                {
                    int x_ji = X_js_data[ii];
                    int a_ji = A_js_data[ii];
                    int N_i = Ns[a_ji];
                    int x_offset = X_starts[x_ji];
                    int a_offset = A_starts[a_ji];

                    // compute the matrix-vector product
                    // dot(X[x_ji], A[a_ji])
                    ${type_Y} y_sum = 0;
                    for (int nn = 0; nn < N_i; ++nn) //Parallel reduction. How big is N_i?
                    {
                        y_sum += X_data[x_offset + nn]
                        * A_data[a_offset + nn * M + mm];
                    }
                    Y_data[y_offset + mm] += alpha * y_sum;
                }
            }
        }
    """

    text = Template(text, output_encoding='ascii').render(**textconf)
    gsize = (int(max(Ms)), int(len(Y)),)
    lsize = None
    _fn = cl.Program(queue.context, text).build().fn
    full_args = (cl_Ns,
                 cl_alpha,
                 A.cl_starts,
                 A.cl_buf,
                 A_js.cl_starts,
                 A_js.cl_lens,
                 A_js.cl_buf,
                 X.cl_starts,
                 X.cl_buf,
                 X_js.cl_starts,
                 X_js.cl_buf,
                 cl_beta,
                 Y_in.cl_starts,
                 Y_in.cl_buf,
                 Y.cl_starts,
                 Y.cl_lens,
                 Y.cl_buf,
                )
    #print [str(arr.dtype)[0] for arr in full_args]
    _fn.set_args(*[arr.data for arr in full_args])
    rval = Plan(queue, _fn, gsize, lsize,
                name='ref_ragged_gather_gemv',
                tag=tag,
               )
    # prevent garbage-collection
    rval.alpha = cl_alpha
    rval.beta = cl_beta
    rval.Ns = cl_Ns
    return rval