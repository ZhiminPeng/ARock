/****************************************************************************
 *
 * ARock for l1 or l2 regularized least square problems.
 *
 *
 * 1) for l2 regularized problem, we solve the following problem
 *
 *     min lambda/2 * ||x||_2^2 + 1/2 ||A'x - b||^2
 *
 * 2) for l1 regularized problem, we solve the following LASSO
 *
 *    min lambda * ||x||_1 + 1/2 ||A'x - b||^2
 *
 * Algorithm:
 *  1) for l2_ls, we use gradient descent method, i.e.,
 *     x^{k+1} = x^{k} - eta * s_k \odot (lambda*x_hat + A*A'x_hat - Ab)
 *
 *  2) for l1_ls, we use forward backward splitting, i.e.,
 *     x^{k+1} = x^{k} - eta * s_k \odot (
 *               x_hat-prox_gamma|x|_1(x_hat-gamma(A*A'x_hat-Ab))
 *
 * Implementation:
 *  1) for l2_ls, the algorithm has 4 main steps;
 *       1. each process randomly selects a coordinate (idx);
 *
 *       2. calculates S_i =  lambda * x(idx) + At(idx,:)*Atx - Ab(idx);
 *
 *       3. updates x(idx) by: x(idx) -= eta * S_i;
 *
 *	 4. updates Atx -= eta*S_i * A(idx, :)'.
 *
 *
 *  2) for l1_ls, the algorithm has 6 main steps;
 *       1. each process randomly selects a coordinate (idx);
 *
 *       2. calculates foward step: 
 *	       forward = x_hat(idx) - gamma (At(idx,:)*Atx - Ab[idx]);
 *
 *	 3. evaluates backward step:
 *             backward = shrink(forward, gamma*lambda);
 *
 *	 4. calculates S_i = x_hat(idx) - backward;
 *
 *       5. updates x(idx) by: x(idx) -= eta * S_i;
 *
 *	 6. updates Atx -= eta*S_i*A(idx, :)'.
 *
 * Tricks:
 *  1) We don't need to precompute AAt, but we still have the same
 *     per iteration complexity O(num_samples). This is achieved by maintaining
 *     Atx in the shared memory. We use atomic write to ensure consistence of
 *     Atx and A'*x. Note that if the data is sparse, precomputing AAt might 
 *     ruin the sparsity of the data.
 *
 *  2) The columns in A should be normalized, otherwise, we may need to choose a
 *     smaller stepsize.
 *
 *  3) Matrix A is row major with size num_features x num_samples. This is for
 *     efficient calculation.
 *
 * TODO: 1) Need to have a better way to choose step size. Current step size requires
 *         normalizatio of the data.
 *
 *    	 2) Need to have a automatic stopping criterion. Current implementation based
 *	   on the total number of epoches.
 *
 *	 3) modify the add and subtract function so that they can be sparse vector type
 *
 * Date Created:  02/26/2015
 * Date Modified: 02/28/2015 (added lots of comments, moved shrink function to algebra.h)
 *                03/17/2015 (shared write after updating several coordinates)
 *                06/09/2015 (add the code to ARock style, add more comments)
 *                06/11/2015 (correct comments style, and fix typos)
 *
 * Author:        Zhimin Peng, Yangyang Xu, Ming Yan, Wotao Yin
 * Contact:       zhimin.peng@math.ucla.edu
 *
 **************************************************************************************/

#ifndef AROCK_INCLUDE_LEAST_SQUARE_H
#define AROCK_INCLUDE_LEAST_SQUARE_H
#include "matrices.h"

/**********************************************************************
 * calculates the residual of the l2 regularized least square.
 * i.e., ||grad f(x)|| = ||lambda x + A(A'x - b)||.
 *
 *Input:
 *     A:      the data matrix with size num_features x num_samples
 *             (type T, it can be sparse matrix SpMat or Matrix)
 *    Ab:      A*b
 *             (Vector)
 *     x:      the weights for different features, unknown variable.
 *             (Vector, size is the number of features)
 *     Atx:    A'*x, which is stored in shared memory for efficient
 *             computation.
 *             (Vector, length is num_samples)
 *     lambda: regularization parameter
 *             (double)
 *      
 * Output: residual.
 *     (double)
 *
 **********************************************************************/
template <typename T>
double l2_least_square_residual(T&      A,
                                Vector& Ab,
                                Vector& x,
                                Vector& Atx,
                                double  lambda);


/**********************************************************************
 * Calculates objective value for l2 regularized least square 
 *
 * Input:
 *     A:      the data matrix with size num_features x num_samples
 *             (type T, it can be sparse matrix SpMat or Matrix)
 *    Ab:      the observation labels for each sample
 *             (Vector)
 *     x:      the unknowns
 *             (Vector, size is the number of features)
 *   Atx:      A'*x, which is stored in shared memory for efficient
 *             computation
 *             (Vector)
 *lambda:      regularization parameter
 *             (double)
 *      
 * Output:     objective value
 *             (double)
 *
 *********************************************************************/
template <typename T>
double l2_objective(T&          A,
                    Vector&     b,
                    Vector&     x,
                    Vector&     Atx,
                    Parameters& para);


/**********************************************************************
 * Calculates the objective value for l1 regularized least square.
 *
 * Input:
 *     A:      the data matrix with size num_features x num_samples;
 *     (type T, it can be sparse matrix SpMat or Matrix)
 *     b:      the observation labels;
 *     (Vector)
 *     x:      the unknowns, weights for different features;
 *     (Vector, size is the number of features)
 *     Atx:    A'*x, which is stored in shared memory for efficient
 *             computation
 *     lambda: regularization parameter
 *      
 * Output: objective value
 *     (double)
 *
 *********************************************************************/
template <typename T>
double l1_objective(T&          A, 
		    Vector&     b, 
		    Vector&     x, 
		    Vector&     Atx,  
		    Parameters& para);


/************************************************************************
 * Finds the optimal solution for l2 regularized least square problem. 
 * The algorithm is parallel asynchronous stochastic coordinate descent
 * method.
 *
 * Input:
 *     A:      data matrix with size num_features x num_samples.
 *             (Matrix or SpMat)
 *     b:      label (the label for the corresponding observation)
 *             (Vector)
 *     x:      the unknown variables. Weights for different features.
 *             (Vector)
 *     lambda: regularization parameter (>=0)
 *             (double)
 *     Atx:    temporary variable in shared memory for storing A'*x
 *             (Vector)
 *     Ab:     temporary variable in shared memory for storing A*b
 *             (Vector)
 *     para: parameters. 
 *     (struct)
 *     
 * Output:
 *     (none)
 *
 **********************************************************************/
template <typename T>
void l2_ls(T&          A, 
	   Vector&     b, 
	   Vector&     x, 
	   Vector&     Atx, 
	   Vector&     Ab, 
	   Parameters& para);


/*************************************************************************
 *  Calculates the optimal solution for l1 regularized least square (lasso)
 *  The algorithm is forward backward splitting.
 *
 * Input:
 *     A:      data matrix; matrix size is num_features x num_samples
 *             (T can be sparse matrix (SpMat) or dense matrix (Matrix) )
 *     b:      label (the label for the corresponding observation, +1/-1)
 *             (Vector)
 *     x:      the unknown variable, weights for different features
 *             (Vector)
 *     lambda: regularization parameter
 *             (double)
 *     Atx:    temporary variable in shared memory for storing A'*x
 *             (Vector)
 *     Ab:     temporary variable in shared memory for storing Ab
 *             (Vector)
 *     para:   related parameters, includes MAX_EPOCH, lambda, flag.
 *             (Parameters, struct)
 *     
 * Output:
 *     (none)
 *
 ***********************************************************************/
template <typename T>
void l1_ls(T&          A, 
	   Vector&     b, 
	   Vector&     x, 
	   Vector&     Atx,
	   Vector&     Ab, 
	   Parameters& para);

#endif

// end of file
