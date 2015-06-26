/**************************************************************************
 * ARock for l1 or l2 regularized logistic regression.
 *
 * 1) for l2 regularization, we solve the following problem
 *
 *    min lambda/2 |x|_2^2 + 1/N \sum_i log(1 + exp(b_i * (a_i * x)) )
 *
 * where N is the total number of samples, and a_i is the ith sample, 
 * b_i is the label for a_i. And we let A be the data matrix, with size
 * num_features x num_samples, and let b be the label vector with length
 * num_samples.
 *
 * 2) for l1 regularization, we solve
 *
 *    min lambda |x|_1 + 1/N \sum_i log(1 + exp(b_i * (a_i * x)) )
 *
 *
 * Algorithm:
 * For l1 regularization problem:
 *    x^{k+1} = x^{k} - eta s_k \odot S_i
 *
 * For l2 regularization problem:
 *    x^{k+1} = x^{k} - eta s_k \odot S_i
 *
 * Implementation:
 *     1. generate a random number idx in [1, num_features];
 *     3. evaluate S_i based on the A'x, x and b in the shared memory;
 *     4. update x[idx] by: x[idx] -= eta * S_i;
 *     5. update A'x[idx] by: A'x[idx]  -= eta * S_i * A'(idx, :)
 *
 * Date Created:  01/29/2015
 * Date Modified: 01/29/2015 (created the file)
 *                02/25/2015 (finished the implementation for both l1 and l2)
 *                02/28/2015 (fixed some bugs and add a lot more comments)
 *                06/11/2015 (change the comments style, and fixed typos)
 *
 * TODO: 1) a better way for choosing step size;
 *       2) a better stopping criterion;
 *             
 * Author:        Zhimin Peng, Yangyang Xu, Ming Yan, Wotao Yin
 * Contact:       zhimin.peng@math.ucla.edu
*************************************************************************/

#ifndef AROCK_INCLUDE_LOGISTIC_H
#define AROCK_INCLUDE_LOGISTIC_H
#include "matrices.h"

/********************************************************************
 *  calculates l2 regularized objective
 *  Input:
 *     A:      the data matrix with size num_features x num_samples
 *     (type T, it can be sparse matrix SpMat or Matrix)
 *     b:      the observation labels for each sample
 *     (Vector)
 *     x:      the unknowns
 *     (Vector, size is the number of features)
 *     Atx:    A'*x, which is stored in shared memory for efficient
 *             computation
 *     lambda: regularization parameter
 *      
 *  Output: objective value
 *     (double)
 *******************************************************************/
template <typename T>
double l2_objective(T& A, Vector& b, Vector& x, Vector& Atx, Parameters para);


/********************************************************************
 *  calculates l1 regularized objective
 *  Input:
 *     A:      the data matrix with size num_features x num_samples
 *     (type T, it can be sparse matrix SpMat or Matrix)
 *     b:      the observation labels for each sample
 *     (Vector)
 *     x:      the unknowns
 *     (Vector, size is the number of features)
 *     Atx:    A'*x, which is stored in shared memory for efficient
 *             computation
 *     lambda: regularization parameter
 *      
 *  Output: objective value
 *     (double)
 *******************************************************************/
template <typename T>
double l1_objective(T& A, Vector& b, Vector& x, Vector& Atx,  Parameters para);
 

// calculate the forward gradient
double forward_gradient(Matrix& A, Vector& b, Vector& Atx, int idx);

double forward_gradient(SpMat& A, Vector& b, Vector& Atx, int idx);


/*******************************************************************
 *
 *  ARock for l2 regularized logistic regression (gradient descent)
 *
 *  Input:
 *     A: data matrix (each row is a feature, each column is a sample.)
 *     (T can be SparseMatrix or Matrix)
 *     b: label (the label for the corresponding observation, +1/-1)
 *     (Vector)
 *     x:      the unknown variable
 *     (Vector)
 *     lambda: regularization parameter
 *     (double)
 *     Ax: temporary variable in shared memory for storing A*x
 *     (Vector)
 *     flag:   flag for output, if true, then output result;
 *     (bool)
 *     
 *  Output:
 *     (none)
 *
 ********************************************************************/
template <typename T>
void l2_logistic(T& A, Vector& b, Vector& x, Vector &Atx, Parameters para);


/***********************************************************************
 *
 *  ARock solves l1 regularized logistic regression with forward
 *  backward splitting
 *
 *  Input:
 *     A: data matrix (each row is a feature, each column is a sample.)
 *     (T can be SparseMatrix or Matrix)
 *     b: label (the label for the corresponding observation, +1/-1)
 *     (Vector)
 *     x:      the unknown variable
 *     (Vector)
 *     lambda: regularization parameter
 *     (double)
 *     Ax: temporary variable in shared memory for storing A*x
 *     (Vector)
 *     flag:   flag for output, if true, then output result;
 *     (bool)
 *     
 *  Output:
 *     (none)
 *
 *********************************************************************/
template <typename T>
void l1_logistic(T& A, Vector& b, Vector& x, Vector &Atx, Parameters para);


/***********************************************************************
 *
 *  Synchronous forward backward splitting for solving l1 regularized
 *  logistic regression.
 *
 *  Input:
 *     A: data matrix (each row is a feature, each column is a sample.)
 *     (T can be SparseMatrix or Matrix)
 *     b: label (the label for the corresponding observation, +1/-1)
 *     (Vector)
 *     x:      the unknown variable
 *     (Vector)
 *     lambda: regularization parameter
 *     (double)
 *     Ax: temporary variable in shared memory for storing A*x
 *     (Vector)
 *     flag:   flag for output, if true, then output result;
 *     (bool)
 *     
 *  Output:
 *     (none)
 *
 *********************************************************************/
template <typename T>
void syn_l1_logistic(T& A, Vector& b, Vector& x, Vector &Atx, Parameters para);


#endif
