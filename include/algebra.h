/***********************************************************************
 *  Header file for linear algebra operations, including functions       
 *  such as norm, dot, sub, add, multiplication                          
 *                                                                       
 * Date Created:  01/29/2015                                             
 * Date Modified: 01/29/2015                                             
 *                02/19/2015                                             
 *                04/26/2015 (make it to Google C++ style)
 *                04/27/2015 (finished adding comments to the functions)
 *
 * Author:        Zhimin Peng, Yangyang Xu, Ming Yan, Wotao Yin          
 * Contact:       zhimin.peng@math.ucla.edu                              
 ***********************************************************************/

#ifndef AROCK_INCLUDE_ALGEBRA_H
#define AROCK_INCLUDE_ALGEBRA_H
#include <string>
#include <iomanip>
#include "matrices.h"


/***************************************
 * store various parameters for basic 
 * initialization, stopping creterion,
 * and output flag
 ***************************************/
struct Parameters
{
  int MAX_EPOCH;     // maximum number of epochs
  double lambda;     // regularization parameter
  bool is_sparse;    // if the data is sparse or not
  std::string type;  // type of the regularizer i.e., l1 or l2
  bool flag;         // flag for the output, if 1 output iterations;
  double step_size;
  double block_size;
 Parameters():
  MAX_EPOCH(10),     // initialize the maximum number of epochs to 10
    lambda(1),       // initialize the penality parameter to 1
    is_sparse(1),    // use sparse data as the default input data type
    step_size(0.5),  // set the initial step_size to 0.5
    type("l1"),      // the type of regularization, initialize to l1
    block_size(20),  // the size of block of coordinates
    flag(0) {}       // flag = 0 meaning no output for function value or residual
};


/*************************************************
 * shrinkage function or soft threshold function,
 *
 * return x -t, if x > t
 * return -x +t, if -x < t
 * return 0 otherwise.
 *
 * Input:
 *     x:      the input value
 *     (double)
 *     t:      the shrinkage parameter
 *     (double)
 *     
 * Output:
 *     result: the value after shrinkage
 *     (double)
 *
 ************************************************/
double shrink(double x, double t);


/************************************************* 
 * return the norm of a vector for a given type
 * 
 * Input:
 *       x -- the input vector
 *            (Vector)
 *    type -- 0 means zero norm,
 *            1 means l1 norm,
 *            2 means l2 norm.
 *            3 means infinity norm
 *            (int)
 * Output:
 *  result -- a scalar
 *              (double)
 ********************************************** */
double norm(Vector& x, int type);


/**************************************
 * calculate the l2 norm of a vector
 *************************************/
double norm(Vector& x);


/***************************************************
 * calculate the l2 norm for each column of
 * dense matrix A
 *
 * Input:
 *     A -- the target matrix
 *          (Matrix)
 * Output:
 *   nrm -- vector of size num_cols to store results
 *          (Vector)
 ***************************************************/
void calculate_column_norm(Matrix& A, Vector& nrm);


/***************************************************
 * calculate the l2 norm for each column of
 * sparse matrix A
 *
 * Input:
 *     A -- the target matrix
 *          (SpMat)
 * Output:
 *   nrm -- vector of size num_cols to store results
 *          (Vector)
 ***************************************************/
void calculate_column_norm(SpMat& A, Vector& nrm);


/***************************************************
 * subtract b from a, i.e.,
 * a = a - b
 * Input:
 *      a -- a vector
 *           (Vector)
 *      b -- a vector
 *           (Vector)
 * Output:
 *      a -- the subtracted vetor
 *           (Vector)
 **************************************************/
void sub(Vector& a, Vector& b);


/**************************************************
 * subtract a scalar times a row
 * of matrix from a given vector,
 * i.e., 
 *   a = a - scalar * A(row, :)
 *
 * Input:
 *      a -- a vector
 *      A -- a matrix
 *    row -- the row number
 * scalar -- the scalar
 *
 * Output:
 *      a -- the subtracted vetor
 *************************************************/
void sub(Vector& a, Matrix& A, int row, double scalar); // dense matrix, dense vector
void sub(Vector& a, SpMat&  A, int row, double scalar); // sparse matrix, dense vector
void sub(SpVec&  a, SpMat&  A, int row, double scalar); // sparse matrix, sparse vector
void sub(SpVec&  a, Matrix& A, int row, double scalar); // dense matrix, sparse vector


/***************************************************
 * add b to a, i.e.,
 * a = a + b
 *
 * Input:
 *      a -- a vector
 *      b -- a vector 
 *
 * Output:
 *      a -- the added vector
 **************************************************/
void add(Vector &a, Vector& b); // add two dense vectors
void add(Vector &a, SpVec&  b); // add two sparse vectors


/***************************************************
 * calculate the inner product of two vectors
 * a = a' * b
 *
 * Input:
 *      a -- a vector
 *           (Vector)
 *      b -- a vector
 *           (Vector)
 *
 * Output:
 * result -- a scalar represent the inner product 
 ***************************************************/
double dot(Vector &a, Vector &b);


/***************************************************
 * calculate inner product of A(row, :) * x
 *
 * result = A(row, :) * x
 *
 * Input:
 *       a -- a vector
 *       b -- a vector 
 *
 * Output:
 *  result -- a scalar represent the inner product 
 ***************************************************/
double dot(SpMat& A,  Vector& x, int row); // sparse matrix
double dot(Matrix& A, Vector& x, int row); // dense matrix


// print a vector
void print(Vector x);

// print a dense matrix
void print(Matrix &A);

// print a sparse matrix 
void print(SpMat &A);


/*************************************************
 * calcuate A' * x (A transpose multiplied x)
 *  Atx = A'*x
 *
 * Input:
 *      A -- a dense matrix
 *      x -- a dense vector
 *
 * Output:
 *    Atx -- the result vector
 *************************************************/
void trans_multiply(Matrix& A, Vector&x, Vector& Atx);


/*************************************************
 * multiply A with x
 *  Ax = A*x
 *
 * Input:
 *      A -- a dense or sparse matrix
 *      x -- a dense vector
 *
 * Output:
 *     Ax -- the result vector
 *************************************************/
void multiply(SpMat &A,  Vector &x, Vector& Ax);
void multiply(Matrix &A, Vector &x, Vector& Ax);


/***************************************************
 * multiply A with A', i.e., AAt = A * A'
 * the calculation is efficient for row major matrix
 *
 * Input:
 *      A -- a dense or sparse matrix
 *
 * Output:
 *    AAt -- the result matrix
 **************************************************/
void multiply(SpMat  &A,  SpMat &AAt);
void multiply(Matrix &A, Matrix &AAt);


/*****************************************************
 * copy matrix A from start row to end-1 row to
 * matrix B, i.e., B = A(start:end-1, :)
 *
 * Input:
 *      A -- a dense or sparse matrix
 *  start -- the index for the starting row
 *    end -- the past-the-end index
 *
 * Output:
 *      B -- the submatrix of A
 ****************************************************/
void copy(Matrix& A, Matrix& B, int start, int end);
void copy(SpMat&  A, SpMat&  B, int start, int end);


/*****************************************************
 * copy part of a vector to another vector 
 * , i.e., y = x(start:end)
 *
 * Input:
 *      x -- a dense vector
 *  start -- the index for the starting row
 *    end -- the past one index for the ending index
 * Output:
 *      y -- the subvector of x
 ****************************************************/
void copy(Vector& x, Vector& y, int start, int end);


#endif
