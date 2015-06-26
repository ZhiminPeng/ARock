/******************************************************************************
 * implementation file for linear algebra operations, include functions such as
 * norm, dot, sub, add, multiply. Please refer to include/algebra.h for more
 * details of the functions.
 *
 * Date Created:  01/29/2015
 * Date Modified: 01/29/2015
 *                02/19/2015
 *                04/27/2015 (remove some redundant functions, clean the code)
 * Author:        Zhimin Peng, Yangyang Xu, Ming Yan, Wotao Yin
 * Contact:       zhimin.peng@math.ucla.edu
 ******************************************************************************/

#include <vector>
#include <string>
#include <iomanip>
#include "matrices.h"
using namespace std;
// shrinkage function
double shrink(double x, double t)
{
  if(x>t) return x-t;
  else if(x<-t) return x+t;
  else return 0.;
}


double norm(Vector& x, int type)
{
  double tmp = 0.0;
  if(type==0)
  {
    for (unsigned i = 0; i < x.size(); ++i)
      if(x[i]!=0) tmp = tmp + 1;
  }else if(type == 2)
  {
    for (unsigned i = 0; i < x.size(); ++i)
      tmp += x[i] * x[i];
    tmp = sqrt(tmp);
  }else if(type == 1)
  {
    for (unsigned i = 0; i < x.size(); ++i)
      tmp += fabs(x[i]);
  }else if(type == 3)
  {
    for (unsigned i = 0; i < x.size(); ++i)
      tmp = max(fabs(x[i]), tmp);
  }
  return tmp;
}


// Calculates the two norm of a vector.
double norm(Vector& x)
{
  return norm(x, 2);
}


// calculate the column norm of a matrix
void calculate_column_norm(Matrix& A, Vector& nrm)
{
  int num_cols = A.cols();
  int num_rows = A.rows();
  int i, j;
#pragma omp parallel private(i, j)
  {
#pragma omp for schedule (static)
    for ( i = 0; i < num_rows; ++i)
    {
      for (j = 0; j < num_cols; ++j)
      {
        nrm[j] += A(i,j)*A(i,j);
      }
    }
  }
  
  for(int j=0;j<num_cols;++j)
    nrm[j] = sqrt(nrm[j]);

  return;
}


// calculate the column norm of a matrix
void calculate_column_norm(SpMat& A, Vector& nrm)
{

  for (int k=0; k<A.outerSize(); ++k)
  {
    for (SpMat::InnerIterator it(A,k); it; ++it)
    {
      nrm[it.index()] += it.value() * it.value();
    }
  }
  int num_samples = A.cols();

  //# pragma omp parallel for num_threads(4)
  {
    for(int j=0;j<num_samples;++j)
      nrm[j] = sqrt(nrm[j]);
  }
  return;
}


// a = a - b
void sub(Vector& a, Vector& b)
{
  for (unsigned i = 0; i < a.size(); ++i)
    a[i] -= b[i];
}


// a = a - scalar * A(row, :)
void sub(Vector& a, Matrix& A, int row, double scalar)
{
  for (unsigned i = 0; i < a.size(); ++i)
    a[i] -= scalar * A(row, i);
}


// a = a - scalar * A(row, :)
void sub(Vector& a, SpMat& A, int row, double scalar)
{
  double result = 0.;
  for (SpMat::InnerIterator it(A, row); it; ++it)
    a[it.index()]-=scalar * it.value();
}


// a = a - scalar * A(row, :)
void sub(SpVec& a, SpMat& A, int row, double scalar)
{
  
  double result = 0.;
  for (SpMat::InnerIterator it(A, row); it; ++it)
    a.coeffRef(it.index())-=scalar * it.value();
}


// a = a - scalar * A(row, :)
void sub(SpVec& a, Matrix& A, int row, double scalar)
{
  
  double result = 0.;
  for (unsigned i = 0; i < a.size(); ++i)
    a.coeffRef(i)-=scalar * A(row, i);
}


// a = a + b
void add(Vector &a, Vector& b)
{
  for (unsigned i = 0; i < a.size(); ++i)
    a[i] += b[i];
  return;
}


void add(Vector &a, SpVec& b)
{
  for(SpVec::InnerIterator it(b); it; ++it)
    a[it.index()] += it.value();
  return;
}


// calculate the inner product of two vectors
double dot(Vector &a, Vector &b)
{
  double result = 0.;
  for (unsigned i = 0; i < a.size(); ++i)
  {
    result += a[i] * b[i];
  }
  return result;
}


// calcuate inner product of A(row, :) * x
double dot(SpMat& A, Vector& x, int row)
{
  double result = 0.;
  for (SpMat::InnerIterator it(A, row); it; ++it)
    result += it.value() * x[it.index()];
  return result;
}


// calcuate inner product of A(row, :) * x
double dot(Matrix& A, Vector& x, int row)
{
  double result = 0.;
  for (unsigned i = 0; i < A.cols(); ++i)
    result += A(row, i) * x[i];
  return result;
}


// print a vector
void print(Vector x)
{
  for (unsigned i = 0; i < x.size(); ++i)
    cout<<x[i]<<" ";
  cout<<endl;
}


// print a dense matrix
void print(Matrix &A)
{
  for (int i = 0; i < A.rows(); ++i)
  {
    for (int j = 0; j < A.cols(); ++j)
      cout << setw(10) << A(i, j);
    cout << endl;
  }
  cout << endl;
}


// print a sparse matrix 
void print(SpMat &A)
{
  cout<<A;
  return;
}


// calculate A' * x
void trans_multiply(Matrix& A, Vector&x, Vector& Atx)
{
  int m = A.rows(), n = A.cols();
  for (int i = 0; i < m; ++i)
  {
    for(int j = 0; j < n; ++j)
    {
      Atx[j] += A(i, j) * x[i];
    }
  }
}


// mutiply A with x, i.e., Ax = A * x
void multiply(SpMat &A, Vector &x, Vector& Ax)
{
  int dim = A.rows();
  for (int k=0; k<A.outerSize(); ++k)
    for (SpMat::InnerIterator it(A,k); it; ++it)
      Ax[k] += it.value() * x[it.index()];
  return;
}


// mutiply A with x, i.e., Ax = A * x
void multiply(Matrix &A, Vector &x, Vector& Ax)
{
  int m = A.rows();
  int n = A.cols();
  for (int i = 0; i < m; ++i)
    for(int j = 0; j < n; ++j)
      Ax[i] += A(i, j) * x[j];
  return;
}


// caculate AAt = A * A' for sparse matrix
void multiply(SpMat &A, SpMat &AAt)
{
  AAt = (A * A.transpose()).pruned();

}

    
// calculate A * A' for dense matrix
void multiply(Matrix &A, Matrix &AAt)
{
  int i, j, k;
  int m = A.rows(), n = A.cols();

#pragma omp parallel private(i, j, k)
  {
#pragma omp for schedule (static)
    for (i = 0; i < m; ++i)
      for (j = 0; j < m; ++j)
        for(k = 0; k < n; ++k)
          AAt(i, j) += A(i, k) * A(j, k);
  }
  return;
}


// B = A(start:end, :)
void copy(SpMat& A, SpMat& B, int start, int end)
{
  int n = A.cols();
  int j=0;
  B = A.block(start, 0, end - start, n);

  return;
}


// B = A(start:end, :)
void copy(Matrix& A, Matrix& B, int start, int end)
{
  int n = A.cols();
  B.resize(end - start, n);
  int j=0;
  for(int i=start;i<end;i++)
    for(j=0;j<n;j++)
      B(i-start,j) = A(i,j);

  return;
}


// y = x(start:end)
void copy(Vector& x, Vector& y, int start, int end)
{
  for(int i=start;i<end;i++)
    y[i-start] = x[i];
  return;
}
