/**************************************************************************
 *  ARock for l1 or l2 regularized logistic regression.
 *
 *  1) for l2 regularization, we solve the following problem
 *
 *    min lambda/2 |x|_2^2 + 1/N \sum_i log(1 + exp(b_i * (a_i * x)) )
 *
 *  where N is the total number of samples, and a_i is the ith sample, 
 *  b_i is the label for a_i. And we let A be the data matrix, with size
 *  num_features x num_samples, and let b be the label vector with length
 *  num_samples.
 *
 *  2) for l1 regularization, we solve
 *
 *    min lambda |x|_1 + 1/N \sum_i log(1 + exp(b_i * (a_i * x)) )
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

#include "algebra.h"
#include "matrices.h"
#include "logistic.h"

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
double l2_objective ( T& A, Vector& b, Vector& x, Vector& Atx, Parameters para ) {
  double lambda = para.lambda;
  double tmp = 0.;
    for ( unsigned i = 0; i < b.size(); ++i ) {
      tmp += log ( 1. + exp ( -b[i] * Atx[i] ) );
    }
  double nrm = norm(x, 2);
  return 0.5 * lambda * nrm * nrm + tmp / ( double ) ( b.size() );
}

template double l2_objective <Eigen::SparseMatrix <double, 1, int> > ( Eigen::SparseMatrix <double, 1, int>&, Vector&, Vector&, Vector& , Parameters );

template double l2_objective <Matrix> ( Matrix&, Vector&, Vector&, Vector& , Parameters );


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
double l1_objective ( T& A, Vector& b, Vector& x, Vector& Atx,  Parameters para ) {
  double lambda = para.lambda;
  double tmp = 0.;
    for ( unsigned i = 0; i < b.size(); ++i ) {
    tmp += log ( 1. + exp ( -b[i] * Atx[i] ) );
    }
  double nrm = norm ( x, 1 );
  return lambda * nrm + tmp / ( double ) ( b.size() );
}

template double l1_objective <Eigen::SparseMatrix <double, 1, int> > ( Eigen::SparseMatrix <double, 1, int>&, Vector&, Vector&, Vector& , Parameters );

template double l1_objective <Matrix> ( Matrix&, Vector&, Vector&, Vector& , Parameters );


// calculate the forward gradient
double forward_gradient ( Matrix& A, Vector& b, Vector& Atx, int idx ) {
  double result = 0.;
  for ( unsigned i = 0; i < A.cols(); ++i )
    result += A ( idx, i ) * b[i] / ( 1.+exp ( b[i] * Atx[i] ) );
  return result;
}

double forward_gradient ( SpMat& A, Vector& b, Vector& Atx, int idx ) {

  double result = 0.;
  int i;
  for ( SpMat::InnerIterator it ( A, idx ); it; ++it ) {
    i = it.index();
    result += it.value() * b[i] / ( 1.+exp ( b[i] * Atx[i] ) );
  }
  return result;
}

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
void l2_logistic ( T& A, Vector& b, Vector& x, Vector &Atx, Parameters para ) {
  int num_features = A.rows();
  int num_samples  = A.cols();
  int num_threads  = omp_get_num_threads();
  int my_rank      = omp_get_thread_num();
  int max_iter     = para.MAX_EPOCH;
  double lambda    = para.lambda;
  bool flag        = para.flag;
  double step_size = 1. / ( lambda+1. );
  int idx = 0;
  double S_i = 0.;
  int local_m      = num_features / num_threads;
  int local_start  = local_m * my_rank;
  int local_end    = local_m * ( my_rank+1 );
    if ( my_rank == num_threads - 1 ) {
      local_end = num_features;
    }
    if ( my_rank == 0 && flag ) {
      cout<<"l2_obj_" << num_threads << "= [ ";
    }
  double Ai_Atx = 0.;                        // initial value for A(i, :)*Atx
  //  Vector local_dAtx(num_samples, 0.);
  SpVec local_dAtx ( num_samples );
  int block_size = 50;
  int num_blocks = num_features / block_size;
  int i;
  int block = 0;
  int local_num_blocks = num_blocks / num_threads;
  int block_id;
  Vector local_dx ( num_features - ( num_blocks-1 ) * block_size, 0. );

  // main loop, each iteration represent an epoch
  for ( int itr = 0;itr < max_iter; itr++ ) {
    for ( block = 0;block < local_num_blocks;block++ ) {
      // generate a random block id
      
      block_id = rand() % num_blocks;
      // block_id = block;
      // block_id = 1;
      // calculate the starting index and ending index for the block
      local_start = block_id * block_size;
      local_end = ( block_id + 1 ) * block_size;
        if ( block_id == num_blocks - 1 ) {
        local_end = num_features;
        }

      // clean the data in local_delta_Atx
      // fill(local_dAtx.begin(), local_dAtx.end(), 0.);
      local_dAtx.setZero();

      // for loop for one epoch
      for ( i = local_start; i < local_end; i++ ) {
        S_i = 0.;
        // idx = rand()%num_features; // select a random index
        idx = i;
        
        S_i = forward_gradient ( A, b, Atx, idx );
        S_i = lambda * x[idx] - S_i / ( double ) ( num_samples );
        
        local_dx[i-local_start] = step_size * S_i;
        sub ( local_dAtx, A, idx, step_size*S_i );
      }
    }
    // ensure consistent write
    // #pragma omp critical
    {
      for ( i=local_start; i < local_end; ++i ) {
        x[i] -= local_dx[i-local_start];
      }
      add ( Atx, local_dAtx );
    }
    
    
    if ( my_rank == 0 && flag && itr % 10 == 0 ) {
      cout<< l2_objective ( A, b, x, Atx, para )<<endl;
    }
    
  }
  
    if ( my_rank == 0 && flag ) {
      cout<<"];"<<endl;
    }
  return;
}

template void l2_logistic <Eigen::SparseMatrix <double, 1, int> > (
    Eigen::SparseMatrix <double, 1, int>&,
    Vector&,
    Vector&,
    Vector&,
    Parameters );

template void l2_logistic <Matrix> (
    Matrix&,
    Vector&,
    Vector&,
    Vector&,
    Parameters );


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
void l1_logistic ( T& A, Vector& b, Vector& x, Vector &Atx, Parameters para )
{
  int num_features = A.rows();
  int num_samples  = A.cols();
  int num_threads  = omp_get_num_threads();
  int my_rank      = omp_get_thread_num();
  double step_size = para.step_size;
  double gamma     = 1.;
  double lambda    = para.lambda;
  bool flag        = para.flag;
  int max_iter     = para.MAX_EPOCH;
  int idx          = 0;
  double S_i       = 0.;
  double forward_step = 0.;
  double backward_step = 0.;
  int local_m      = num_features / num_threads;
  int local_start  = local_m * my_rank;
  int local_end    = local_m * ( my_rank + 1 );
    if ( my_rank == num_threads - 1 ) {
      local_end = num_features;
    }
    if ( my_rank == 0 && flag ) {
      cout<<"l1_obj_" << num_threads << "= [ ";
    }
  double x_hat_i;
  SpVec local_dAtx ( num_samples ); // local vector to hold difference of Atx
  int block_size = 50;
  int num_blocks = num_features / block_size;
  int i;
  int block = 0;
  int local_num_blocks = num_blocks / num_threads;
  int block_id;
  Vector local_dx ( num_features - ( num_blocks - 1 ) * block_size, 0. );
  
  // main loop; each iteration represent an epoch
  for ( int itr = 0;itr < max_iter; itr++ ) {
    for ( block = 0;block < local_num_blocks;block++ ) {
      // generate a random block id
      block_id = rand() % num_blocks;
      
      // calculate the starting index and ending index for the block
      local_start = block_id * block_size;
      local_end = ( block_id + 1 ) * block_size;
        if ( block_id == num_blocks - 1 ) {
        local_end = num_features;
        }
      
      // clean the data in local_delta_Atx
      local_dAtx.setZero();       
      // fill(local_dAtx.begin(), local_dAtx.end(), 0.); // use this if dense data is used
      
      for ( i = local_start; i < local_end; i++ ) {
        S_i = 0.;
        // idx = rand()%num_features; // select a random index
        idx = i;
        x_hat_i = x[idx];
        forward_step = forward_gradient ( A, b, Atx, idx );
        forward_step = x_hat_i + gamma * forward_step / num_samples; // x_i - gamma * grad_i
        backward_step = shrink ( forward_step, gamma * lambda );
        S_i = x_hat_i - backward_step;
        local_dx[i-local_start] = step_size * S_i; // store the updated dx in local variable
        sub ( local_dAtx, A, idx, step_size * S_i );  // update the difference of Atx
      }
      
      //# pragma omp critical
      {
        // step 3. update x
        for ( i = local_start; i < local_end; ++i ) {
          x[i] -= local_dx[i - local_start];
        }
        
        // step 4. update Atx based on dAtx
        add ( Atx, local_dAtx ); // add the difference to Atx
      }
      
    }

    if ( my_rank == 0 && flag && itr % 10 == 0 ) {
      cout<< l1_objective ( A, b, x, Atx, para )<<endl;
    }
  }

    if ( my_rank == 0 && flag ) {
      cout<<"];"<<endl;
    }
  return;
}

template void l1_logistic <Eigen::SparseMatrix <double, 1, int> > (
    Eigen::SparseMatrix <double, 1, int>&,
    Vector&,
    Vector&,
    Vector &,
    Parameters );

template void l1_logistic <Matrix> (
    Matrix&,
    Vector&,
    Vector&,
    Vector &,
    Parameters );



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
void syn_l1_logistic ( T& A, Vector& b, Vector& x, Vector &Atx, Parameters para ) {
  int num_features = A.rows();
  int num_samples  = A.cols();
  int num_threads  = omp_get_num_threads();
  int my_rank      = omp_get_thread_num();
  double step_size = para.step_size;
  double gamma     = 1.;
  double lambda    = para.lambda;
  bool flag        = para.flag;
  int max_iter     = para.MAX_EPOCH;
  int idx          = 0;
  double S_i       = 0.;
  double forward_step = 0.;
  double backward_step = 0.;
  int local_m      = num_features / num_threads;
  int local_start  = local_m * my_rank;
  int local_end    = local_m * ( my_rank + 1 );
  if ( my_rank == num_threads - 1 ){
      local_end = num_features;
  }
  if ( my_rank == 0 && flag ){
      cout<<"l1_obj_" << num_threads << "= [ ";
  }
  double x_hat_i;
  SpVec local_dAtx ( num_samples ); // local vector to hold difference of Atx
  int block_size = 50;
  int num_blocks = num_features / block_size;
  int i;
  int block = 0;
  int local_num_blocks = num_blocks / num_threads;
  int block_id;
  Vector local_dx ( num_features - ( num_blocks - 1 ) * block_size, 0. );
  
  // main loop; each iteration represent an epoch
  for ( int itr = 0;itr < max_iter; itr++ ) {
    for ( block = 0;block < local_num_blocks;block++ ) {
      // generate a random block id
      block_id = rand() % num_blocks;
      
      // calculate the starting index and ending index for the block
      local_start = block_id * block_size;
      local_end = (block_id + 1) * block_size;
        if ( block_id == num_blocks - 1 ) {
        local_end = num_features;
        }
      
      // clean the data in local_delta_Atx
      local_dAtx.setZero();       
      // fill(local_dAtx.begin(), local_dAtx.end(), 0.); // use this if dense data is used
      
      for ( i = local_start; i < local_end; i++ ) {
        S_i = 0.;
        // idx = rand()%num_features; // select a random index
        idx = i;
        x_hat_i = x[idx];
        forward_step = forward_gradient ( A, b, Atx, idx );
        forward_step = x_hat_i + gamma * forward_step / num_samples; // x_i - gamma * grad_i
        backward_step = shrink ( forward_step, gamma * lambda );
        S_i = x_hat_i - backward_step;
        local_dx[i-local_start] = step_size * S_i; // store the updated dx in local variable
        sub ( local_dAtx, A, idx, step_size * S_i );  // update the difference of Atx
      }
      
      //# pragma omp critical
#pragma omp barrier
{
        // step 3. update x
        for ( i = local_start; i < local_end; ++i ) {
          x[i] -= local_dx[i - local_start];
        }

        // step 4. update Atx based on dAtx
#pragma omp critical
        add ( Atx, local_dAtx ); // add the difference to Atx
      }
      
    }
#pragma omp barrier

    if ( my_rank == 0 && flag && itr % 10 == 0 ) {
      cout<< l1_objective ( A, b, x, Atx, para )<<endl;
    }
  }

    if ( my_rank == 0 && flag ) {
      cout<<"];"<<endl;
    }
  return;
}

template void syn_l1_logistic <Eigen::SparseMatrix <double, 1, int> > (
    Eigen::SparseMatrix <double, 1, int>&,
    Vector&,
    Vector&,
    Vector&,
    Parameters );

template void syn_l1_logistic <Matrix> (
    Matrix&,
    Vector&,
    Vector&,
    Vector&,
    Parameters );


// end of the file
