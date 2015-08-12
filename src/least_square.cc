/*************************************************************************************
 * ARock for solving l1 or l2 regularized least square problem.
 * Date Created:  02/26/2015
 * Date Modified: 02/28/2015 (added lots of comments, moved shrink function to algebra.h)
 *                03/17/2015 (shared write after updating several coordinates)
 *                06/09/2015 (added the code to ARock style, added more comments)
 * Author:        Zhimin Peng, Yangyang Xu, Ming Yan, Wotao Yin
 * Contact:       zhimin.peng@math.ucla.edu
 *
 *************************************************************************************/

#include "algebra.h"
#include "matrices.h"
#include "least_square.h"

/**********************************************************************
 *
 * Calculates objective value for l2 regularized least square
 *
 * Input:
 *     A:      the data matrix with size num_features x num_samples
 *     (type T, it can be sparse matrix SpMat or Matrix)
 *    Ab:      the observation labels for each sample
 *     (Vector)
 *     x:      the unknowns
 *     (Vector, size is the number of features)
 *     Atx:    A' * x, which is stored in shared memory for efficient
 *             computation
 *     lambda: regularization parameter
 *
 * Output: objective value
 *     (double)
 *
 **********************************************************************/

template <typename T>
double l2_objective(T& A,
                    Vector& b,
                    Vector& x,
                    Vector& Atx,
                    Parameters &para) {
    double lambda = para.lambda;
    Vector grad_loss = Atx;
    sub(grad_loss, b);   // Atx - b
    double nrm_grad_loss = norm(grad_loss, 2);
    double nrm_x = norm(x, 2);
    return 0.5 * lambda * nrm_x * nrm_x + 0.5 * nrm_grad_loss * nrm_grad_loss;
}

template double l2_objective<Eigen::SparseMatrix<double, 1, int> >(Eigen::SparseMatrix<double, 1, int>&,
                                                                   Vector&,
                                                                   Vector&,
                                                                   Vector& ,
                                                                   Parameters&);

template double l2_objective<Matrix>(Matrix&,
                                     Vector&,
                                     Vector&,
                                     Vector&,
                                     Parameters&);

/**********************************************************************
 *
 * Calculates the objective value for l1 regularized least square.
 * Input:
 *     A:      the data matrix with size num_features * num_samples;
 *     (type T, it can be sparse matrix (SpMat) or dense matrix (Matrix))
 *     b:      the observation labels;
 *     (Vector)
 *     x:      the unknowns, weights for different features;
 *     (Vector, size is the number of features)
 *     Atx:    A' * x, which is stored in shared memory for efficient
 *             computation
 *     lambda: regularization parameter
 *
 * Output: objective value
 *    (double)
 *
 *********************************************************************/

template <typename T>
double l1_objective(T&          A,
                    Vector&     b,
                    Vector&     x,
                    Vector&     Atx,
                    Parameters& para) {
    double lambda = para.lambda;
    Vector grad_loss = Atx;
    sub(grad_loss, b);  // calculates the gradient of the loss function Atx - b
    double nrm_grad_loss = norm(grad_loss, 2);
    double nrm_x = norm(x, 1);
    return lambda * nrm_x + 0.5 * nrm_grad_loss * nrm_grad_loss;
}

template double l1_objective<Eigen::SparseMatrix<double, 1, int> >(Eigen::SparseMatrix<double, 1, int>&,
                                                                   Vector&,
                                                                   Vector&,
                                                                   Vector&,
                                                                   Parameters&);

template double l1_objective<Matrix>(Matrix&,
                                     Vector&,
                                     Vector&,
                                     Vector&,
                                     Parameters&);

/************************************************************************
 * Solve the l2 regularized least square problem with ARock.
 *
 * Input:
 *     A: data matrix with size num_features * num_samples.
 *        we store data in this way to calculate dot product more efficient
 *     b: label (the label for the corresponding observation)
 *     (Vector)
 *     x: the unknown variables. Weights for different features.
 *     (Vector)
 *     lambda: regularization parameter ( >= 0)
 *     (double)
 *     Atx: temporary variable in shared memory for storing A' * x
 *     (Vector)
 *     Ab: temporary variable in shared memory for storing A * b
 *     (Vector)
 *     para: parameters.
 *     (struct)
 *
 * Output:
 *     (none)
 *
 ************************************************************************/

template <typename T>
void l2_ls(T&          A,
           Vector&     b,
           Vector&     x,
           Vector&     Atx,
           Vector&     Ab,
           Parameters& para) {
    int num_features = A.rows();                    // total number of features
    int num_samples  = A.cols();                    // total number of samples
    int num_threads  = omp_get_num_threads();       // total number of threads
    int my_rank      = omp_get_thread_num();        // rank of current thread
    int MAX_ITER     = para.MAX_EPOCH;              // maximum number of epochs
    double lambda    = para.lambda;                 // regularization parameter
    bool flag        = para.flag;                   // output flag, if 1, output.
    double STEP_SIZE = para.step_size;              // step size  // 1. / (lambda + 2.);
    int idx          = 0;                           // randomize index
    double S_i       = 0.;                          // initial S_i
    int local_m      = num_features / num_threads;  // number of updates for each core
    int local_start  = local_m * my_rank;           // starting index for local loop
    int local_end    = local_m * (my_rank + 1);     // ending index for local loop
    double Ai_Atx    = 0.;                          // initial value for A(i, :) * Atx
    SpVec local_dAtx(num_samples);
    if (my_rank == num_threads - 1) {
        local_end = num_features;
    }
    if (my_rank == 0 && flag) {
        cout << "l2_obj_" << num_threads << "= [ ";
    }
    int block_size = 50;
    int num_blocks = num_features / block_size;
    int i;
    int block = 0;
    int local_num_blocks = num_blocks / num_threads;
    int block_id;
    Vector local_dx(num_features - (num_blocks - 1) * block_size, 0.);
    // main loop; each iteration represents an epoch
    for (int itr = 0; itr < MAX_ITER; itr++) {
        for (block = 0; block < local_num_blocks; block++) {
            // Step 1. generate a random block id
            block_id = rand() % num_blocks;
            // calculate the starting index and ending index for the block
            local_start = block_id * block_size;
            local_end = (block_id + 1) * block_size;
            if (block_id == num_blocks - 1) {
                local_end = num_features;
            }
            // clean the data in local_delta_Atx
            local_dAtx.setZero();
            // local for loop for each thread
            for (i = local_start; i < local_end; i++) {
                idx = i;
                // Step 2. calculate S_i
                Ai_Atx = dot(A, Atx, idx);  // Ai_Atx = A(idx,:) * Atx;
                S_i = lambda * x[idx] + Ai_Atx - Ab[idx];
                local_dx[i - local_start] = STEP_SIZE * S_i;
                sub(local_dAtx, A, idx, STEP_SIZE * S_i);
            }
            // ensure consistent write
            // Step 3. update x
            for (i = local_start; i < local_end; ++i) {
                x[i] -= local_dx[i - local_start];
            }
            // Step 4. update Atx based on the new x
            add(Atx, local_dAtx);
        }
        
        // use thread 0 for output objective value
        if (my_rank == 0 && flag) {
            cout << l2_objective(A, b, x, Atx, para) << endl;
        }
        
    }
    if (my_rank == 0 && flag) {
        cout << "];" << endl;
    }
    return;
}

template void l2_ls<Eigen::SparseMatrix<double, 1, int> >(Eigen::SparseMatrix<double, 1, int>&,
                                                          Vector&,
                                                          Vector&,
                                                          Vector&,
                                                          Vector&,
                                                          Parameters&);

template void l2_ls<Matrix >(Matrix&,
                             Vector&,
                             Vector&,
                             Vector&,
                             Vector&,
                             Parameters&);


/*****************************************************************************
 *
 * Solves the l1 regularized least square problem with ARock
 *
 * Input:
 *     A: data matrix; matrix size is num_features x num_samples
 *          we store data in this way for the purpose of efficiency.
 *     (T can be sparse matrix (SpMat) or dense matrix (Matrix) )
 *     b: label (the label for the corresponding observation, +1 / -1)
 *     (Vector)
 *     x: the unknown variable, weights for different features
 *     (Vector)
 *     lambda: regularization parameter
 *     (double)
 *     Atx: temporary variable in shared memory for storing A' * x
 *     (Vector)
 *     Ab: temporary variable in shared memory for storing Ab
 *     (Vector)
 *     para: related parameters, includes MAX_EPOCH, lambda, flag.
 *     (Parameters, struct)
 *
 * Output:
 *     (none)
 *
 **************************************************************************/
template <typename T>
void l1_ls(T&          A,
           Vector&     b,
           Vector&     x,
           Vector&     Atx,
           Vector&     Ab,
           Parameters& para) {
    int num_features     = A.rows();                     // number of features
    int num_samples      = A.cols();                     // number of samples
    int num_threads      = omp_get_num_threads();        // number of threads
    int my_rank          = omp_get_thread_num();         // thread id
    double STEP_SIZE     = para.step_size;               // step size
    double gamma         = 0.01;                         // forward step size, should be 1 / norm(A(:,idx),2)
    double lambda        = para.lambda;                  // regularization parameter
    bool flag            = para.flag;                    // output flag, if 1, then output
    int MAX_ITER         = para.MAX_EPOCH;               // maximum number of epochs
    int idx              = 0;                            // store random index
    double S_i           = 0.;                           // S_i
    double forward_step  = 0.;                           // store the forward step
    double backward_step = 0.;                           // backward step
    int local_m          = num_features / num_threads;   // local number of iterations
    int local_start      = local_m * my_rank;            // starting index for local for loop
    int local_end        = local_m * (my_rank + 1);      // ending index for local for loop
    double grad_i        = 0.;                           // ith component in the forward gradient
    double x_hat_i       = 0.;                           // ith component of x_hat
    if (my_rank == num_threads - 1) {
        local_end = num_features;                        // offset the last block.
    }
    if (my_rank == 0 && flag) {
        cout << "l1_obj_" << num_threads << "= [ ";
    }
    SpVec local_dAtx(num_samples);
    int block_size = 50;
    int num_blocks = num_features / block_size;
    int i;
    int block = 0;
    int local_num_blocks = num_blocks / num_threads;
    int block_id;
    Vector local_dx(num_features - (num_blocks - 1) * block_size, 0.);
    
    // main loop; each iteration represent an epoch
    for (int itr = 0; itr < MAX_ITER; itr++) {
        for (block = 0; block < local_num_blocks; block++) {
            // Step 1. generate a random block id
            block_id = rand() % num_blocks;
            
            // calculate the starting index and ending index for the block
            local_start = block_id * block_size;
            local_end   = (block_id + 1) * block_size;
            if (block_id == num_blocks - 1) {
                local_end = num_features;
            }
            
            // clean the data in local_delta_Atx
            local_dAtx.setZero();
            
            // for loop for one epoch
            for (i = local_start; i < local_end; i++) {
                
                idx = i;
                
                // Step 2. calculate forward step
                x_hat_i       = x[idx];
                grad_i        = dot(A, Atx, idx);
                grad_i       -= Ab[idx];
                forward_step  = x_hat_i - gamma * grad_i;
                
                // step 3. calculate the backward step
                backward_step = shrink(forward_step, gamma * lambda);
                
                // step 4. calculate S_i
                S_i = x_hat_i - backward_step;
                local_dx[i - local_start] = STEP_SIZE * S_i;
                sub(local_dAtx, A, idx, STEP_SIZE * S_i);
            }
            
            // step 5. update x
            for (i = local_start; i < local_end; ++i) {
                x[i] -= local_dx[i - local_start];
            }
            // step 6. update Atx based on the new x
            add(Atx, local_dAtx);
        }
        
        if (my_rank == 0 && flag && itr % 10 == 0) {
            // output from thread 0.
            cout << l1_objective(A, b, x, Atx, para) << endl;
        }
    }
    
    if (my_rank == 0 && flag) {
        cout << "];" << endl;
    }
    return;
}

template void l1_ls<Eigen::SparseMatrix<double, 1, int> >(Eigen::SparseMatrix<double, 1, int>&, Vector&, Vector&, Vector&, Vector&, Parameters&);

template void l1_ls<Matrix >(Matrix&, Vector&, Vector&, Vector&, Vector&, Parameters&);

// end of file
