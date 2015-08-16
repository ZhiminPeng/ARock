/***********************************************************************
 * Test the ARock solver for solving l1 or l2 regularized least square
 * problem.
 *
 * For l1 regularized, we solve the following problem 
 *
 *    min lambda * |x|_1 + 0.5 * |A * x - b| ^ 2
 *
 * For l2 regularized, we solve the following problem 
 *
 *    min lambda/2 * |x|_2^2 + 0.5 * |A * x - b| ^ 2
 *
 * Date Created:  02/20/2015
 * Date Modified: 02/24/2015
 *                02/25/2015 (modified the input arguments, no ordering)
 *                07/21/2015 (organize the style of the code)
 *
 * Author:        Zhimin Peng, Yangyang Xu, Ming Yan, Wotao Yin
 * Contact:       zhimin.peng@math.ucla.edu
 ***********************************************************************/

#include <iostream>
#include <iomanip> 
#include <fstream>
#include <cmath>
#include <omp.h>
#include <stdexcept>      // std::invalid_argument
#include <stdlib.h>
#include <string>
#include "matrices.h"      // matric class header file
#include "algebra.h"       // linear algebra header file
#include "least_square.h"  // least square solver header file
#include "MarketIO.h"      // file IO header file

// display help message when the input is not recognized
void exit_with_help();

// parse the input arguments
void parse_input_argv(Parameters&, int, char**, std::string&, std::string&, int&);

// main function
int main(int argc, char *argv[]) {
  int n                     = 1;
  int n_threads_to_use      = 1;
  int max_n_threads_by_user = 2;

  /*****************************
     1. set up arguments
  *****************************/
  Parameters para;
  std::string data_file_name;
  std::string label_file_name;
  // parse the input argument, and update para, max_n_threads_by_user and data file_names
  parse_input_argv(para, argc, argv, data_file_name, label_file_name, max_n_threads_by_user);
  
  // 2D vector to store the results
  vector< vector<double> > result(max_n_threads_by_user, vector<double>(2));
  
  /*****************************
     2. load data from file
  *****************************/
  std::cout << "% Start to load data!" << std::endl;
  if (para.is_sparse) {
    /******************************************************************
     * Except for the types of A and b and some screen output,
     * this part is the same for the sparse and dense types of A and b
     * TODO: A better to unify the two different cases, instead of
     * making two copies of code.
     *****************************************************************/
    SpMat A;
    Vector b;
    
    loadMarket(A, data_file_name);
    loadMarket(b, label_file_name);
    n = A.rows();
    // check if the size of the data match.
    if (b.size() != A.cols()) {
      std::cout << "The size of A and b don't match!" << std::endl;
      return 0;
    }
    Vector x(n, 0.);
    
    
    /*****************************
      3. start the parallel test
    *****************************/
    
    int num_features = A.rows();
    int num_samples = A.cols();
    Vector Ab(num_features, 0.);
    multiply(A, b, Ab);
    
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "The problem has " << num_samples << " samples, " << num_features << " features." << std::endl;
    std::cout << "The data matrix is sparse, " << "lambda is: " << para.lambda << "." << std::endl;
    std::cout << "---------------------------------------------" << std::endl;

    // start ARock to solve least square problem
    std::cout << "% start ARock to solve: " << std::endl;
    std::cout << "    " << para.type << " regularized least square problem!" << std::endl;

    // we are going to use 1, 2, 4, ...... numbers of threads.    
    for (n_threads_to_use = 1; n_threads_to_use <= max_n_threads_by_user; n_threads_to_use *= 2) {
      Vector x(n, 0.);
      Vector Atx(num_samples, 0.);
      double start_time = omp_get_wtime();
# pragma omp parallel num_threads(n_threads_to_use) shared(A, b, x, Atx, para)
      {
        if (para.type == "l2") {
          l2_ls(A, b, x, Atx, Ab, para);
        } else if (para.type == "l1") {
          l1_ls(A, b, x, Atx, Ab, para);
        } else {
          throw std::invalid_argument("Error: Unknown regularization type!\n");
        }
      }
      double end_time = omp_get_wtime();
      // use result[..][0] for the amount of time,
      // and result[..][1] for final objective value      
      result[n_threads_to_use-1][0] = end_time - start_time; 
      if (para.type == "l2") {
        result[n_threads_to_use - 1][1] = l2_objective(A, b, x, Atx, para);
      } else if (para.type == "l1") {
        result[n_threads_to_use - 1][1] = l1_objective(A, b, x, Atx, para);
      } else {
        throw std::invalid_argument("Error: Unknown regularization type!\n");
      }
    }
  } else {
    // dense data 
    Matrix A;
    Vector b;
    loadMarket(A, data_file_name);
    loadMarket(b, label_file_name);
    n = A.rows();
    // check if the size of the data match.
    if (b.size() != A.cols()) {
      std::cout << "The size of A and b don't match!" << std::endl;
      return 0;
    }
    Vector x(n, 0.);

    // start ARock to solve least square problem
    std::cout << "% start ARock to solve: " << std::endl;
    std::cout << "    " << para.type << " regularized least square problem!" << std::endl;
    
    
    /*****************************
      3. start the parallel test 
    *****************************/
    int num_features = A.rows();
    int num_samples = A.cols();
    Vector Ab(num_features, 0.);
    multiply(A, b, Ab);
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "The problem has " << num_samples << " samples, " << num_features << " features." << std::endl;
    std::cout << "The data matrix is dense, " << "lambda is: " << para.lambda << "." << std::endl;

    // we are going to use 1, 2, 4, ...... numbers of threads.
    for (n_threads_to_use = 1; n_threads_to_use <= max_n_threads_by_user; n_threads_to_use *= 2) {
      Vector x(n, 0.);
      Vector Atx(num_samples, 0.);
      double start_time = omp_get_wtime();
# pragma omp parallel num_threads(n_threads_to_use) shared(A, b, x, Atx,  para)
      {
        if (para.type == "l2") {
          l2_ls(A, b, x, Atx, Ab, para);
        } else if (para.type == "l1") {
          l1_ls(A, b, x, Atx, Ab, para);
        } else {
          throw std::invalid_argument("Error: Unknown regularization type!\n");
        }
      }
      double end_time = omp_get_wtime();
      result[n_threads_to_use-1][0] = end_time - start_time;
      if (para.type == "l2") {
        result[n_threads_to_use - 1][1] = l2_objective(A, b, x, Atx, para);
      } else if (para.type == "l1") {
        result[n_threads_to_use - 1][1] = l1_objective(A, b, x, Atx, para);
      } else {
        throw std::invalid_argument("Error: Unknown regularization type!\n");
      }
    }
  }
  std::cout << "---------------------------------------------" << std::endl;
  std::cout << setw(15) << "# cores used";
  std::cout << setw(15) << "time(s)";
  std::cout << setw(15) << "final obj";
  std::cout << std::endl;
  for (int i = 1; i <= max_n_threads_by_user; i *= 2) {
    std::cout << setw(15) << setprecision(2) << i;
    std::cout << setw(15) << setprecision(2) << scientific << result[i - 1][0];
    std::cout << setw(15) << setprecision(2) << result[i - 1][1];
    std::cout << std::endl;
  }
  std::cout << "---------------------------------------------" << std::endl;
  return 0;
}


void exit_with_help() {
    std::cout << "The usage for least_square is: \n \
    ./ least_square [options] \n                                      \
    -type      <set type for solver, can be l1 or l2, default l2.> \n \
    -lambda    <regularization parameter, default 1.> \n              \
    -is_sparse <if the data format is sparse or not. default 1.> \n     \
    -data      <the file name for the data file, matrix format features x samples.> \n \
    -label     <the file name for the labels.> \n                       \
    -nthread   <the total number of threads, default is set to 2.> \n   \
    -epoch     <the total number of epoch, default is set to 10.> \n    \
    -flag      <the flag for output default 0.>"
              << std::endl;
    abort();
}

void parse_input_argv(Parameters& para,
                      int argc,
                      char *argv[],
                      std::string& data_file_name,
                      std::string& label_file_name,
                      int& max_n_threads_by_user) {
  for (int i = 1; i < argc; ++i) {
    if (argv[i][0] != '-') {
      break;
    }
    if (++i >= argc) {
      exit_with_help();
    }
    if (std::string(argv[i - 1]) == "-type") {
      para.type = argv[i];
    }
    else if (std::string(argv[i - 1]) == "-lambda") {
      para.lambda = atof(argv[i]);
    }
    else if (std::string(argv[i - 1]) == "-step_size") {
      para.step_size = atof(argv[i]);
    }
    else if (std::string(argv[i - 1]) == "-is_sparse") {
      para.is_sparse = bool(atoi(argv[i]));
    }
    else if (std::string(argv[i - 1]) == "-epoch") {
      para.MAX_EPOCH = atoi(argv[i]);
    }
    else if (std::string(argv[i - 1]) == "-data") {
      data_file_name = std::string(argv[i]);
    }
    else if (std::string(argv[i - 1]) == "-label") {
      label_file_name = std::string(argv[i]);
    }
    else if (std::string(argv[i - 1]) == "-nthread") {
      max_n_threads_by_user = atoi(argv[i]);
    }
    else if (std::string(argv[i - 1]) == "-flag") {
      para.flag = bool(atoi(argv[i]));
    }
    else
      exit_with_help();
  }
  return;
}
