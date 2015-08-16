/***********************************************************************
 * Test ARock for solving l1 or l2 regularized logistic regression
 *
 * For l1 regularized, we solve the following problem
 *
 *    min lambda |x|_1 + 1/m log(1 + exp(-b_i* a_i * x) )
 *
 * For l2 regularized, we solve the following problem
 * 
 *    min lambda/2 |x|_2^2 + 1/m log(1 + exp(-b_i* a_i * x) )
 *
 * Date Created:  02/20/2015
 * Date Modified: 02/24/2015
 *                02/25/2015 (modified the input arguments, no ordering)
 *                02/28/2015 (fixed some bugs, check all of the code)
 *                06/10/2015 (add some comments)
 *                08/11/2015 (fixed the output format)
 * Author:        Zhimin Peng, Yangyang Xu, Ming Yan, Wotao Yin
 * Contact:       zhimin.peng@math.ucla.edu
 **********************************************************************/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>  
#include <stdlib.h>
#include <string>
#include "matrices.h"
#include "algebra.h"
#include "logistic.h"
#include "MarketIO.h"

// display help message when the input is not recognized
void exit_with_help();

// parse the input arguments
void parse_input_argv(Parameters&, int, char**, std::string&, std::string&, int&);

// main function
int main(int argc, char *argv[]) {

  int n                     = 1;
  int n_threads_to_use      = 1;  
  int max_n_threads_by_user = 2;
  
  /*************************
    1. set up arguments
   *************************/
  Parameters para;
  std::string data_file_name;
  std::string label_file_name;
  parse_input_argv(para, argc, argv, data_file_name, label_file_name, max_n_threads_by_user);
  vector<vector<double> > result(max_n_threads_by_user, vector<double>(2));

  /**************************
    2. load data from file
   **************************/
  // TODO: need a smart way to combine the sparse, and dense case together
  // current, we have lots of duplicated code
  if (para.is_sparse) {
    SpMat A;
    Vector b;
    loadMarket(A, data_file_name);
    loadMarket(b, label_file_name);
    n = A.rows();
    // check the size of the data if match.
    if (b.size() != A.cols()) {
      std::cout<<"The size of A and b don't match!"<<std::endl;
      return 0;
    }
    Vector x(n, 0.);
    
    // start with asyn ls
    std::cout << "% start parallel ayn to solve "
              << para.type
              << " logistic regression!"
              << std::endl;
    
    /****************************
      3. start the parallel test
     ****************************/
    int num_features = A.rows();
    int num_samples = A.cols();

    std::cout << "---------------------------------------------\n";
    std::cout << "The problem has " << num_samples
              << " samples, " << num_features << " features.\n";
    std::cout << "The data matrix is sparse, "
              << "lambda is: " << para.lambda << ".\n";
    
    for (n_threads_to_use = 1; n_threads_to_use <= max_n_threads_by_user; n_threads_to_use = n_threads_to_use * 2) {
      Vector x(n, 0.);
      Vector Atx(num_samples, 0.);
      double start = omp_get_wtime();
# pragma omp parallel num_threads (n_threads_to_use) shared(A, b, x, Atx, para)
      {
        if (para.type == "l2") {
          l2_logistic(A, b, x, Atx, para);
        }
        else if (para.type == "l1") {
          l1_logistic(A, b, x, Atx, para);
        }
      }
      double end = omp_get_wtime();
      
      result[n_threads_to_use - 1][0] = end - start;
      if (para.type == "l2") {
        result[n_threads_to_use - 1][1] = l2_objective(A, b, x, Atx, para);
      }
      if (para.type == "l1") {
        result[n_threads_to_use - 1][1] = l1_objective(A, b, x, Atx, para);
      }
    }
  }  else {

    Matrix A;
    Vector b;
    loadMarket(A, data_file_name);
    loadMarket(b, label_file_name);
    
    n = A.rows();
    // check the size of the data if match.
    if ( b.size() != A.cols() ) {
      std::cout<<"The size of A and b don't match!"<<std::endl;
      return 0;
    }
    
    Vector x ( n, 0. );
    
    //----------------------------------------------------
    // start with asyn ls
    std::cout<<"% start parallel ayn to solve "<<para.type<<" logistic regression!"<<std::endl;
    
    
    /*
      ===========================
      3. start the parallel test
      ===========================
    */
    int num_features = A.rows();
    int num_samples = A.cols();


    std::cout<<"---------------------------------------------"<<std::endl;
    std::cout<<"The problem has "<<num_samples<<" samples, "<<num_features<<" features."<<std::endl;
    std::cout<<"The data matrix is dense, "<<"lambda is: "<<para.lambda<<"."<<std::endl;

    for ( n_threads_to_use = 1; n_threads_to_use <= max_n_threads_by_user; n_threads_to_use = n_threads_to_use * 2 ) {
      Vector x ( n, 0. );
      Vector Atx ( num_samples, 0. );
      double start = omp_get_wtime();
# pragma omp parallel num_threads ( n_threads_to_use ) shared ( A, b, x, Atx, para )
      {
        if ( para.type == "l2" ) {
          l2_logistic ( A, b, x, Atx, para );
        }
        else if ( para.type == "l1" ) {
          l1_logistic ( A, b, x, Atx, para );
        }
      }
      double end = omp_get_wtime();
      // std::cout<<"% time: " << end - start << " sec."<<std::endl;
      result[n_threads_to_use-1][0] = end - start;
      
      if ( para.type == "l2" ) {
        result[n_threads_to_use-1][1] = l2_objective ( A, b, x, Atx, para );
      }
      if ( para.type == "l1" ) {
        result[n_threads_to_use-1][1] = l1_objective ( A, b, x, Atx, para );
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


// function for printing help message;
void exit_with_help()
{
  std::cout<< "The usage for logistic regression is: \n \
            ./logistic [options] \n \
              -type      <regularization type, can be l1 or l2, default l2>\n \
              -lambda    <regularization paramter, default 1> \n \
              -is_sparse <if the data format is sparse or not. default 0> \n \
              -data      <the file name for the data file, matrix format features x samples>\n \
              -label     <the file name for the labels.> \n \
              -nthread   <the total number of threads, default is set to 2.> \n \
              -epoch     <the total number of epoch, default is set to 10> \n \
              -nthread   <the total number of threads, default is total number of threads in the system.> \n \
              -flag      <the flag for output default 0.>" <<std::endl;
  abort();
}

void parse_input_argv ( Parameters& para,
                      int argc,
                      char *argv[],
                      std::string& data_file_name,
                      std::string& label_file_name,
                      int& max_n_threads_by_user ) {
  for ( int i = 1; i < argc; ++i ) {
    if ( argv[i][0] != '-' ) {
      break;
    }
    if ( ++i >= argc ) {
      exit_with_help();
    }
    if ( std::string ( argv[i-1] ) == "-type" ) {
      para.type = argv[i];
    }
    else if ( std::string ( argv[i-1] ) == "-lambda" ) {
      para.lambda = atof ( argv[i] );
    }
    else if ( std::string ( argv[i-1] ) == "-is_sparse" ) {
      para.is_sparse = bool (atoi ( argv[i] ) );
    }
    else if ( std::string ( argv[i-1] ) == "-epoch" ) {
      para.MAX_EPOCH = atoi ( argv[i] );
    }
    else if ( std::string ( argv[i-1] ) == "-data" ){
      data_file_name = std::string ( argv[i] );
    }
    else if ( std::string ( argv[i-1] ) == "-label" ) {
      label_file_name = std::string ( argv[i] );
    }
    else if ( std::string ( argv[i-1] ) == "-nthread" ) {
      max_n_threads_by_user = atoi ( argv[i] ) ;
    }
    else if ( std::string ( argv[i-1] ) == "-flag" ) {
      para.flag = bool(atoi ( argv[i] ));
    }
    else {
      exit_with_help();
    }
  }
  return;
}
