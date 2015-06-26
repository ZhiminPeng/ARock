/***********************************************************************
 *
 * Solve l1 regularized least square problem with asynchronous parallel
 * stochastic coordinate forward backward splitting method
 *
 *    min mu/2 |x|_2^2 + 1/m log(1 + exp(-b_i* a_i * x) )
 *
 * Date Created:  02/20/2015
 * Date Modified: 02/24/2015
 *             02/25/2015 (modified the input arguments, no ordering)
 *
 * Author:        Zhimin Peng, Yangyang Xu, Ming Yan, Wotao Yin
 * Contact:       zhimin.peng@math.ucla.edu
***********************************************************************/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>  
#include <stdlib.h>
#include <string>
#include "matrices.h"
#include "algebra.h"
#include "least_square.h"
#include "MarketIO.h"

// display help message when the input is not recognized
void exit_with_help();

// parse the input arguments
void parse_input_argv(Parameters&, int, char**, std::string&, std::string&, int& );

// main function
int main(int argc, char *argv[])
{

  int n                 = 1;
  int thread_count      = 1;
  int total_num_threads = 2;

  /****************************
       1. set up arguments
  *****************************/
  Parameters para;
  std::string data_file_name;
  std::string label_file_name;  
  parse_input_argv(para, argc, argv, data_file_name, label_file_name, total_num_threads);
  // 2D vector to store the results
  vector< vector<double> > result(total_num_threads, vector<double>(2));

  /****************************
      2. load data from file
  *****************************/
  if(para.is_sparse)
  {
    SpMat A;
    Vector b;

    loadMarket(A, data_file_name);
    loadMarket(b, label_file_name);
    n = A.rows();
    // check the size of the data if match.
    if(b.size()!=A.cols())
    {
      cout<<"The size of A and b don't match!"<<endl;
      return 0;
    }
    Vector x(n, 0.);

    //----------------------------------------------------
    // start with asyn ls
    cout<<"% start ARock to solve "<<para.type<<" regularized least square!"<<endl;
    
    
    /****************************
      3. start the parallel test
    *****************************/
    
    int num_features = A.rows();
    int num_samples = A.cols();

    Vector Ab(num_features, 0.);
    multiply(A, b, Ab);
    
    cout<<"---------------------------------------------"<<endl;
    cout<<"The problem has "<<num_samples<<" samples, " << num_features<<" features."<<endl;
    cout<<"The data matrix is sparse, " <<"lambda is: " << para.lambda<<"."<<endl;
    
    for(thread_count = 1; thread_count<=total_num_threads; thread_count = thread_count*2)
    {
      Vector x(n, 0.);
      Vector Atx(num_samples, 0.);
      double start = omp_get_wtime();
# pragma omp parallel num_threads(thread_count) shared(A, b, x, Atx, para)
      {
        if(para.type=="l2")
          l2_ls(A, b, x, Atx, Ab, para);
        else if(para.type=="l1")
          l1_ls(A, b, x, Atx, Ab, para);
      }
      double end = omp_get_wtime();
      result[thread_count-1][0] = end - start;
      if(para.type=="l2")
      {
        result[thread_count-1][1] = l2_objective(A, b, x, Atx, para);
      }
      if(para.type=="l1")
      {
        result[thread_count-1][1] = l1_objective(A, b, x, Atx, para);
      }
    }
  }

  if(!para.is_sparse)
  {
    Matrix A;
    Vector b;
    loadMarket(A, data_file_name);
    loadMarket(b, label_file_name);

    n = A.rows();
    // check the size of the data if match.
    if(b.size()!=A.cols())
    {
      cout<<"The size of A and b don't match!"<<endl;
      return 0;
    }
    
    Vector x(n, 0.);
    
    //----------------------------------------------------
    // start with asyn ls
    cout<<"% start parallel ayn to solve "<<para.type<<" logistic regression!"<<endl;
    
    
    /*
      ===========================
      3. start the parallel test
      ===========================
    */
    int num_features = A.rows();
    int num_samples = A.cols();

    Vector Ab(num_features, 0.);
    multiply(A, b, Ab);
    cout<<"---------------------------------------------"<<endl;
    cout<<"The problem has "<<num_samples<<" samples, " << num_features<<" features."<<endl;
    cout<<"The data matrix is dense, " <<"lambda is: " << para.lambda<<"."<<endl;

    for(thread_count = 1; thread_count<=total_num_threads; thread_count=thread_count*2)
    {
      Vector x(n, 0.);
      Vector Atx(num_samples, 0.);
      double start = omp_get_wtime();
# pragma omp parallel num_threads(thread_count) shared(A, b, x, Atx,  para)
      {
        if(para.type=="l2")
          l2_ls(A, b, x, Atx, Ab,  para);
        else if(para.type=="l1")
          l1_ls(A, b, x, Atx, Ab, para);
      }
      double end = omp_get_wtime();
      // cout<<"% time: " << end - start << " sec."<<endl;
      result[thread_count-1][0] = end - start;

      if(para.type=="l2")
      {
        result[thread_count-1][1] = l2_objective(A, b, x, Atx, para);
      }
      if(para.type=="l1")
      {
        result[thread_count-1][1] = l1_objective(A, b, x, Atx, para);
      }
    }
  }
  cout<<"---------------------------------------------"<<endl;
  cout<<setw(15)<<"# cores";
  cout<<setw(15)<<"time(s)";
  cout<<setw(15)<<"objective";
  cout<<endl;
  for(int i=0;i<total_num_threads;i++)
  {
    cout<<setw(15)<<setprecision(2)<<i+1;
    cout<<setw(15)<<setprecision(2)<<scientific<<result[i][0];
    cout<<setw(15)<<setprecision(2)<<result[i][1];
    cout<<endl;
  }
  cout<<"---------------------------------------------"<<endl;
  
  return 0;
}


void exit_with_help()
{
  std::cout<<"The usage for least_square is: \n \
            ./least_square [options] \n \
              -type      <set type for solver, can be l1 or l2, default l2>\n \
              -lambda    <regularization parameter, default 1> \n \
              -is_sparse <if the data format is sparse or not. default 1> \n \
              -data      <the file name for the data file, matrix format features x samples>\n \
              -label     <the file name for the labels.> \n \
              -nthread   <the total number of threads, default is set to 2.> \n \
              -epoch     <the total number of epoch, default is set to 10> \n \
              -flag      <the flag for output default 0.>"
           <<std::endl;
  abort();
}

void parse_input_argv(Parameters& para,
                      int argc,
                      char *argv[],
                      std::string& data_file_name,
                      std::string& label_file_name,
                      int& total_num_threads)
{
  for (int i = 1; i < argc; ++i)
  {
    if(argv[i][0]!='-') break;
    if(++i>=argc)
      exit_with_help();
    if(std::string(argv[i-1])== "-type")
      para.type = argv[i];
    else if(std::string(argv[i-1])== "-lambda")
      para.lambda = atof(argv[i]);
    else if(std::string(argv[i-1])== "-step_size")
      para.step_size = atof(argv[i]);
    else if(std::string(argv[i-1])== "-is_sparse")
      para.is_sparse = atoi(argv[i]);
    else if(std::string(argv[i-1])== "-epoch")
      para.MAX_EPOCH = atoi(argv[i]);
    else if(std::string(argv[i-1])== "-data")
      data_file_name = std::string(argv[i]);
    else if(std::string(argv[i-1])== "-label")
      label_file_name = std::string(argv[i]);
    else if(std::string(argv[i-1])== "-nthread")
      total_num_threads = atoi(argv[i]);
    else if(std::string(argv[i-1])== "-flag")
      para.flag = atoi(argv[i]);
    else
      exit_with_help();
  }
  return;
}
