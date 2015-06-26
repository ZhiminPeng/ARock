#include <iostream>
#include <iomanip>
#include <fstream>
extern "C"
{
#include <omp.h>  
#include <stdlib.h>
  // #include <time.h>
}
#include <cmath>
#include "matrices.h"
#include "algebra.h"
#include "jacobi.h"
#include <string>
void exit_with_help()
{
  std::cout<<"The usage for jacobi solver is: \n \
            ./jacobi [options] \n \
              -data      <the file name for the data file, data should be a square matrix>\n \
              -label     <the file name for the labels.> \n \
              -is_sparse <if the data format is sparse or not. default 1> \n \
              -nthread   <the total number of threads, default is set to 2.> \n \
              -epoch     <the total number of epoch, default is set to 10> \n \
              -flag      <the flag for output default 0.>"<<std::endl;
  abort();
}


void parse_input_argv(Parameters& para,
                      int argc,
                      char *argv[],
                      std::string& data_file_name,
                      std::string& label_file_name,
                      int& total_num_threads,
                      int& n,
                      int& num_diag)
{
  for (int i = 1; i < argc; ++i)
  {
    if(argv[i][0]!='-') break;
    if(++i>=argc)
      exit_with_help();
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
    else if(std::string(argv[i-1])== "-n")
      n = atoi(argv[i]);
    else if(std::string(argv[i-1])== "-num_diag")
      num_diag = atoi(argv[i]);
    else if(std::string(argv[i-1])== "-flag")
      para.flag = atoi(argv[i]);
    else
      exit_with_help();
  }
  return;
}

typedef Eigen::Triplet<double> T;

void buildProblem(vector<T>& coefficients, int n, int num_diag)
{
  coefficients.reserve(num_diag * n);
  int abs_i = 0;
  int count = 0;
  int left = -num_diag/2;
  int right = num_diag/2;
  for (int i = left; i <= right; ++i)
  {
    abs_i = abs(i);
    for (int j = 0; j < n-abs_i; ++j)
    {
      if(i<0) coefficients.push_back(T(j+abs_i, j, pow(0.5, abs_i -1.)));
      if(i==0) coefficients.push_back(T(j, j, 4.));
      if(i>0) coefficients.push_back(T(j, j+abs_i, pow(0.5, abs_i-1.)));      
    }
   
  }
  return;
}

// main function
int main(int argc, char *argv[])
{
  unsigned seed = 1;
  bool flag = false;
  int n = 5;
  int num_diag = 5;
  Parameters para;
  int total_num_threads = 2;
  string label_file_name = "";
  string data_file_name = "";
  parse_input_argv(para, argc, argv, data_file_name, label_file_name, total_num_threads, n, num_diag);
  vector< vector<double> > result(total_num_threads, vector<double>(2));

  vector<T> coefficients;
  buildProblem(coefficients, n, num_diag);
  SpMat A(n, n);
  A.setFromTriplets(coefficients.begin(), coefficients.end());

  Vector b(n, 1.);


  /*==============================
        create the problem
    ==============================*/

  int thread_count = 0;

  for(thread_count = 1; thread_count<=total_num_threads; ++thread_count)
  {
    Vector x(n, 0.);
    para.counter = 0;
    double start = omp_get_wtime();
# pragma omp parallel num_threads(thread_count) shared(A, b, x, para)
    {
      // balanced_jacobi(A, b, x, para);
      new_jacobi(A, b, x, para);
    }
    double end = omp_get_wtime();
    result[thread_count-1][0] = end - start;
    result[thread_count-1][1] = calculate_residual(A, b, x);
  }


  cout<<"---------------------------------------------"<<endl;
  cout<<setw(15)<<"# cores";
  cout<<setw(15)<<"time(s)";
  cout<<setw(15)<<"||Ax -b||";
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
