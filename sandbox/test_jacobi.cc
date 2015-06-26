#include <iostream>
#include <iomanip>
#include <fstream>
extern "C"
{
#include <omp.h>  
#include <stdlib.h>
  // #include <time.h>
}

#include "../util/matrices.h"
#include "../util/algebra.h"
#include "../src/jacobi.cc"
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
                      int& total_num_threads)
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
    else if(std::string(argv[i-1])== "-flag")
      para.flag = atoi(argv[i]);
    else
      exit_with_help();
  }
  return;
}



// main function
int main(int argc, char *argv[])
{
  unsigned seed = 1;
  bool flag = false;
  int n;

  std::string data_file_name;
  std::string label_file_name;  
  Parameters para;
  int total_num_threads = 2;
  parse_input_argv(para, argc, argv, data_file_name, label_file_name, total_num_threads);
  vector< vector<double> > result(total_num_threads, vector<double>(2));
  
  /*==============================
        create the problem
    ==============================*/
  if(para.is_sparse)
  {
    SpMat A;
    Vector b;
    loadMarket(A, data_file_name);
    loadMarket(b, label_file_name);
    n = A.rows();

    if(b.size()!=A.cols())
    {
      cout<<"The size of A and b don't match!"<<endl;
      return 0;
    }
    cout<<"% start parallel ayn to solve linear equation"<<endl;
    cout<<"---------------------------------------------"<<endl;
    cout<<"The size of the problem is " <<n<<endl;

    /*==============================
      start the parallel solver
      ==============================*/
    int thread_count = 1;
    /*
      cout<<"old method:"<<endl;
      vector< vector<double> > old_result(total_num_threads, vector<double>(2));
      for(thread_count = 1; thread_count<=total_num_threads; ++thread_count)
      {
      Vector x(n, 0.);
      double start = omp_get_wtime();
      # pragma omp parallel num_threads(thread_count) shared(A, b, x, flag)
      {
      old_jacobi(A, b, x, flag);
      }
      double end = omp_get_wtime();
      old_result[thread_count-1][0] = end - start;
      old_result[thread_count-1][1] = calculate_residual(A, b, x);
      
      // cout<<"% time: " << end - start << " sec."<<endl;
      // cout<<"% res : " << calculate_residual(A, b, x)<<endl;
      }
    */
    for(thread_count = 1; thread_count<=total_num_threads; ++thread_count)
    {
      Vector x(n, 0.);
      para.counter = 0;
      double start = omp_get_wtime();
# pragma omp parallel num_threads(thread_count) shared(A, b, x, para)
      {
        balanced_jacobi(A, b, x, para);
      }
      double end = omp_get_wtime();
      result[thread_count-1][0] = end - start;
      result[thread_count-1][1] = calculate_residual(A, b, x);
    }
  }

  if(!para.is_sparse)
  {
    Matrix A;
    Vector b;
    loadMarket(A, data_file_name);
    loadMarket(b, label_file_name);
    n = A.rows();
    if(b.size()!=A.cols())
    {
      cout<<"The size of A and b don't match!"<<endl;
      return 0;
    }
    cout<<"% start parallel ayn to solve linear equation"<<endl;
    cout<<"---------------------------------------------"<<endl;
    cout<<"The size of the problem is " <<n<<endl;

    /*==============================
      start the parallel solver
      ==============================*/
    int thread_count = 1;

    for(thread_count = 1; thread_count<=total_num_threads; ++thread_count)
    {
      Vector x(n, 0.);
      double start = omp_get_wtime();
      para.counter = 0;
# pragma omp parallel num_threads(thread_count) shared(A, b, x, para)
      {
        balanced_jacobi(A, b, x, para);
      }
      double end = omp_get_wtime();
      result[thread_count-1][0] = end - start;
      result[thread_count-1][1] = calculate_residual(A, b, x);
    }
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
