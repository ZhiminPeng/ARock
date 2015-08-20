Download, Build and Run
=======================
ARock's build system relies on `GNU make <https://www.gnu.org/software/make/>`_. It can be easily build on Linux and Unix environments, and various versions of Microsoft Windows. A relative up-to-date C++ compiler (e.g., gcc >= 4.1) is required in all cases.


Download ARock
----------------
The ARock package can be downloaded from the following link::

  https://github.com/ZhiminPeng/ARock
  
  
Building ARock
----------------
On Linux or Unix machines with g++ and GNU make installed in standard locations, building ARock can be as simple as::

  cd ARock
  make

The executable files are in the bin folder. The following is the list of executable files::

  r_least_square [solver for regularized least square problems]
  r_logistic     [solver for regularized logistic regression]


Testing on Linux and Unix
-------------------------
Once ARock has been successfully compiled, it is a good idea to verify that the executable files are functioning properly. You can go to the test folder by the following::

  cd test


To test ARock for l1 regularized least square problem, run the following commands::

  ../bin/r_least_square -data rcv1_data.mtx -label rcv1_label.mtx -epoch 10 -is_sparse 1 -nthread 2 -type l1


To test ARock for solving l1 regularized logistic regression problem, run the following commands::

  ../bin/r_logistic -data rcv1_data.mtx -label rcv1_label.mtx -epoch 10 -is_sparse 1 -nthread 2 -type l1



Testing on Microsoft Windows
----------------------------
Tested platform: Windows 8, 64-bit, Visual Studio Express 2013 Desktop Version

  1. Click on ``ARock\visual_studio\ARock_visual_studio.sln``
     to launch.

  2. If you are running a different version of Visual Studio, you must
     set the Platform Toolset to the current version.

     For example, for "Visual Studio 2012", for the projects "least_squares"
     and "logistic_regression", go to
     ``Properties->Configuration Properties->Platform Toolset``
     and select ``Visual Studio 2012 (v110)``

  3. For each project, under ``Properties->Debugging->Command Arguments``
     you will find the following options on one line::

       -nthread 2
       -epoch 10
       -lambda 1.0
       -type l1
       -data "$(SolutionDir)\..\test\rcv1_data.mtx"
       -label "$(SolutionDir)\..\test\rcv1_label.mtx"
       -is_sparse 1

   Check `here <http://www.math.ucla.edu/~zhimin.peng/ARock/opt/regression.html>`_
   if you want to make changes to these options. For example, if your
   PC has 4 cores, then you can use -nthread 4 to use all the 4 cores.

  4. Both Debug and Release configurations are available.

  5. Press F5 or Ctrl-F5 to run the code.
