Build package
==================
ARock's build system relies on `GNU make <https://www.gnu.org/software/make/>`_. It can be easily build on Linux and Unix environments, and, at least in theory, various versions of Microsoft Windows. A relative up-to-date C++ compiler (e.g., gcc >= 4.1) is required in all cases.


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


Testing the installation
--------------------------
Once ARock has been successfully compiled, it is a good idea to verify that the executable files are functioning properly. You can go to the test folder by the following::

  cd test


To test ARock for l1 regularized least square problem, run the following commands::

  ../bin/r_least_square -data rcv1_data.mtx -label rcv1_label.mtx -epoch 10 -is_sparse 1 -nthread 2 -type l1


To test ARock for solving l1 regularized logistic regression problem, run the following commands::

  ../bin/r_logistic -data rcv1_data.mtx -label rcv1_label.mtx -epoch 10 -is_sparse 1 -nthread 2 -type l1
