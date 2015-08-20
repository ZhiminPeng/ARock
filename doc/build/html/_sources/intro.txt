Introduction
**************


Overview
===========
ARock is an asynchronous parallel C++ library for solving equations and optimization problems on shared memory systems. The package calls `Eigen <http://eigen.tuxfamily.org/index.php?title=Main_Page>`_ Library for matrix and vector operations. OpenMP is used for parallelization. ARock includes the async-parallel implementations of

  * the relaxed Jacobi method for solving linear equations (coming soon...);
  * Newton's method for solving nonlinear equations (coming soon...);
  * the operator splitting method for regularized empirical risk minimization problems (LASSO, logistic regression);
  * the alternating direction method of multipliers (ADMM) for consensus and decentralized optimization problems (coming soon...).

  
ARock is written in C++. Python, Julia and Matlab interfaces are under development.
    

Dependencies
==============
* Functioning C++ compilers (`gcc <https://www.gnu.org/software/gcc/releases.html>`_ )
* `GNU make <https://www.gnu.org/software/make/>`_
* or Microsoft Visual C++


References
============
.. [ARock2015] Zhimin Peng, Yangyang Xu, Ming Yan, Wotao Yin, *ARock: an Algorithmic Framework for Asynchronous Parallel Coordinate Updates*,  arXiv preprint arXiv:1506.02396 (2015)
  

License and copyright
=====================
Except for the Eigen library header files, all files distributed with ARock are made available under the `New BSD license <http://www.opensource.org/licenses/bsd-license.php>`_,
which states::

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

     - Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.

     - Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

     - Neither the name of the owner nor the names of its contributors
       may be used to endorse or promote products derived from this software
       without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

Note that Eigen is `MPL2-licensed <https://www.mozilla.org/MPL/2.0/>`_.
