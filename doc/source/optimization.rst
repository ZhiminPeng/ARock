Optimization
**************
ARock currently supports solving convex optimization problems with operator splitting methods. We provide async-parallel gradient descent method for solving :math:`\ell_2` regularized least square and logistic regression problems. We also provide async-parallel forward-backward splitting method for solving :math:`\ell_1` regularized least square and logistic regression problem.


.. toctree::
   :maxdepth: 1

   opt/regression
   opt/consensus
   opt/decentralize
