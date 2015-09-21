Tested platform: Windows 8, 64-bit, Visual Studio 2015 Desktop Version

1. Click on [ARock Home Folder]\visual_studio_2015\visual_studio_2015.sln
   to launch.

2. If you are running a different version of Visual Studio, you must
   set the Platform Toolset to the current version.

   For example, for "Visual Studio 2012", for the projects "least_squares"
   and "logistic_regression", go to
     Properties->Configuration Properties->Platform Toolset
   and select "Visual Studio 2012 (v110)"

3. For each project, under Properties->Debugging->Command Arguments
   you will find the following options on one line

    -nthread 2
    -epoch 10
    -lambda 1.0
    -type l1
    -data "$(SolutionDir)\..\test\rcv1_data.mtx"
    -label "$(SolutionDir)\..\test\rcv1_label.mtx"
    -is_sparse 1

   Check http://www.math.ucla.edu/~zhimin.peng/ARock/opt/regression.html
   if you want to make changes to these options. For example, if your
   PC has 4 cores, then you can use -nthread 4 to use all the 4 cores.

4. Both Debug and Release configurations are available, but under x64.
