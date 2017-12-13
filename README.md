**mypy**: An Useful Toolbox for Control Engineers 
======

Overview
========

**mypy** is an useful toolbox for control engineers.
It can be used for a wide range of purposes: to identify a system transfer functions, to optimize a trajectory plan, ..., and so on.

Requirements
============

* Python 3.5+.
* *numpy*, *scipy*, and *sympy* (all included in **anaconda**)
* *mycvxopt* requires *cvxopt*.

Install
=======
Just append *mypy* to your *PYTHONPATH*, or

	import sys
	sys.path.append(r"/home/me/mypy")

Documentation
=============

Documentation is not available for now.

Examples
=============

* Optimized Multisine: better S/N-ratio excitation 

![optimized_multisine](images/optimized_multisine.jpg)

![FFT_of_optimized_multisine](images/FFT_of_optimized_multisine.jpg)

* System Identification: linear least squares, iterative weighted linear least squares, nonlinear least squares,  and maximum likelihood estimation solution. 

![Bodeplot](images/Bodeplot.jpg)

![System_Identification_from_10_Hz_to_500_Hz](images/System_Identification_from_10_Hz_to_500_Hz.jpg)

* Trajectory Planning: B-spline trajectory of jerk infinity-norm minimization with constraints

![pos_jerk_infinity_norm_minimization](images/pos_jerk_infinity_norm_minimization.png)
![vel_jerk_infinity_norm_minimization](images/vel_jerk_infinity_norm_minimization.png)
![acc_jerk_infinity_norm_minimization](images/acc_jerk_infinity_norm_minimization.png)
![jer_jerk_infinity_norm_minimization](images/jer_jerk_infinity_norm_minimization.png)


Copyright and License
=============

	2017-, Shimoda Takaki, The University of Tokyo

mypy is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

mypy is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

