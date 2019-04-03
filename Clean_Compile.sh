find . -name "*.so" -exec rm {} \;

#OpenMP compiled
#f2py -m Fortran_functions --fcompiler=gfortran --f90flags='-fopenmp' -lgomp -c Fortran_functions.f95
#f2py -m Approximate_Deconvolution --fcompiler=gfortran --f90flags='-fopenmp' -lgomp -c Approximate_Deconvolution.f95
#f2py -m Multigrid_solver --fcompiler=gfortran --f90flags='-fopenmp' -lgomp -c Multigrid_solver.f95


#Serial compiled - basic functions used everywhere
f2py -c --fcompiler=gfortran -m Fortran_functions Fortran_functions.f95
f2py -c --fcompiler=gfortran -m Spectral_Poisson Spectral_Poisson.f95
f2py -c --fcompiler=gfortran -m Multigrid_solver Multigrid_solver.f95
# #Standard closures
f2py -c --fcompiler=gfortran -m Standard_Models Standard_Models.f95
# #Relaxation filtering
f2py -c --fcompiler=gfortran -m Relaxation_filtering Relaxation_filtering.f95
# #ML Closures
f2py -c --fcompiler=gfortran -m Ml_convolution Ml_convolution.f95
f2py -c --fcompiler=gfortran -m ML_Regression ML_Regression.f95
f2py -c --fcompiler=gfortran -m ML_Nearest_neighbors ML_Nearest_neighbors.f95
f2py -c --fcompiler=gfortran -m ML_feature_functions ML_feature_functions.f95
f2py -c --fcompiler=gfortran -m ML_Logistic_functions ML_Logistic_functions.f95
f2py -c --fcompiler=gfortran -m ML_AD_Classification ML_AD_Classification.f95
f2py -c --fcompiler=gfortran -m ML_TBDNN ML_TBDNN.f95

#Change this according to your distribution of python
for x in *.cpython-36m-x86_64-linux-gnu.so; do mv "$x" "${x%.cpython-36m-x86_64-linux-gnu.so}.so"; done