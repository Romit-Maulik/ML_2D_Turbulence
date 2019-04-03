cd Fortran_Objects/
find . -name "*.so" -exec rm {} \;

#OpenMP compiled
#f2py -m Fortran_functions --fcompiler=gfortran --f90flags='-fopenmp' -lgomp -c Fortran_functions.f95
#f2py -m Approximate_Deconvolution --fcompiler=gfortran --f90flags='-fopenmp' -lgomp -c Approximate_Deconvolution.f95
#f2py -m Multigrid_solver --fcompiler=gfortran --f90flags='-fopenmp' -lgomp -c Multigrid_solver.f95

#Serial compiled - basic functions used everywhere
f2py -c --fcompiler=gfortran -m Fortran_Functions Fortran_Functions.f95
f2py -c --fcompiler=gfortran -m Spectral_Poisson Spectral_Poisson.f95
f2py -c --fcompiler=gfortran -m Multigrid_Solver Multigrid_Solver.f95
# #Standard closures
f2py -c --fcompiler=gfortran -m Standard_Models Standard_Models.f95
# #Relaxation filtering
f2py -c --fcompiler=gfortran -m Relaxation_Filtering Relaxation_Filtering.f95
# #ML Closures
f2py -c --fcompiler=gfortran -m Ml_Convolution Ml_Convolution.f95
f2py -c --fcompiler=gfortran -m ML_Regression ML_Regression.f95
f2py -c --fcompiler=gfortran -m ML_Nearest_Neighbors ML_Nearest_Neighbors.f95
f2py -c --fcompiler=gfortran -m ML_Feature_Functions ML_Feature_Functions.f95
f2py -c --fcompiler=gfortran -m ML_Logistic_Functions ML_Logistic_Functions.f95
f2py -c --fcompiler=gfortran -m ML_AD_Classification ML_AD_Classification.f95
f2py -c --fcompiler=gfortran -m ML_TBDNN ML_TBDNN.f95

#Change this according to your distribution of python
for x in *.cpython-36m-x86_64-linux-gnu.so; do mv "$x" "${x%.cpython-36m-x86_64-linux-gnu.so}.so"; done

cd ..