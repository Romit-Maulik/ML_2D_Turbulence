import os, sys
import numpy as np
import h5py

# Fixing paths
HERE = os.path.dirname(os.path.abspath(__file__))+r'/'
PARENT = os.path.dirname(os.getcwd())+r'/'
FORTRAN_OBJECTS = PARENT + 'Fortran_Objects/'
sys.path.insert(0,FORTRAN_OBJECTS)

import Spectral_Poisson

# Read DNS data files and save to h5
for file_num in range(11):
    # # DNS resolution
    # filename = 'fort.'+str(500+file_num)
    # data = np.loadtxt(filename,skiprows=2)[:,-1] #Only need vorticity
    # side_shape = int(np.sqrt(np.shape(data)[0]))
    # dx = 2.0*np.pi/float(side_shape)
    
    # # Finding streamfunction
    # vort = np.reshape(data,newshape=(side_shape,side_shape))
    # stream = np.zeros(shape=np.shape(vort))
    # Spectral_Poisson.solve_poisson(stream, -vort, dx, dx)

    # LES resolution
    filename = 'Source.'+str(500+file_num)
    data = np.loadtxt(filename)
    side_shape = int(np.sqrt(np.shape(data)[0]))
    dx = 2.0*np.pi/float(side_shape)

    vortf = np.reshape(data[:,0],newshape=(side_shape,side_shape))
    streamf = np.reshape(data[:,1],newshape=(side_shape,side_shape))
    strainf = np.reshape(data[:,2],newshape=(side_shape,side_shape))
    vortgradf = np.reshape(data[:,3],newshape=(side_shape,side_shape))
    lapf = np.reshape(data[:,4],newshape=(side_shape,side_shape))
    absvel = np.reshape(data[:,5],newshape=(side_shape,side_shape))
    
    sgs_sm = np.reshape(data[:,6],newshape=(side_shape,side_shape))
    sgs_lt = np.reshape(data[:,7],newshape=(side_shape,side_shape))
    sgs_ad = np.reshape(data[:,8],newshape=(side_shape,side_shape))
    sgs = np.reshape(data[:,9],newshape=(side_shape,side_shape))

    ad_diff = np.subtract(sgs,sgs_ad)**2
    lt_diff = np.subtract(sgs,sgs_lt)**2
    sm_diff = np.subtract(sgs,sgs_sm)**2

    one_hot_sgs = np.zeros(shape=(side_shape,side_shape,3),dtype='int')

    for i in range(side_shape):
        for j in range(side_shape):
            if ad_diff[i,j] < lt_diff[i,j] and ad_diff[i,j] < sm_diff [i,j]:
                one_hot_sgs[i,j] = [1,0,0]
            elif lt_diff[i,j] < ad_diff[i,j] and lt_diff[i,j] < sm_diff [i,j]:
                one_hot_sgs[i,j] = [0,1,0]
            else:
                one_hot_sgs[i,j] = [0,0,1]

    h5_filename = 'time_'+str(file_num)+'.h5'
    h5f = h5py.File(h5_filename, 'w')
    h5f.create_dataset('omegaf', data=vortf)
    h5f.create_dataset('streamf', data=streamf)
    h5f.create_dataset('strainf', data=strainf)
    h5f.create_dataset('vortgradf', data=vortgradf)
    h5f.create_dataset('lapf', data=lapf)
    h5f.create_dataset('one_hot_sgs', data=one_hot_sgs) # For classification tasks
    h5f.close()

    