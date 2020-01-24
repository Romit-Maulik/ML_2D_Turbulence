import numpy as np
import matplotlib.pyplot as plt

def load_field(filename,data_size):
    #Loading data from Tecplot file (check number of skiprows) - no tecplot header
    fine_data = np.loadtxt(filename,skiprows=2)
    # Store in numpy array
    vort= np.arange((data_size + 1)* (data_size + 1), dtype='double').reshape(data_size + 1, data_size + 1)
    iter = 0
    for j in range(0, data_size + 1):
        for i in range(0, data_size + 1):
            vort[i, j] = fine_data[iter, 0]
            iter = iter + 1

    vortper = np.arange((data_size) * (data_size), dtype='double').reshape(data_size, data_size)
    vortper = vort[0:data_size,0:data_size]

    return vortper

def calculate_vort_srf(vorticity_field, data_size):
    #Define array for structure function storage
    sf_x = np.zeros(shape=(1,2),dtype='double')
    sf_y = np.zeros(shape=(1,2),dtype='double')

    #Sampling in x
    r = 1
    while r <= (data_size/2 - 1):
        sf_val = 0.0
        for j in range(data_size):
            w = vorticity_field[0,j]
            wpr = vorticity_field[r,j]
            sf_val = sf_val + (w-wpr)**2

        sf_x = np.concatenate((sf_x,np.asarray([[r/data_size*2.0*np.pi,sf_val]])),axis=0)
        r = r+1

    sf_x[:,1] = sf_x[:,1]/float(r)

    #Sampling in y
    r = 1
    while r <= (data_size/2 - 1):
        sf_val = 0.0
        for i in range(data_size):
            w = vorticity_field[i,0]
            wpr = vorticity_field[i,r]
            sf_val = sf_val + (w-wpr)**2

        sf_y = np.concatenate((sf_y,np.asarray([[r/data_size*2.0*np.pi,sf_val]])),axis=0)
        r = r+1

    sf_y[:,1] = sf_y[:,1]/float(r)

    # Unite sampling
    sf = np.copy(sf_x)
    sf[:,1] = (sf[:,1] + sf_y[:,1])

def plot_sf(sf,type_field,cv=None):
    plt.semilogx(sf[:,0],sf[:,1],label=type_field,color=cv)


if __name__ == '__main__':
    dns_0 = load_field('fort.510',2048)
    dns_1 = load_field('fort.501',2048)
    
    plt.figure()
    plot_sf(dns_0,'Structure Function 10')
    plot_sf(dns_1,'Structure Function 1')
    # plt.ylim([0,1600])
    plt.ylabel(r'$S_\omega$',fontsize='16')
    plt.xlabel(r'$r$',fontsize='16')
    plt.legend(loc='upper left')
    plt.show()