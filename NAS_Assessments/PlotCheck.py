import numpy as np
import matplotlib.pyplot as plt
import Fortran_Functions

def load_field(filename,data_size):
    #Loading data from Tecplot file (check number of skiprows) - no tecplot header
    fine_data = np.loadtxt(filename)

    # Store in numpy array
    vort= np.arange((data_size + 1)* (data_size + 1), dtype='double').reshape(data_size + 1, data_size + 1)

    iter = 0
    for j in range(0, data_size + 1):
        for i in range(0, data_size + 1):
            vort[i, j] = fine_data[iter, 0]
            iter = iter + 1

    vortper = np.arange((data_size) * (data_size), dtype='double').reshape(data_size, data_size)

    for j in range(0, data_size):
        for i in range(0, data_size):
            vortper[i, j] = vort[i, j]

    return vortper

def load_dns_field(filename,data_size):
    #Loading data from Tecplot file (check number of skiprows) - no tecplot header
    fine_data = np.loadtxt(filename,skiprows=2)

    # Store in numpy array
    vort= np.arange((data_size + 1)* (data_size + 1), dtype='double').reshape(data_size + 1, data_size + 1)

    iter = 0
    for j in range(0, data_size + 1):
        for i in range(0, data_size + 1):
            vort[i, j] = fine_data[iter, 2]
            iter = iter + 1

    vortper = np.arange((data_size) * (data_size), dtype='double').reshape(data_size, data_size)

    for j in range(0, data_size):
        for i in range(0, data_size):
            vortper[i, j] = vort[i, j]

    return vortper



#Load first
kplot_udns  = np.load('Spectra_UDNS.npy')[0]
eplot_udns = np.load('Spectra_UDNS.npy')[1]


eplot_nas = np.load('Spectra_NAS.npy')[1]
eplot_bl = np.load('Spectra_Baseline.npy')[1]

#DNS
kplot_DNS = np.load('DNS_32_Spectra.npy')[0]
eplot_DNS = np.load('DNS_32_Spectra.npy')[1]


scale_plot = np.array([[10,0.1],[100,1.0e-4]])

plt.loglog(kplot_DNS,eplot_DNS,label=r'DNS')
plt.loglog(kplot_udns,eplot_udns,label=r'UNS',color='red')
plt.loglog(kplot_udns,eplot_bl,label='Baseline',color='orange')
plt.loglog(kplot_udns,eplot_nas,label='NAS',color='green')


plt.loglog(scale_plot[:,0],scale_plot[:,1],color='black',linestyle='dashed',label=r'$k^{-3}$')
plt.xlim([1,500])
plt.ylim([1e-8,1])
plt.xlabel(r'$k$',fontsize='16')
plt.ylabel(r'$E(k)$',fontsize='16')

plt.tick_params(axis='both', labelsize='12')
plt.legend(fontsize='12',loc='upper right')
plt.show()

# #Dns field
# data_size = 2048
# dns_vort = load_dns_field('fort.504',data_size)

#Fourier filtered DNS
data_size = 256
coarse_vort = load_field('Source.504',data_size)
# arr_len = int(0.5*np.sqrt(float(data_size*data_size + data_size*data_size)))-1
# eplot_FDNS = np.zeros(arr_len+1,dtype='double')
# kplot_FDNS = np.arange(0,arr_len+1,1,dtype='double')
# Fortran_functions.spec(coarse_vort,eplot_FDNS)

# # Comparison of DNS and filtered DNS
# x_dns = np.linspace(0, 2.0*np.pi, 2048)
# X_dns, Y_dns = np.meshgrid(x_dns, x_dns)

#Filtered DNS resolution
x = np.linspace(0, 2.0*np.pi, 256)
X, Y = np.meshgrid(x, x)
levels = np.linspace(-25,25,10)
aspectratio=1.0

# fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(9,4.5))
# ax[0].set_xlabel('x',fontsize=18)
# ax[0].set_ylabel('y',fontsize=18)
# ax[1].set_xlabel('x',fontsize=18)
# ax[1].set_ylabel('y',fontsize=18)

# #Format axes
# ax[0].tick_params(left=True,bottom=True, labelbottom='on',labelleft='on',labelsize=12)
# ax[0].tick_params(left=True,bottom=True, labelbottom='on',labelleft='on',labelsize=12)
# ax[1].tick_params(left=True,bottom=True, labelbottom='on',labelleft='on',labelsize=12)
# ax[1].tick_params(left=True,bottom=True, labelbottom='on',labelleft='on',labelsize=12)
# cbar_ax = fig.add_axes([0.17, 0.95, 0.7, 0.02])

# cs1 = ax[0].contour(X_dns,Y_dns,dns_vort,cmap='jet',levels=levels)
# cs2 = ax[1].contour(X,Y,coarse_vort,cmap='jet',levels=levels)
# fig.colorbar(cs1, cax=cbar_ax,orientation='horizontal')

# plt.show()


#LES resolution assessments
fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(8,8))
cbar_ax = fig.add_axes([0.17, 0.95, 0.7, 0.02])
plt.subplots_adjust(wspace=0.3,hspace=0.3)


ax[1,0].set_xlabel(r'$x$',fontsize=18)
ax[1,0].set_ylabel(r'$y$',fontsize=18)
ax[1,1].set_xlabel(r'$x$',fontsize=18)
ax[1,1].set_ylabel(r'$y$',fontsize=18)

ax[0,0].set_ylabel(r'$y$',fontsize=18)
ax[0,0].set_xlabel(r'$x$',fontsize=18)
ax[1,0].set_xlabel(r'$x$',fontsize=18)
ax[1,0].set_ylabel(r'$y$',fontsize=18)

ax[0,1].set_ylabel(r'$y$',fontsize=18)
ax[0,1].set_xlabel(r'$x$',fontsize=18)

#Format axes
ax[0,0].tick_params(labelsize=12)
ax[1,0].tick_params(labelsize=12)


#Machine Learning Stabilized
field = np.load('Field_NAS.npy')
cs1 = ax[0,0].contour(X,Y,field,cmap='jet',levels=levels)
field = np.load('Field_UDNS.npy')
cs2 = ax[1,0].contour(X,Y,field,cmap='jet',levels=levels)

#Format axes
ax[0,1].tick_params(bottom=True, top=False, left=True, labelbottom='on',labelleft='on',labelsize=12)
ax[1,1].tick_params(bottom=True, top=False, left=True, labelbottom='on',labelleft='on',labelsize=12)
ax[0,0].tick_params(bottom=True, top=False, left=True, labelbottom='on',labelleft='on',labelsize=12)
ax[1,0].tick_params(bottom=True, top=False, left=True, labelbottom='on',labelleft='on',labelsize=12)


cbar_ax = fig.add_axes([0.17, 0.95, 0.7, 0.02])

#Machine Learning Stabilized
field = np.load('Field_Baseline.npy')
cs4 = ax[0,1].contour(X,Y,field,cmap='jet',levels=levels)

cs5 = ax[1,1].contour(X,Y,coarse_vort,cmap='jet',levels=levels)

fig.colorbar(cs1, cax=cbar_ax,orientation='horizontal')

plt.show()

