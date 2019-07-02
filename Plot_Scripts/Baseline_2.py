import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
	vorticity_field = np.load(filename)
	return vorticity_field

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

	# X and Y direction alone predictions



	return sf, sf_x, sf_y

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


def plot_sf(sf,type_field,cv=None):
	plt.semilogx(sf[:,0],sf[:,1],label=type_field,color=cv)

fil_dns = load_field('Source.504',256)
ml = load_data('Field_ML.npy')
ad = load_data('Field_AD.npy')
udns = load_data('Field_UDNS.npy')

sf_fdns, sf_fdns_x, sf_fdns_y = calculate_vort_srf(fil_dns,256)
sf_udns, sf_udns_x, sf_udns_y = calculate_vort_srf(udns,256)
sf_ad, sf_ad_x, sf_ad_y = calculate_vort_srf(ad,256)
sf_ml, sf_ml_x, sf_ml_y = calculate_vort_srf(ml,256)

plt.figure()
plot_sf(sf_fdns,'FDNS')
plot_sf(sf_udns,'UNS','red')
plot_sf(sf_ad,'AD','blue')
plot_sf(sf_ml,'ML','orange')

plt.ylim([0,1600])
# plt.xlim([sf_fdns[0,1],sf_fdns[0,-1]])
plt.ylabel(r'$S_\omega$',fontsize='16')
plt.xlabel(r'$r$',fontsize='16')
plt.legend(loc='upper left')
plt.show()