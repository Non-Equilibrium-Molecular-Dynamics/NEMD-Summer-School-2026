import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 16}) 

intrinsic_out ='intrinsic.pdf'
lab_out ='non-intrinsic.pdf'

fit  = np.loadtxt('./lab_density_fit.dat',skiprows=1)
data = np.loadtxt('./itim_lab.dat', skiprows=1)
dataI= np.loadtxt('itim_intrinsic.dat',skiprows=1)

# we first plot the intrinsic density profile

plt.plot(dataI[:,0],dataI[:,1])
plt.xlim([-7,7])
plt.ylim([0,1])
plt.xlabel(r'$(z -\xi)/ \sigma$')
plt.ylabel(r'$\rho \, \sigma^3$')
plt.tight_layout()
plt.savefig(intrinsic_out)
plt.close()
print('intrinsic profile saved in '+intrinsic_out)

# then the lab-frame one, with layers and fit
plt.plot(data[:,0],data[:,1],lw=2)
plt.plot(data[:,0],data[:,2])
plt.plot(data[:,0],data[:,3])
plt.plot(data[:,0],data[:,4])
plt.plot(data[:,0],data[:,5])
plt.plot(data[:,0],data[:,6])

plt.plot(fit[:,0],fit[:,2],'--',c='w')
mid = np.sum(data[:,0]*data[:,1])/np.sum(data[:,1])
plt.xlim([mid-12,mid+12])
plt.xlabel(r'$z / \sigma$')
plt.ylabel(r'$\rho \, \sigma^3$')
plt.tight_layout()
plt.savefig(lab_out)
plt.close()
print('lab-frame profile saved in '+lab_out)


