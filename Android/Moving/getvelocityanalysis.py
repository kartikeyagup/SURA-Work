import matplotlib.pyplot as plt
import numpy as np

#FILE INPUT PART
a=open('accelerometer_prateek_moving.txt')
lines=a.read().split('\n')
splitted=[]
for elem in lines:
	splitted.append(elem.split())
splitted.pop()

ax,ay,az,time=[],[],[],[]
for elem in splitted:
	ax.append(elem[1])
	ay.append(elem[3])
	az.append(elem[5])
	time.append(elem[7])

ax=map(float,ax)
ay=map(float,ay)
az=map(float,az)
time=list(np.cumsum(map(lambda x: float(x)/(10**9),time)))

print len(ax),len(time),time[-1]

timeunit=0.1

#FUNCTIONS PART

def findstaticbias(accelarrayx,lim=3):
	#Finds the static bias considering the the 1st lim readings and last lim readings default is 3 
	sum1=sum(accelarrayx[0:lim]) +sum(accelarrayx[len(accelarrayx)-lim:len(accelarrayx)])
	return sum1/(2*lim)

def fixstaticbias(accelarrayx):
	staticbias=findstaticbias(accelarrayx)				#calclulating the static bias
	applied=map(lambda x: x-staticbias, accelarrayx)	#Applying it now	
	return applied

def getvelocity(accelarrayx,timearray):
	fixed=fixstaticbias(accelarrayx)
	#TODO: Make this order n instead of n*n
	velarray=[0.0]*(len(accelarrayx))
	for i in xrange(len(fixed)):
		velarray[i]=np.trapz(fixed[0:1+i],timearray[0:i+1])
	return velarray


def getdisplacement(velarrayx,timearray):
	#TODO: Make this of order n instead of n*n
	disparrayx=[0.0]*len(velarrayx)
	for i in xrange(len(velarrayx)):
		disparrayx[i]=np.trapz(velarrayx[0:1+i],timearray[0:i+1])
	return disparrayx

#DATA COMPUTATION PART
fixedbias=fixstaticbias(ax)
print np.trapz(fixedbias)
velocity=getvelocity(ax,time)
displacement=getdisplacement(velocity,time)
# xaxis=[i for i in xrange(len(ax))]


#PLOTTING PART
plt.figure(1)

plt.subplot(411)
plt.ylabel('ax (m/s2)')
# plt.plot(xaxis,ax,'r--',xaxis,lx,'b--')
plt.plot(ax)


plt.subplot(412)
plt.ylabel('ax corrected (m/s2)')
# plt.plot(xaxis,ay,'r--',xaxis,lx,'b--')
plt.plot(fixedbias)


plt.subplot(413)
plt.ylabel('vx (m/s)')
plt.plot(velocity)

plt.subplot(414)
plt.ylabel('x (m)')
plt.plot(displacement)

plt.show()
