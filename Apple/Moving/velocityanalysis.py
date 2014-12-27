import matplotlib.pyplot as plt
import numpy as np
import csv


fileread=[]
with open('2014-12-26_22-37-07.csv','rb') as csvfile:
	spamreader= csv.reader(csvfile)
	for row in spamreader:
		fileread.append(row)

required=map(lambda x: x[18:22], fileread)
acceldata=map(lambda x: map(float,x),required[1:])

starttime=acceldata[0][0]
timecorrected=map(lambda x: [x[0]-starttime]+map(lambda y: y*9.81,x[1:4]),acceldata )

[timearr,ax,ay,az]= map(list,zip(*timecorrected))

def findstaticbias(accelarrayx,lim=10):
	#Finds the static bias considering the the 1st lim readings and last lim readings default is 3 
	sum1=sum(accelarrayx[0:lim]) +sum(accelarrayx[len(accelarrayx)-lim:len(accelarrayx)])
	return sum1/(2*lim)

def getlimitingvalue(accelarrayx,lim=10):
	l1=map(abs,accelarrayx[0:lim])
	l2=map(abs,accelarrayx[len(accelarrayx)-lim:len(accelarrayx)])
	return max(max(l1),max(l2))

def fixstaticbias(accelarrayx):
	staticbias=findstaticbias(accelarrayx)				#calclulating the static bias
	applied=map(lambda x: x-staticbias, accelarrayx)	#Applying it now	
	limitvalue=getlimitingvalue(applied)
	print "limitvalus is",limitvalue
	for i in xrange(len(accelarrayx)):
		if abs(applied[i])<limitvalue:
			applied[i]=0
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


fixedbias=fixstaticbias(az)
print np.trapz(fixedbias)
velocity=getvelocity(az,timearr)
displacement=getdisplacement(velocity,timearr)

#PLOTTING PART
plt.figure(1)

plt.subplot(411)
plt.ylabel('Ax (m/s2)')
# plt.plot(xaxis,ax,'r--',xaxis,lx,'b--')
plt.plot(az)


plt.subplot(412)
plt.ylabel('Corrected Ax (m/s2)')
# plt.plot(xaxis,ay,'r--',xaxis,lx,'b--')
plt.plot(fixedbias)


plt.subplot(413)
plt.ylabel('Vx (m/s)')
plt.plot(velocity)

plt.subplot(414)
plt.ylabel('x (m)')
plt.xlabel('Time : (100=1s)')
plt.plot(displacement)

plt.show()
