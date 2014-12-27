import matplotlib.pyplot as plt
import numpy as np
import csv
import scipy.signal

fileread=[]
with open('2014-12-25_20-41-08.csv','rb') as csvfile:
	spamreader= csv.reader(csvfile)
	for row in spamreader:
		fileread.append(row)

required=map(lambda x: x[18:22], fileread)
acceldata=map(lambda x: map(float,x),required[1:])

starttime=acceldata[0][0]
timecorrected=map(lambda x: [x[0]-starttime]+map(lambda y: y*9.81,x[1:4]),acceldata )

[timearr,ax,ay,az]= map(list,zip(*timecorrected))

# Applying median filter
ax=scipy.signal.medfilt(ax,5)
ay=scipy.signal.medfilt(ay,5)
az=scipy.signal.medfilt(az,5)

sx=sum(ax)/len(ax)
sy=sum(ay)/len(ay)
sz=sum(az)/len(az)

ax= map(lambda x: x-sx, ax)
ay= map(lambda x: x-sy, ay)
az= map(lambda x: x-sz, az)

lx=[0.0]*len(ax)


print "vx" , np.trapz(ax,timearr)
print "vy" , np.trapz(ay,timearr)
print "vz" , np.trapz(az,timearr)


xaxis=[i for i in xrange(len(ax))]

plt.figure(1)
plt.subplot(311)
# plt.plot(xaxis,ax,'r--',xaxis,lx,'b--')
plt.plot(ax)

# # plt.gca().add_line(line)
# # plt.add_line([0,30],[0.0,0.0],linewidth=2,color='red')

plt.subplot(312)
# plt.plot(xaxis,ay,'r--',xaxis,lx,'b--')
plt.plot(ay)

plt.subplot(313)
# plt.plot(xaxis,az,'r--',xaxis,lx,'b--')
plt.plot(az)

plt.show()	