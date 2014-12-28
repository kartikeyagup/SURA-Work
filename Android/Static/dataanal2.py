import matplotlib.pyplot as plt
import numpy as np

a=open('accelerometer_shrey_dad.txt')
lines=a.read().split('\n')
splitted=[]
for elem in lines:
	splitted.append(elem.split())
splitted.pop()


# print splitted


ax=[]
ay=[]
az=[]
for elem in splitted:
	ax.append(elem[1])
	ay.append(elem[3])
	az.append(elem[5])



ax=map(float,ax)
ay=map(float,ay)
az=map(float,az)

sx=sum(ax)/len(ax)
sy=sum(ay)/len(ay)
sz=sum(az)/len(az)


ax= map(lambda x: x-sx, ax)
ay= map(lambda x: x-sy, ay)
az= map(lambda x: x-sz, az)

ax=ax[:500]
ay=ay[:500]
az=az[:500]


# ax=ax[1:]
# ay=ay[1:]
# # ax=map(abs,ax)

lx=[0.0]*len(ax)
print "vx" , np.trapz(ax)
print "vy" , np.trapz(ay)
print "vz" , np.trapz(az)



xaxis=[i for i in xrange(len(ax))]

# # line=plt.Line2D((0,0.0),(30,3.0),lw=2.5)

plt.figure(1)
plt.subplot(311)
plt.ylabel('ax (m/s2')
# plt.plot(xaxis,ax,'r--',xaxis,lx,'b--')
plt.plot(ax)

# # plt.gca().add_line(line)
# # plt.add_line([0,30],[0.0,0.0],linewidth=2,color='red')

plt.subplot(312)
plt.ylabel('ay (m/s2)')
# plt.plot(xaxis,ay,'r--',xaxis,lx,'b--')
plt.plot(ay)

plt.subplot(313)
plt.ylabel('az (m/s2)')
# plt.plot(xaxis,az,'r--',xaxis,lx,'b--')
plt.plot(az)

plt.show()


# # pylab.figure(1)
# # pylab.xlabel("Time")
# # pylab.plot(xaxis,ax)
# # pylab.show()

# # print ax
# # print ay