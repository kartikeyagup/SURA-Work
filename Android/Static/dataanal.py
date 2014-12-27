import matplotlib.pyplot as plt
import numpy as np

a=open('accelerometer.txt')
lines=a.read().split('\n')
splitted=[]
for elem in lines:
	splitted.append(elem.split('\t'))
ax=[]
ay=[]
for elem in splitted:
	ax.append(elem[0])
	ay.append(elem[-1])

ax=map(float,ax)
ay=map(float,ay)


ax=ax[1:]
ay=ay[1:]
# ax=map(abs,ax)

lx=[0.0]*len(ax)
print np.trapz(ax)


xaxis=[i for i in xrange(len(ax))]

# line=plt.Line2D((0,0.0),(30,3.0),lw=2.5)

plt.figure(1)
plt.subplot(211)
plt.plot(xaxis,ax,'r--',xaxis,lx,'b--')

# plt.gca().add_line(line)
# plt.add_line([0,30],[0.0,0.0],linewidth=2,color='red')

plt.subplot(212)
plt.plot(xaxis,ay,'r--',xaxis,lx,'b--')
plt.show()


# pylab.figure(1)
# pylab.xlabel("Time")
# pylab.plot(xaxis,ax)
# pylab.show()

# print ax
# print ay