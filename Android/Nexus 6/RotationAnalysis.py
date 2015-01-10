import numpy as np
import math
# import numpy.matrix
import copy
import csv
import matplotlib.pyplot as plt

#File input part
filename='Jan 10, 2015 5:51:42 PM_SensorFusion3.csv'
fileread=[]
with open(filename,'rb') as csvfile:
	spamreader= csv.reader(csvfile)
	for row in spamreader:
		fileread.append(row)

timed =map(lambda x: map(float,x),fileread[1:])
# starttime=timed[0][0]
# timed=map(lambda x: [x[0]-starttime] + x[1:], timed)

print "Number or entries before the dupilicates removal: ", len(timed)

#Removing the vlaues with same timestamp
prevtime=timed[0][0]
timecorrected=[timed[0]]
for i in xrange(len(timed)):
	if not(timed[i][0]==prevtime):
		prevtime=timed[i][0]
		timecorrected.append(timed[i])

print "Number of values after duplicate removal: ", len(timecorrected)

[timearr,r0,r1,r2,r3,r4,r5,r6,r7,r8,gr0,gr1,gr2,gr3,gr4,gr5,gr6,gr7,gr8,imid]=map(list, zip(*timed))

requiredpoints=[]
imid1=0
for elem in timed:
	# print elem[19]
	# print elem
	if (elem[19])==imid1:
		requiredpoints.append(elem)
		# print elem
		imid1+=1

# print requiredpoints

def get33matrix(arr):
	ans=[]
	for i in xrange(3):
		ans+=[arr[3*i:3*(i+1)]]
	return np.matrix(ans)

def getmatrices(arr):
	ans=[0,0,0]
	ans[0]=get33matrix(arr[1:10])
	ans[1]=get33matrix(arr[10:19])
	ans[2]=arr[19]
	return ans

matrixform=map(getmatrices,requiredpoints)

print len(matrixform)

relmatrices=[0]*(len(matrixform)-1)
gyrorelmatrices=[0]*(len(matrixform)-1)
for i in xrange(len(matrixform)-1):
	relmatrices[i]=matrixform[i+1][0].getT() * matrixform[i][0]
	gyrorelmatrices[i]=matrixform[i+1][1].getT() * matrixform[i][1]

# relangles=[0]*4
# gyrelangles=[0]*4

# for elem in relmatrices:
# 	print elem

def getangles(matrixa):
	ans=[0]*3
	listed=matrixa.tolist()
	ans[0]=math.atan2(listed[2][1],listed[2][2])
	ans[1]=math.atan2(-listed[2][0],math.sqrt(listed[2][1]**2 + listed[2][2]**2))
	ans[2]=math.atan2(listed[1][0],listed[0][0])
	ans=map(lambda x: x*180/math.pi,ans)
	return ans

# relangles=map(lambda x: (math.acos(x.tolist()[0][0]))*180/math.pi, relmatrices)
relangles=map(getangles, relmatrices)
# gyrorelangles=map(lambda x: (math.acos(x.tolist()[0][0]))*180/math.pi, gyrorelmatrices)
gyrorelangles=map(getangles, gyrorelmatrices)

for elem in relangles:
	print elem
print ""
print ""

for elem in gyrorelangles:
	print elem

# print matrixform

# print get33matrix(requiredpoints[0][1:10])


#Functions


#Applying Part

#Plotting Part

