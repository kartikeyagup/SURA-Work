import math
import matplotlib.pyplot as plt
import numpy as np
import csv

def getRotMatrix(g,m):
	Hx = m[1]*g[2] - m[2]*g[1]
	Hy = m[2]*g[0] - m[0]*g[2]
	Hz = m[0]*g[1] - m[1]*g[0]
	h=[Hx,Hy,Hz]
	H=math.sqrt(Hx*Hx + Hy*Hy + Hz*Hz)
	if(H<0.1):
		return false
		#device is close to free fall (or in space?), or close to
        #magnetic north pole. Typical values are  > 100.
	invH=1.0 / H
	h=map(lambda x: x*invH, h)
	invA = 1.0/ math.sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2])
	g=map(lambda x: x*invA, g)
	Mx = g[1]*h[2] - g[2]*h[1]
	My = g[2]*h[0] - g[0]*h[2]
	Mz = g[0]*h[1] - g[1]*h[0]
	R=[i for i in h]+[Mx,My,Mz]+[j for j in g]
	return R

fileread=[]
with open('2014-12-25_20-41-08.csv','rb') as csvfile:
	spamreader= csv.reader(csvfile)
	for row in spamreader:
		fileread.append(row)

requiredGravity=map(lambda x: x[41:44], fileread)
gravityData=map(lambda x: map(float,x),requiredGravity[1:]) #each element of gravityData is a list containing three values of graviy

requiredMagnetic=map(lambda x: x[44:47], fileread)
magneticData=map(lambda x: map(float,x),requiredMagnetic[1:]) #each element of gravityData is a list containing three values of geomagnetic

rotMatrices=map(getRotMatrix,gravityData,magneticData) # mapping with the getRotMatrix function for each of the values of gavityData and magneticData
print rotMatrices[20]	#each element of rotMatrices is a list containing 9 elements of the corresponding Rotation Matrix