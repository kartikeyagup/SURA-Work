import numpy as np
import copy
import csv
import matplotlib.pyplot as plt

#File input part
filename='Jan 5, 2015 11_20_17 AM.csv'
fileread=[]
with open(filename,'rb') as csvfile:
	spamreader= csv.reader(csvflie)
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

timed=timecorrected

startindex,done=0,False
while not(done):
	if np.prod(timed[startindex])>0:
		done=True
	else:
		startindex+=1

starttime=timed[startindex][0]
timed=map(lambda x: [x[0]-starttime] + x[1:], timed[startindex:])


print "Number of values after initialisation: ", len(timed)

[timearr,wx,wy,wz,ax,ay,az,gravx,gravy,gravz,magx,magy,magz]=map(list, zip(*timed))

#Functions
def FindStaticBias(Elements, Size=50):
	#Gets the average of the 1st Size number of elements in the Elements and the last Size default is 10.
	# print Elements[:Size]
	return (sum(Elements[:Size])+sum(Elements[-Size:]))/(2*Size)

def GetLimitingValue(Elements, Size=50):
	#This is to get a limiting value under which the Elements will be truncated to 0
	#Considering the 1st Size and last Size elements
	#Returns the max absolute value in the first Size and Last Size elements
	ElementsInBeginning=map(abs,Elements[:Size])
	ElementsInEnd=map(abs,Elements[-Size:])
	return max(max(ElementsInBeginning),max(ElementsInEnd))

def ApplyCorrections(Elements):
	#This function applies corrections to Elements
	Corrected=copy.deepcopy(Elements)
	#Static Bias Correction
	StaticValue=FindStaticBias(Corrected)
	# print StaticValue
	Corrected=map(lambda x: x- StaticValue, Corrected)
	#TODO truncation widow part
	#Truncating to 0
	LimitingValue=GetLimitingValue(Corrected)
	print LimitingValue
	for i in xrange(len(Corrected)):
		if abs(Corrected[i])<LimitingValue:
			Corrected[i]=0
	windowsSize=6
	return Corrected

# def CorrectVelocityArray(VelocityElements,TimeElements):
# 	WindowSize=50
# 	i,SofarCounter,ValSofar=0,1,VelocityElements[0]
# 	for i in xrange(len(VelocityElements)):
# 		if VelocityElements[i]==ValSofar:
# 			SofarCounter+=1
# 		else:
# 			ValSofar=VelocityElements[i]
# 			SofarCounter=1
# 		if SofarCounter>=WindowSize:
# 			VelocityElements[i]=0
# 	return VelocityElements		

def GetVeleocityArray(AccelerationElements,TimeElements):
	CorrectedAcceleration=ApplyCorrections(AccelerationElements)
	VelocityArray=[0.0]*len(AccelerationElements)
	WindowSize=50
	SofarCounter,ValSofar=1,VelocityArray[0]
	for i in xrange(1,len(AccelerationElements)):
		VelocityArray[i]=VelocityArray[i-1]+0.5*(CorrectedAcceleration[i-1]+CorrectedAcceleration[i])*(TimeElements[i]-TimeElements[i-1])
		if VelocityArray[i]==ValSofar:
			SofarCounter+=1
		else:
			ValSofar=VelocityArray[i]
			SofarCounter=1
		if SofarCounter>=WindowSize:
			VelocityArray[i]=0
	return VelocityArray

def GetDisplacementArray(VelocityElements,TimeElements):
	DisplacementArray=[0.0]*len(VelocityElements)
	for i in xrange(1,len(VelocityElements)):
		DisplacementArray[i]=DisplacementArray[i-1]+0.5*(VelocityElements[i]+VelocityElements[i-1])*(TimeElements[i]-TimeElements[i-1])
	return DisplacementArray

def GetRotationGyro(RotMatrix,(wx,wy,wz),timediff):
	# RotMatrix is the present rotation matrix.
	# wx,wy,wz are the gyroscope readings
	# timediff is the time difference between the rotationmatrix and the present gx,gy,gz
	# Returns the Rotation matrix obtained from them as a 9 elements array.	
	# Keeping X axis along magnetic north, Z axis along gravity
	RotationMatrix=[0.0]*9
	#TODO: Put in trapezoidal Rule
	DeltaR=[1.0]*9
	DeltaR[1]=  wz*timediff
	DeltaR[2]= -wy*timediff
	DeltaR[3]= -wz*timediff
	DeltaR[5]=  wx*timediff
	DeltaR[6]=  wy*timediff
	DeltaR[7]= -wx*timediff
	RotationMatrix[0]=RotMatrix[0]*DeltaR[0]+RotMatrix[1]*DeltaR[3]+RotMatrix[2]*DeltaR[6]
	RotationMatrix[1]=RotMatrix[0]*DeltaR[1]+RotMatrix[1]*DeltaR[4]+RotMatrix[2]*DeltaR[7]
	RotationMatrix[2]=RotMatrix[0]*DeltaR[2]+RotMatrix[1]*DeltaR[5]+RotMatrix[2]*DeltaR[8]
	RotationMatrix[3]=RotMatrix[3]*DeltaR[0]+RotMatrix[1]*DeltaR[3]+RotMatrix[2]*DeltaR[6]
	RotationMatrix[4]=RotMatrix[3]*DeltaR[1]+RotMatrix[1]*DeltaR[4]+RotMatrix[2]*DeltaR[7]
	RotationMatrix[5]=RotMatrix[3]*DeltaR[2]+RotMatrix[1]*DeltaR[5]+RotMatrix[2]*DeltaR[8]
	RotationMatrix[6]=RotMatrix[6]*DeltaR[0]+RotMatrix[1]*DeltaR[3]+RotMatrix[2]*DeltaR[6]
	RotationMatrix[7]=RotMatrix[6]*DeltaR[1]+RotMatrix[1]*DeltaR[4]+RotMatrix[2]*DeltaR[7]
	RotationMatrix[8]=RotMatrix[6]*DeltaR[2]+RotMatrix[1]*DeltaR[5]+RotMatrix[2]*DeltaR[8]
	#TODO put in corrective measures in this to 
	return RotationMatrix

def GetRotationMagGrav(mx,my,mz,gx,gy,gz):
	# mx,my,mz are the magnetic readings
	# gx,gy,gz are the gravity readings
	# Returns the Rotation matrix obtained from them as a 9 elements array.	
	# Keeping X axis along magnetic north, Z axis along gravity
	m_mag=(mx**2+my**2+mz**2)**0.5
	g_mag=(gx**2+gy**2+gz**2)**0.5
	M=map(lambda x: x/m_mag, [mx,my,mz])
	G=map(lambda x: x/g_mag, [gx,gy,gz])
	H=[0.0]*3
	H[0]=my*gz-mz*gy
	H[1]=mz*gx-mx*gz
	H[2]=mx*gy-my*gx
	h_mag=(H[0]**2+H[1]**2+H[2]**2)**0.5
	H=map(lambda x: x/h_mag, H)
	RotMatrix=M+H+G
	return RotMatrix

def ApplyRotationMatrix(RotMatrix,(ax,ay,az)):
	# RotMatrix is a 1*9 array having the 9 elements of the rotation matrix
	# ax,ay,az are the elements on which RotMatrix is applied.
	# Returns a 1*3 array having the matrix multiplied
	AppliedOutput=[0.0]*3
	AppliedOutput[0]=RotMatrix[0]*ax+RotMatrix[1]*ay+RotMatrix[2]*az
	AppliedOutput[1]=RotMatrix[3]*ax+RotMatrix[4]*ay+RotMatrix[5]*az
	AppliedOutput[2]=RotMatrix[6]*ax+RotMatrix[7]*ay+RotMatrix[8]*az
	return AppliedOutput

#Applying Part

rotmatrices=map(GetRotationMagGrav,magx,magy,magz,gravx,gravy,gravz)
accelerationxyz=list(zip(*[ax,ay,az]))
accelerationXYZ=map(ApplyRotationMatrix,rotmatrices,accelerationxyz)

[ax,ay,az]=map(list,zip(*accelerationxyz))

correctedax=ApplyCorrections(ax)
correcteday=ApplyCorrections(ay)
correctedaz=ApplyCorrections(az)

velocityx=GetVeleocityArray(ax,timearr)
velocityy=GetVeleocityArray(ay,timearr)
velocityz=GetVeleocityArray(az,timearr)

# displacementx=CorrectVelocityArray(velocityx,timearr)
# displacementy=CorrectVelocityArray(velocityy,timearr)
# displacementz=CorrectVelocityArray(velocityz,timearr)

displacementx=GetDisplacementArray(velocityx,timearr)
displacementy=GetDisplacementArray(velocityy,timearr)
displacementz=GetDisplacementArray(velocityz,timearr)


[ax2,ay2,az2]=map(list,zip(*accelerationXYZ))
correctedax2=ApplyCorrections(ax2)
correcteday2=ApplyCorrections(ay2)
correctedaz2=ApplyCorrections(az2)

velocityx2=GetVeleocityArray(ax2,timearr)
velocityy2=GetVeleocityArray(ay2,timearr)
velocityz2=GetVeleocityArray(az2,timearr)

# displacementx=CorrectVelocityArray(velocityx,timearr)
# displacementy=CorrectVelocityArray(velocityy,timearr)
# displacementz=CorrectVelocityArray(velocityz,timearr)

displacementx2=GetDisplacementArray(velocityx2,timearr)
displacementy2=GetDisplacementArray(velocityy2,timearr)
displacementz2=GetDisplacementArray(velocityz2,timearr)


#Plotting Part

plt.figure(1)

plt.subplot(4,3,1)
plt.ylabel('Ax (m/s2)')
plt.plot(ax)

plt.subplot(4,3,2)
plt.ylabel('Ay (m/s2)')
plt.plot(ay)

plt.subplot(4,3,3)
plt.ylabel('Az (m/s2)')
plt.plot(az)

plt.subplot(4,3,4)
plt.ylabel('Corr Ax')
plt.plot(correctedax)

plt.subplot(4,3,5)
plt.ylabel('Corr Ay')
plt.plot(correcteday)

plt.subplot(4,3,6)
plt.ylabel('Corr Az')
plt.plot(correctedaz)

plt.subplot(4,3,7)
plt.ylabel('Vx (m/s)')
plt.plot(velocityx)

plt.subplot(4,3,8)
plt.ylabel('Vy (m/s)')
plt.plot(velocityy)

plt.subplot(4,3,9)
plt.ylabel('Vz (m/s)')
plt.plot(velocityz)

plt.subplot(4,3,10)
plt.ylabel('x (m)')
# plt.xlabel('Time : (100=1s)')
plt.plot(displacementx)

plt.subplot(4,3,11)
plt.ylabel('y (m)')
# plt.xlabel('Time : (100=1s)')
plt.plot(displacementy)

plt.subplot(4,3,12)
plt.ylabel('z (m)')
# plt.xlabel('Time : (100=1s)')
plt.plot(displacementz)


plt.figure(2)

plt.subplot(4,3,1)
plt.ylabel('Ax (m/s2)')
plt.plot(ax2)

plt.subplot(4,3,2)
plt.ylabel('Ay (m/s2)')
plt.plot(ay2)

plt.subplot(4,3,3)
plt.ylabel('Az (m/s2)')
plt.plot(az2)

plt.subplot(4,3,4)
plt.ylabel('Corr Ax')
plt.plot(correctedax2)

plt.subplot(4,3,5)
plt.ylabel('Corr Ay')
plt.plot(correcteday2)

plt.subplot(4,3,6)
plt.ylabel('Corr Az')
plt.plot(correctedaz2)

plt.subplot(4,3,7)
plt.ylabel('Vx (m/s)')
plt.plot(velocityx2)

plt.subplot(4,3,8)
plt.ylabel('Vy (m/s)')
plt.plot(velocityy2)

plt.subplot(4,3,9)
plt.ylabel('Vz (m/s)')
plt.plot(velocityz2)

plt.subplot(4,3,10)
plt.ylabel('x (m)')
# plt.xlabel('Time : (100=1s)')
plt.plot(displacementx2)

plt.subplot(4,3,11)
plt.ylabel('y (m)')
# plt.xlabel('Time : (100=1s)')
plt.plot(displacementy2)

plt.subplot(4,3,12)
plt.ylabel('z (m)')
# plt.xlabel('Time : (100=1s)')
plt.plot(displacementz2)


plt.show()