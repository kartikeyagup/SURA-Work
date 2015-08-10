import csv

f1='1439189804924_corrpairs.csv'

fileread=[]
with open(f1,'rb') as csvfile:
	spamreader= csv.reader(csvfile)
	for row in spamreader:
		fileread.append(row)

fform =map(lambda x: map(float,x),fileread)

def getPoints(l):
	ans=[]
	for i in xrange(len(l)/2):
		ans.append((l[2*i],l[2*i+1]))
	return ans

fformed = map(getPoints, fform)

def getpairsfrom2(l1,l2,camframe):
	ans=[]
	for i in xrange(len(l1)):
		ans.append([camframe,l1[i],camframe+1,l2[i]])
	return ans

def getpairsof(l):
	bigans=[]
	for i in xrange(len(l)/2):
		bigans.append(getpairsfrom2(l[2*i],l[2*i+1],i))
	return bigans

def GetMoreCorres(prevcorr,newcorr):
	previndex=0
	for i in xrange(len(newcorr)):
		present=newcorr[i]
		while (previndex<len(prevcorr)):
			if (present[0]==prevcorr[previndex][-2]):
				if present[1]==prevcorr[previndex][-1]:
					prevcorr[previndex]+=[present[2],present[3]]
					break
			previndex+=1
		if (previndex>=len(prevcorr)):
			prevcorr+=newcorr[i:]
			break
	return prevcorr

paired = getpairsof(fformed)

pr1=paired[0]
for i in xrange(1,len(paired)):
	pr1=GetMoreCorres(pr1, paired[i])

print pr1

# print paired
# print fformed[0]