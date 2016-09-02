from operator import itemgetter,attrgetter
import math
points=[]
pointsX=[]
pointsY=[]
points=input()
temp = [map(float,x) for x in points]
points = temp
pointsX = sorted(points, key=itemgetter(0))
pointsY = sorted(points, key=itemgetter(1))
def distance(a, b):
	return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
def divideandconquer(left, right):
	mid = (left + right) / 2
	global pointsX
	global pointsY
	if left+2 == right:
		return min(distance(pointsX[left], pointsX[mid]),
		  distance(pointsX[mid], pointsX[right]), distance(pointsX[left], pointsX[right]))
	elif left+1 == right:
		return distance(pointsX[left], pointsX[right])
	elif left == right:
		return 100000.0
	else:
		leftmin = divideandconquer(left, mid-1)
		rightmin = divideandconquer(mid+1, right)
		curmin = min(leftmin, rightmin)
		acrossmin = merge(left, mid, right, curmin)
		return min(acrossmin, curmin)
		
def merge(left, mid, right, curmin):
	global pointsX
	global pointsY
	x = pointsX[mid][0]
	y = pointsX[mid][1]
	leftbound = x - curmin
	rightbound = x + curmin
	pointsYprime = []
	for point in pointsY:
		if point[0] >= leftbound and point[0] <= rightbound:
			pointsYprime.append(point)
	acrossmin = 1000000.0
	for i in range(0,len(pointsYprime)):
		curY=pointsYprime[i][1]
		for j in range(i+1,len(pointsYprime)):
			if pointsYprime[j][1]>curY+curmin:
				break
			else:
				tempdist=distance(pointsYprime[i],pointsYprime[j])
		                acrossmin=min(acrossmin,tempdist)
	return acrossmin	 			 
res = divideandconquer(0,len(points)-1)
print res	

	

		
		


