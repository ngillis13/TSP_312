#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools

#this funtion takes a matrix and recues it, first by row and then by col, to make sure that all have 0s, 
#and then adds up the cost. returnes the new matrix and the cost

def reduceMatrix(matrix, N):
	lowerBound = 0
	matrixCopy = np.copy(matrix)
	for i in range (N):
		min = matrixCopy[i].min()
		for j in range (N):
			matrixCopy[i][j] = matrixCopy[i][j] - min
			lowerBound += min

	minList = np.amin(matrixCopy, axis=0) 
	for i in range (len(minList)):
		min = minList[i]
		if (min != 0 and min != np.inf):
			for j in range (N):
				matrixCopy[j][i] = matrixCopy[j][i] - min
				lowerBound += min

	return matrixCopy, lowerBound

#this functions creates a new data object called state. it takes in a matrix, start node, end node, 
#size of one side of the matrix, the current lower bound, and the current path. it then calculates the 
#new matrix, new bound, and new path and makes a state object from the given information. 
#then returns the object

def makeState (matrix, start, end, N, lowerBound, path):
	lowerCopy = lowerBound
	matrixCopy = np.copy(matrix)
	newPath = path.copy()
	lowerCopy += matrixCopy[start][end]
	matrixCopy[end][start] = np.inf
	for i in range (N):
		matrixCopy[start][i] = np.inf
		matrixCopy[i][end] = np.inf

	newPath.append(end + 1)

	state = matrixState(matrixCopy, lowerCopy, newPath)

	return state

#this gets the BSSF for the first iteration (upper bound)

def getBSSF(matrix, N, lowerBound):
	sum = 0
	count = 0
	for i in range (N):
		for j in range (N):
			if (matrix[i][j] != np.inf):
				sum += matrix[i][j]
				count += 1
	dist = (sum/count) * N
	return (lowerBound + dist) * 2


def findShortestPath(city, remainingCities):

	i = 0
	min = city.costTo(remainingCities[0])
	minCity = remainingCities[0]
	index = i
	while (i < len(remainingCities)):
		if (city.costTo(remainingCities[i]) < min):
			min = city.costTo(remainingCities[i])
			minCity = remainingCities[i]
			index = i
		i += 1
	
	return index





def findGroup(city, Groups):
	for i in range (len(Groups)):
		tempArray = Groups[i].citiesArray
		for j in range(len(tempArray)):
			if (city == tempArray[j]):
				return Groups[i]

	return None

#an object to conatin the state. contains a matrix, a distance, and a path
class matrixState:
	def __init__(self, matrix, distance, path):
		self.matrix = matrix
		self.distance = distance
		self.path = path
	def getDistance():
		return self.distance

class subGroup:
	def __init__(self, citiesArray):
		self.citiesArray = citiesArray
		self.startNode = None
		self.endNode = None
		self.pathArray = ([])
		self.totalDistance = 0
		self.targetGroup = None
		self.name = citiesArray[0]._name
		
	def find_shortest_path(self):
		if len(self.citiesArray) == 1:
		    self.pathArray.append(self.citiesArray[0])
		elif len(self.citiesArray) == 2:
		    self.pathArray.append(self.startNode)
		    self.pathArray.append(self.endNode)
		else:
			tempArray = self.citiesArray.copy()
			tempArray.pop(tempArray.index(self.startNode))
			tempArray.pop(tempArray.index(self.endNode))
			self.pathArray.append(self.startNode)
			currentCity = self.startNode
			while len(tempArray) > 0:
				index = findShortestPath(currentCity, tempArray)
				currentCity = tempArray.pop(index)
				self.pathArray.append(currentCity)
			self.pathArray.append(self.endNode)


class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None
		self.cost = 0

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def greedy( self,time_allowance=60.0 ):
		
		results = {}
		start_time = time.time()
		unvisitedCities = []
		path = []
		i = 0
		self.cost = 0
		while i<len(self._scenario.getCities()):
			unvisitedCities.append(self._scenario.getCities()[i])
			i+=1
		currNode=self._scenario.getCities()[0]
		originNode = currNode
		#path.append(originNode)
		index = findShortestPath(currNode, unvisitedCities)
		self.cost += currNode.costTo(unvisitedCities[index])
		currNode = unvisitedCities.pop(index)
		path.append(currNode)
		while len(unvisitedCities) > 0:
			index = findShortestPath(currNode, unvisitedCities)
			self.cost += currNode.costTo(unvisitedCities[index])
			currNode = unvisitedCities.pop(index)
			path.append(currNode)
		
		end_time = time.time()

		bssf = TSPSolution(path)

		results['cost'] = self.cost
		results['time'] = end_time - start_time
		results['count'] = len(path)
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

		
		

	#set path-array = []
	#set all nodes to visited = false OR for all nodes set visited_array [node_id, false]
	#set curr_node to source_node 
	#shorest_path = find_shortest_path(curr_node)
	#curr_node = shorest_path.dest_id
	#path_array.append (curr_node)
	#set node = visited
#
	#while (curr_node != end_node) {
	#	shorest_path = find_shortest_path(curr_node)
	#	if (shorest_path != -1){
	#		curr_node = shorest_path.dest_id
	#		path_array.append (curr_node)
	#		set node = visited
	#	}
	#	else {
	#		path_found = false
	#		break
	#	}
#
	#}
	
	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
		
	def branchAndBound( self, time_allowance=60.0 ):
		#track fpr report
		maxSatesStored = 0
		totalStates = 1
		BSSFUpdates = 1
		statesPruned = 0

		optimal = True


		start_time = time.time()
		
		
		citiesList = self._scenario.getCities()

		N = len(citiesList)
		#number of solutions
		count = 0

		#get a number list to represent all cities
		allCities = ([])
		for i in range (N):
			allCities.append(i + 1)

		#make the first basic matrix
		baseMatrix = np.full((N, N), np.inf)

		for i in range (N):
			for j in range (N):
				baseMatrix[i][j] = citiesList[i].costTo(citiesList[j])

		#reduce basic matrix and lowerbound
		baseMatrix, lowerBound = reduceMatrix(baseMatrix, N)

		path = ([1])
		#cities left will be repopulated and recalculated every interation
		remainingCities = allCities.copy()
		remainingCities.remove(1)

		#create the first state
		baseSate = matrixState(baseMatrix, lowerBound, path)

		BSSF =  getBSSF(baseMatrix, N, lowerBound)
		bestPath = ([])
		queue = ([])

		#this keeps track of where you are going from in each itteration 
		currCity = path[len(path) - 1]

		for i in range (len(remainingCities)):
			
			#make a state for each possibale route 
			nextState = makeState(baseMatrix, currCity - 1, remainingCities[i] - 1, N, lowerBound, path)
			totalStates += 1
			#decide if you should prune state or not
			if (nextState.distance < BSSF):
				queue.append(nextState)
			else :
				statesPruned += 1
		#calculate max number of states stored
		if (len(queue) > maxSatesStored):
			maxSatesStored = len(queue)

		#store queue, and then trim it to only 500
		queue.sort(key=lambda x: x.distance)

		if (len(queue) > 500):
			queue = queue[:500]

		#now you just go until the queue is empty (you have seen all non-pruned states)
		#or until you get interupted (see below)
		while (len(queue) != 0):

			currState = queue.pop()
			currMatrix = currState.matrix
			currBound = currState.distance
			currPath = currState.path

			#find which cities have not been visited yet
			currRemCities = allCities.copy()
			for i in range (len(currPath)):
				currRemCities.remove(currPath[i])

			#if you have reached the end of a line, and all cities have been visited, check to see
			#the cost, if it's lower then BSSF, set new BSSF
			if (len(currPath) >= N):
				count += 1
				if (currBound < BSSF):
					BSSF = currBound
					bestPath = currPath
					BSSFUpdates += 1

			#if not the end of the line, keep going
			else:
				currCity = currPath[len(currPath) - 1]

				for i in range (len(remainingCities)):
					#once again, make new states for all that haven't been visited
					nextState = makeState(currMatrix, currCity - 1, remainingCities[i] - 1, N, currBound, currPath)
					totalStates += 1
					if (nextState.distance < BSSF):
						queue.append(nextState)
					else :
						statesPruned += 1

				if (len(queue) > maxSatesStored):
					maxSatesStored = len(queue)

				queue.sort(key=lambda x: x.distance)

				if (len(queue) > 500):
					queue = queue[:500]
			#here we make sure that we don't go over the given time. if we break, we set optimal to
			#false since we don't know if we found the optimal solution
			if (time.time() - start_time > time_allowance):
				optimal = False
				break

		#set the results
		
		end_time = time.time()
		results = {}
		results['cost'] = BSSF
		results['time'] = end_time - start_time
		results['count'] = count

		cities = []
		for i in range (len (bestPath)):
			cities.append(citiesList[bestPath[i] - 1])
		solution = TSPSolution(cities)

		results['soln'] = solution
		results['max'] = None
		results['total'] = None
		results['pruned'] = None

		#print for reporting 

		print("optimal solution = ", optimal)
		print ("max Sates Stored = ", maxSatesStored)
		print("BSSF Updates = ", BSSFUpdates)
		print("total States = ", totalStates)
		print("states Pruned = ", statesPruned)
		 
		return results
	
		
	def fancy( self,time_allowance=60.0 ):

		###############################################################################################################################################################################
		###############################################################################################################################################################################
		# PART 1: DALAN
		###############################################################################################################################################################################
		###############################################################################################################################################################################
		
		###############################################################################################################################################################################
		# PART 1.1:
		# we start by taking each city and adding up all the paths connected to that city
		# then we sort the cities by this total distance with the largest distance being at 
		# the top of the list
		###############################################################################################################################################################################
		
		start_time = time.time()
		# Here I initialize a cities list and I initialize a totalDistances list with each of those cities total edge distances
		cities = self._scenario.getCities()
		totalDistances = []
		pairing = {}
		for i in range (len(cities)):
			totalDistance = 0
			for j in range(len(cities)):
				if (i != j):
					if (cities[i].costTo(cities[j]) != math.inf):
						totalDistance += cities[i].costTo(cities[j])
			pairing[totalDistance] = cities[i]
			totalDistances.append(totalDistance)

		# Here I am just sorting the cities by distance and storing that in an array called sortedCities
		# I make a new sorted cities list that will take a sorted version of the distances list and become a list of the corresponding cities to those distances	
		sortedCities = []
		totalDistances.sort()
		for i in range(len(totalDistances)-1,-1,-1):
			sortedCities.append(pairing[totalDistances[i]])

		# I reverse the list here to make the sortedCities list go from longest total distance to shortest total distance
		sortedCities.reverse()

		###############################################################################################################################################################################
		# PART 1.2:
		# we then take the first city (with largest distance) and sort all the cities it's connected
		# to (taking out the ones already in a group) by distance starting with the sm
		###############################################################################################################################################################################

		#Here I Initialize a dictionary that will contain each node as a key and as the value it will have a list containing all connected nodes to that key sorted by distance (lowest to highest)
		sortedEdgeDistances = {}
		# I loop through each city
		for i in range(len(sortedCities)):
			
			currList = []
			edgeDistances = {}
			# I loop through each other city and if the distance isn't infinite, I store the city and length to reference later
			for j in range(len(sortedCities)):
				if (i != j):
					if (sortedCities[i].costTo(sortedCities[j]) != math.inf):
						edgeDistances[sortedCities[i].costTo(sortedCities[j])] = sortedCities[j]
						currList.append(sortedCities[i].costTo(sortedCities[j]))
			currList.sort()
			currSortedEdgeDistances = []
			# Here I sort and finalize the edge distances lost for the current node
			for j in range(len(currList)):
				currSortedEdgeDistances.append(edgeDistances[currList[j]])

			sortedEdgeDistances[sortedCities[i]] = currSortedEdgeDistances

		###############################################################################################################################################################################
		# PART 1.3:
		# Start by adding the next closest node to our starting node. This is the beginning of our group.
		# To add to the group, we go through the list (of connected nodes sorted by distance) and find the 
		# node with the shortest cost that has an edge connecting to each of the current nodes in the group
		# Repeat the process until there are no nodes left that connect to all current nodes in the group and/or cap it at a certain length
		# Continue the process until each node is in a subgroup
		###############################################################################################################################################################################
		
		# Here I make a list that keeps track of whether each node is in a group already. I initilize all values to false at first
		inGroup = {}
		groups = []
		for i in range(len(sortedCities)):
			inGroup[sortedCities[i]] = False
		
		# I then go through each of the cities (in order of the sortedCities list) and if the city isn't in a group yet, I create a new group and add all valid cities
		for i in range(len(sortedCities)):
			
			# Checking if the current city is in agroup yet
			if (inGroup[sortedCities[i]] == False):

				# Here I initialize a new group callde currGroup, I add the current city to the group and set the current city as in a group
				currGroup =[]
				currGroup.append(sortedCities[i])
				inGroup[sortedCities[i]] = True

				# I then loop through all of the connected cities to the current node and if they connect to all current cities in the group and aren't in agroup yet I add them to the group
				for j in range (len(sortedEdgeDistances[sortedCities[i]])):
					# Here I check if the node I'm looking at is currently in a group
					if (inGroup[sortedEdgeDistances[sortedCities[i]][j]] == False):
						isValid = True
						# Here I check if the node I am looking at connects to all other nodes currently in this group
						for k in range(len(currGroup)):
							if (sortedEdgeDistances[sortedCities[i]][j].costTo(currGroup[k]) == math.inf):
								isValid = False
						# If it does, I add the node I am looking at to the group and mark it as in a group
						if (isValid == True):
							currGroup.append(sortedEdgeDistances[sortedCities[i]][j])
							inGroup[sortedEdgeDistances[sortedCities[i]][j]] = True
				groups.append(subGroup(currGroup))


		###############################################################################################################################################################################
		# SUMMARY OF USEFUL VARIABLES:
		# 
		# sortedCities: A list containing each city sorted by the total distance of all that city's edges. This list is sorted from highest total distance to lowest total distance
		# sortedEdgeDistances: A dictionary where the keys are each city (in the same order as sortedCities) and the values are a list containing each connected city to that key node sorted from closest to farthest in terms of cost
		# groups: A list of lists where each internal list contains all the city objects in that group. Each internal list is one of our final groupings
		#
		###############################################################################################################################################################################


		###############################################################################################################################################################################
		###############################################################################################################################################################################
		# PART 2: NATHAN
		###############################################################################################################################################################################
		###############################################################################################################################################################################
		
		# Nate's Task:
		
		# Sort the subgroups by number of nodes

		# Starting with the subgroup with the fewest nodes, look at all outbound edges leading to nodes in
		# different groups (making sure that group hasn't been visited before). Choose the edge with the 
		# shortest distance, with a penelty for going to larger groups (for instance, times each cost by 1/2 of 
		# the number of cities in the group). 
		
		# The start and end of the edge we choose determines the enter and exit nodes
		# for these two subgroups

		# keep going until all subgroups have been visited. thus, with each subgroup being only visited once, 
		# each subgroup must have been visited, and a path now exist. there is the chance that not every subgroup
		# can be visited, but with the penelty included in the cost, we hope to avoid that

		
		
		# Sort the subgroups by number of nodes

		groups.sort(key=lambda x: len(x.citiesArray))

		# Starting with the subgroup with the fewest nodes, look at all outbound edges leading to nodes in
		# different groups (making sure that group hasn't been visited before). Choose the edge with the 
		# shortest distance, with a penelty for going to larger groups (for instance, times each cost by 1/2 of 
		# the number of cities in the group). 


		# these two arrays are to keep track of nodes we can't make a path to. visitedGroups tells us what groups 
		# already have a path leading to them. end cities tells us which cities already have a path leading to another
		# group. in other words, it tracks current end Nodes of all the groups, so we can't go to that city from 
		# another group/ we can't have the start node and the end node be the same. only exception is for groups
		# with only one node 

		vistedGroups = []
		endCities = []

		# here we go through each group for it to find it's path to another group
		for i in range(len(groups)):
			min = np.inf
			pathPair = [None, None]	# will be used to store the path nodes
			currGroup = groups[i]

			# itterate through the cities in the citiesArray contained in the group object
			for j in range (len(currGroup.citiesArray)):
				if (currGroup.citiesArray[j] != currGroup.startNode or len(currGroup.citiesArray) == 1):
					currCity = currGroup.citiesArray[j]
					pathOptions = sortedEdgeDistances[currCity]

					# use pathOptions here to itterate through every possible connection from each city in the group
					for k in range (len(pathOptions)):
						# use function findGroup to get the group object containg the target node
						# we are currently testing
						targetGroup = findGroup(pathOptions[k], groups)

						# here we need to make sure that the path we are testing isn't: 
						# 1. to a group that already has a path to it
						# 2. going back to it's own group
						# 3. isn't connecting to the end node of the targetGroup (use endCities array) unless only 
						# one city in group
						if ((targetGroup not in vistedGroups) 
						and targetGroup != currGroup 
						and (pathOptions[k] not in endCities or len(targetGroup.citiesArray) == 1)
						and targetGroup.targetGroup != currGroup):

							# here we set the penalty for larger groups, calculate the cost, and if it's lower then	
							# the current minimum, set it as the new min. we also set the pathPair values to 
							# the cities we are going from and to, and store what group it will be traversing to  
							penalty = (len(targetGroup.citiesArray) / 2)
							cost = currCity.costTo(pathOptions[k])
							if (cost * penalty < min):

								pathPair[0] = currCity	# end node of current group
								pathPair[1] = pathOptions[k] # start node of target group
								finalTargetGroup = targetGroup
								min = cost * penalty

			# once we have been through all possible solutions, we set our group vaiables				

			endCities.append(pathPair[0])	# append exit city to endCities for future use
			currGroup.endNode = pathPair[0]	# set the end node for our current group
			currGroup.targetGroup = finalTargetGroup	# set the targetGroup in our current group object
			finalTargetGroup.startNode = pathPair[1]	# set the start node for the target group
			vistedGroups.append(finalTargetGroup)	# append the target group to the visited groups array for 
													# future itterations

		# once this loop comletes each group should have:
		# 1. a cities array
		# 2. a start and end node
		# 3. a target group (mostly for debuging)
		# they will NOT yet have:
		# 1. pathArray
		# 2. totalDistance
		# use code below for debuging if dessired	

		# for debuging
		#for i in range (len(groups)):
		#	print (groups[i].targetGroup.name)
	


		###############################################################################################################################################################################


		###############################################################################################################################################################################
		###############################################################################################################################################################################
		# PART 3: KIMBALL
		###############################################################################################################################################################################
		###############################################################################################################################################################################
		# now we should have start and end nodes within each subgroup. now, within each subgroup we use a 
		# algorithm to find the shorest path within the group, starting and ending at the start and end nodes
		# we include the path as an array within the grouping object
		#
		# finally, we add the path together, adding the connections from group to group with the path arrays
		# within the groupings. we also count the cost in the same manner. 
		#
		#
		
		finalPath = []
		for i in range(len(groups)):
		    groups[i].find_shortest_path()


		currentGroup = groups[0]
		while (len(finalPath) < len(cities)):
			for i in range(len(currentGroup.pathArray)):
				finalPath.append(currentGroup.pathArray[i])
			currentGroup = currentGroup.targetGroup



		

		#tempArray = groups.copy()
		#currentGroup = tempArray[0]
		#while len(tempArray) > 0:
		#	for i in range(len(currentGroup.pathArray)):
		#		finalPath.append(currentGroup.pathArray[i])
#
		#	for i in range(len(tempArray)):
		#		if currentGroup.targetGroup == tempArray[i].startNode:
		#			currentGroup = tempArray[i]
		#			tempArray.pop(i)
		#			break

		for i in range (len(finalPath) - 1):
			test = finalPath[i].costTo(finalPath[i + 1])
			if (test == np.inf):
				print ("inf path: i = ", i)
		


		test = finalPath[len(finalPath) - 1].costTo(finalPath[0])

		bssf = TSPSolution(finalPath)

		end_time = time.time()

		###############################################################################################################################################################################

		results = {}
		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = None
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


