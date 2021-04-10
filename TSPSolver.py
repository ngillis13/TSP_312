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
	
	print (min)
	return index 


class subGroup:
	def __init__(self, citiesArray):
		self.citiesArray = citiesArray
		self.startNode = None
		self.endNode = None
		self.pathArray = ([])
		self.totalDistance = 0


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
		pass



	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy( self,time_allowance=60.0 ):

		# we start by taking each city and adding up all the paths connected to that city
		# then we sort the cities by this total distance with the largest distance being at 
		# the top of the list

		# we then take the first city (with largest distance) and sort all the cities it's connected
		# to (taking out the ones already in a group) by distance starting with the smallest. 

		# Start by adding the next closest node to our starting node. This is the beginning of our group.
		# To add to the group, we go through the list (of connected nodes sorted by distance) and find the 
		# node with the shortest cost that has an edge connecting to each of the current nodes in the group
		# Repeat the process until there are no nodes left that connect to all current nodes in the group and/or cap it at a certain length
		# Continue the process until each node is in a subgroup




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





		# now we should have start and end nodes within each subgroup. now, within each subgroup we use a 
		# algorithm to find the shorest path within the group, starting and ending at the start and end nodes
		# we include the path as an array within the grouping object

		# finally, we add the path together, adding the connections from group to group with the path arrays
		# within the groupings. we also count the cost in the same manner. 


		# subgroup_class:
			# array of cities (actual cities, not just an index)
			# start city
			# end city
			# path_array (actual cities, not just an index)
			# total distance 

		pass
		



