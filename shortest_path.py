import networkx as nx
import matplotlib.pyplot as plt
import tqdm
import numpy as np 
import seaborn as sns
import pandas as pd
import heapq

from modules.graph import *
from modules.utils import *


def create_flight_network(working_df):
    """
    Input:
    working_df: pd.DataFrame, the working dataframe
    
    Output:
    G: nx.DiGraph, the flight network
    
    About:
    This function creates a directed graph using the networkx library.
    The nodes of the graph are the airports and the edges are the flights between the airports.
    The graph has the following attributes:
    - Node attributes: city
    - Edge attributes: distance, date
    """
    G = nx.DiGraph()

    for row in tqdm.tqdm(working_df.itertuples(), total=len(working_df)):
        # Get the origin and destination airports
        origin_airport = row.Origin_airport
        destination_airport = row.Destination_airport
        
        # Add origin and destination nodes with their city as attributes
        G.add_node(origin_airport,city=row.Origin_city)
        G.add_node(destination_airport, city=row.Destination_city)
        
        # Add an edge with distance and date
        G.add_edge(origin_airport, destination_airport, 
                   distance=row.Distance,
                   date=row.Fly_date)
    
    return G

def create_city_flight_network(working_df):
    """
    Input:
    working_df: pd.DataFrame, the working dataframe
    
    Output:
    G: nx.DiGraph, the flight network
    
    About:
    This function creates a directed graph using the Class CustomDiGraph.
    The nodes of the graph are the cities and the edges are the flights between the cities.
    The graph has the following attributes:
    - Edge attributes: distance
    """
    G = CustomDiGraph()
    for row in tqdm.tqdm(working_df.itertuples(), total=len(working_df)):
        # Get the origin and destination airports
        origin_city = row.Origin_city
        destination_city = row.Destination_city
        
        # Add origin and destination nodes with their city as attributes
        G.add_node(origin_city)
        G.add_node(destination_city)
        
        # Add an edge with distance and date
        G.add_edge(origin_city, destination_city, 
                   weight=row.Distance)
    
    return G

def deploy_dijkstra(flight_network, source):
    """
    Input:
    flight_network: nx.DiGraph
    source: str, the origin airport code

    Output:
    shortest_paths: dict, key: destination airport code, value: shortest path from source to destination
    predecessors: dict, key: destination airport code, value: predecessor of the destination airport

    About:
    This function computes the shortest paths from a source airport to all other airports in the network using Dijkstra's algorithm.
    """
    # Step 1: Initialization 
    # Initialise the shortest_paths with source node set to a distance of 0 and other nodes set to infinity
    shortest_paths = {node: float('inf') for node in flight_network.nodes}
    shortest_paths[source] = 0
    # Initialize the predecessors dictionary
    predecessors = {node: None for node in flight_network.nodes}
    # Create a priority queue and add the source node with priority 0
    pq = [(0, source)]
    # Convert the priority queue into a heap so that we can extract the node with the smallest distance at each step
    heapq.heapify(pq)

    # Step 2: Main loop
    while len(pq) > 0:
        # Extract the node with the smallest distance using heappop, which returns a tuple (distance, node)
        current_distance, current_node = heapq.heappop(pq)
        # If the current distance is larger than the shortest path to the current node
        if current_distance > shortest_paths[current_node]:
            continue # Skip the rest of the loop
        
        # Now, for each neighbor of the current node, we check if the distance to the neighbor can be shortened
        for neighbor in flight_network.neighbors(current_node):
            # Compute the distance to the neighbor using the distance attribute of the edge between the current node and the neighbor
            distance = flight_network[current_node][neighbor]['distance']
            # Compute the new distance
            new_distance = current_distance + distance
            # If the new distance is smaller than the shortest path to the neighbor
            if new_distance < shortest_paths[neighbor]:
                # Update the shortest path to the neighbor
                shortest_paths[neighbor] = new_distance
                # Update the predecessor of the neighbor
                predecessors[neighbor] = current_node
                # Add the neighbor to the priority queue with the new distance as priority
                heapq.heappush(pq, (new_distance, neighbor))

    return shortest_paths, predecessors

def get_graph_for_a_date(G, date):
    """
    Inputs:
    G: (nx.DiGraph): Original graph.
    date: (str): Date to filter edges.

    Output:
    graph_for_a_date: (nx.DiGraph): Filtered graph.

    About:
    Filters the graph to include only flights available on the given date.
    """
    # Create a copy of the original graph
    graph_for_a_date = G.copy()
    # Iterate over all edges in the graph
    for u, v, data in list(G.edges(data=True)):
        # If the date of the flight is not the given date, remove the edge
        if data['date'] != date:
            # Remove the edge from the graph so that it is not considered in the shortest path computation
            graph_for_a_date.remove_edge(u, v)
    return graph_for_a_date

def reconstruct_path(predecessors, source, destination):
    """
    Inputs:
    - predecessors (dict): Mapping of each node to its predecessor.
    - source (str): The source node.
    - destination (str): The destination node.

    Output:
    path: (list): List of nodes representing the shortest path.

    About:
    Reconstructs the shortest path from source to destination.
    """
    # Initialize an empty path - this will be the shortest path from the source to the destination
    path = [] 
    # Set the current node to the destination - we will work backwards from the destination to the source
    current_node = destination
    # Loop until we get to the source node (whose the predecessor is None)
    while current_node is not None:
        path.append(current_node)
        current_node = predecessors[current_node]
    # If the last node in the path is the source, reverse the path and return it
    if path[-1] == source:
        path.reverse() 
        return path
    # If the last node is not the source, there is no path from source to destination
    return [] 

def get_shortest_paths_for_a_date(flight_network, origin_city_name, destination_city_name, date):
    """
    Inputs:
    flight_network: nx.DiGraph, the flight network
    origin_city_name: str, the name of the origin city
    destination_city_name: str, the name of the destination city
    date: str, the date of the flight

    Output:
    df: pd.DataFrame, the table with the best routes

    About:
    This function computes the best routes between all possible airport pairs between the origin and destination cities on a given date.
    """
    # Get the graph for the given date
    #print(f"Non-Filtered graph has {len(flight_network.nodes)} nodes and {len(flight_network.edges)} edges.") # a debug print statement
    flight_network = get_graph_for_a_date(flight_network, date)
    #print(f"Filtered graph has {len(flight_network.nodes)} nodes and {len(flight_network.edges)} edges.") # a debug print statement
    # Initialize an empty list to store the results
    results = []
    # Get all the airports in the origin city
    origin_airports = [node for node, data in flight_network.nodes(data=True) if data['city'] == origin_city_name]
    # Get all the airports in the destination city
    destination_airports = [node for node, data in flight_network.nodes(data=True) if data['city'] == destination_city_name]
    #print(f"Origin airports: {origin_airports}") # Debug: Check origin and destination airports
    #print(f"Destination airports: {destination_airports}") # Debug: Check origin and destination airports
    # Loop over all pairs of origin and destination airports
    for origin_airport in origin_airports:
        for destination_airport in destination_airports:
            # Compute the shortest paths and predecessors using Dijkstra's algorithm
            shortest_paths, predecessors = deploy_dijkstra(flight_network, origin_airport)
            # Reconstruct the path from the origin airport to the destination airport
            path = reconstruct_path(predecessors, origin_airport, destination_airport)
            # If the path is not empty, add it to the results
            if len(path) > 0:
                # Calculate the total distance for the path
                total_distance = sum(
                    # because there exists a path, calculate the total distance by summing the distances between each pair of nodes
                    flight_network[path[i]][path[i + 1]]['distance']
                    for i in range(len(path) - 1)
                )
                # Add the result with the total distance
                results.append([origin_airport, destination_airport, '->'.join(path), total_distance])
            else:
                # No route found
                results.append([origin_airport, destination_airport, 'No route found', None])
    
    # Create a DataFrame from the results
    df = pd.DataFrame(results, columns=['Origin_city_airport', 'Destination_city_airport', 'Best_route', 'Total_distance'])
    return df