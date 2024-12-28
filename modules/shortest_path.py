import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np 
import seaborn as sns
import pandas as pd
import heapq
import random


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

def create_undirected_flight_network(working_df):

    """
    Creates an undirected graph from flight data with edge cost of 1.
    
    Args:
        working_df (pd.DataFrame): DataFrame with Origin_airport and Destination_airport columns
    Returns:
        nx.Graph: Undirected graph with weighted edges - cost of 1
    """
    G = nx.Graph()
    for row in tqdm(working_df.itertuples(), total=len(working_df)):
        origin_airport = row.Origin_airport
        destination_airport = row.Destination_airport
        G.add_edge(origin_airport, destination_airport, cost=1)
    return G

def minimum_cut_phase(G):

    """
    Implements one phase of the Stoer-Wagner algorithm. 

    Args:
        G (nx.Graph): The input graph
    Returns:
        tuple: (cut_weight, vertices_order, last_two)
    """
    # Start with first vertex - though any vertex can be chosen as the starting vertex
    a = random.choice(list(G.nodes()))
    # Vertices order is a list containing nodes in the order in which vertices are added to the set containing a. It is needed to reconstruct the cut
    vertices_order = [a]
    # added is a set containing vertices that have been added to the set containing a
    added = {a}
    # initialize a dictionary containing the weights of the vertices - by weight, we mean the sum of the cost of the edges connecting the vertex to the vertices in the set containing a
    weights = {v: 0 for v in G.nodes()}

    # Update weights for neighbors of a by retrieving the cost of the edge connecting them to a. All costs are set to 1.
    for neighbor in G.neighbors(a):
        weights[neighbor] = G[a][neighbor]['cost']
    
    # add remaining vertices to the set of vertices containing a
    while len(added) < len(G.nodes()):
        # Find most tightly connected vertex. Tightly connected vertex is the vertex with the highest weight
        # Initialize max_weight to negative infinity and next_vertex to None. next_vertex will store the vertex with the highest weight
        
        max_weight = -float('inf')
        next_vertex = None
        
        # for each vertices, if that vertex is not in the set containing a and the weight of that vertex is greater than max_weight, update max_weight and next_vertex
        for v in G.nodes():
            if v not in added and weights[v] > max_weight:
                max_weight = weights[v]
                next_vertex = v
        # break out of the while loop if no next vertex is found
        if next_vertex is None:
            break 
        
        # Add next_vertex to the set containing a and to the vertices_order list.
        vertices_order.append(next_vertex)
        added.add(next_vertex)

        # for the neighbors of next_vertex, update their weights by adding the cost of the edge connecting them to next_vertex
        for neighbor in G.neighbors(next_vertex):
            if neighbor not in added:
                weights[neighbor] += G[next_vertex][neighbor]['cost']
    
    # The last vertex in vertices_order is the last vertex added to the set containing a
    last_vertex = vertices_order[-1]
    # cut weight is the sum of the cost of the edges connecting the last vertex to the vertices in the set containing a
    cut_weight = sum(G[last_vertex][v]['cost'] for v in G.neighbors(last_vertex))
    
    return cut_weight, vertices_order, vertices_order[-2:]

def merge_vertices(G, merged_nodes, u, v):

    """
    Merges vertex v into vertex u.
    
    Args:
        G (nx.Graph): The input graph
        merged_nodes (dict): Dictionary tracking merged vertices
        u, v: Vertices to merge
    """
    # Remove self loops
    if G.has_edge(v, v):
        G.remove_edge(v, v)
    
    # merged_nodes keepst rack of the nodes that have been merged into a node.
    # keys are nodes and values are lists of nodes that have been merged into the key node.
    
    # Merge nodes - add all nodes from v to u
    merged_nodes[u].extend(merged_nodes[v])
    # Delete the merged node from the dictionary
    del merged_nodes[v]
    
    
    # for each neighbor of v, if the neighbor is not u, update the weight of the edge connecting u to the neighbor by adding the 
    # cost of the edge connecting v to the neighbor. This serves to merge the edges connecting v to the neighbors of v to u.

    for neighbor in list(G.neighbors(v)):
        if neighbor != u:
            w = G[v][neighbor]['cost']
            if G.has_edge(u, neighbor):
                G[u][neighbor]['cost'] += w
            else:
                G.add_edge(u, neighbor, cost=w)
    # finally, remove the node v from the graph
    G.remove_node(v)

def stoer_wagner_min_cut(G):

    """
    Computes the minimum cut using Stoer-Wagner algorithm. It makes use of the following helper functions:
    - minimum_cut_phase: Implements one phase of the Stoer-Wagner algorithm.
    - merge_vertices: Merges vertex v into vertex u.
    
    Args:
        G (nx.Graph): The input graph
    Returns:
        tuple: (min_cut_value, partition)
            min_cut_value: The minimum cut value - an int
            partition: The partition with the minimum cut - a list of nodes in one partition
    """
    
    # Create working copy
    G_work = G.copy()
    # Initialize merged nodes dictionary containing the nodes that have been merged
    merged_nodes = {node: [node] for node in G.nodes()}
    
    # Define a minimum cut value, initially set to infinity, 
    min_cut_value = float('inf')

    # Initialize the partition to an empty list. this will store the nodes in the partition with the minimum cut. partition 
    min_cut_partition = []
    
    # Continue until there are only two nodes left
    while len(G_work.nodes()) > 1:
        # Find the minimum cut for the current phase - the current phase is the phase where the last two nodes are merged
        cut_value, vertices_order, last_two = minimum_cut_phase(G_work)
        
        # Update the minimum cut value and partition if the current cut value is less than the minimum cut value
        if cut_value < min_cut_value:
            min_cut_value = cut_value
            # The min_cut_partition is the list of vertices in the set containing the last vertex in vertices_order
            min_cut_partition = merged_nodes[last_two[1]]
        
        # Merge the last vertex into the second to last vertex - the last two vertices are the last two vertices in vertices_order.
        merge_vertices(G_work, merged_nodes, last_two[0], last_two[1])
    
    return min_cut_value, min_cut_partition

def get_cut_edges(G, partition):
    """
    Finds edges crossing the cut.
    
    Args:
        G (nx.Graph): The input graph
        partition (list): List of nodes in one partition
    Returns:
        list: Edges crossing the cut
    """
    cut_edges = []
    for u, v in G.edges():
        if (u in partition) != (v in partition):
            cut_edges.append((u, v))
    return cut_edges

def visualize_network(G, title, partition=None, fig_size = (12, 8)):

    """
    Visualizes the flight network.
    
    Args:
        G (nx.Graph): The flight network
        title (str): Plot title
        partition (list, optional): List of nodes in one partition
        fig_size (tuple, optional): Figure size
    """
    # Set the figure size
    plt.figure(figsize=fig_size)

    # Compute layout. pos is a dictionary containing the positions of the nodes. spring_layout is used to compute the positions.
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    
    # Draw nodes with colors based on partition
    if partition:
        node_colors = ['firebrick' if node in partition else 'paleturquoise' 
                      for node in G.nodes()]
    else:
        node_colors = ['teal' for node in G.nodes()]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=500, alpha=0.6)
    
    # Draw labels on the nodes corresponding to the airport codes
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def disconnect_graph(G):
    """
    Disconnects the graph into two separate components by removing a minimum number of flights. Makes use of the following helper functions:
    - stoer_wagner_min_cut: Computes the minimum cut using Stoer-Wagner algorithm.
    - get_cut_edges: Finds edges crossing the cut.
    - visualize_network: Visualizes the flight network.
    
    Args:
        G (nx.Graph): An undirected graph containing flight data
    Returns:
        cut_edges: A list of edges that, if removed, would disconnect the graph into two separate components.
    """    
    # Visualize original network
    print("Visualizing original network...")
    visualize_network(G, "Original Flight Network")
    
    # Find minimum cut
    print("Finding minimum cut...")
    min_cut_value, partition = stoer_wagner_min_cut(G)
    
    # Get cut edges
    cut_edges = get_cut_edges(G, partition)
    
    # Print results
    print(f"\nMinimum number of flights to remove: {min_cut_value}")
    print("\nFlights to remove:")
    for u, v in cut_edges:
        print(f"{u} <-> {v}")
    
    # Create separated network
    G_cut = G.copy()
    G_cut.remove_edges_from(cut_edges)
    
    # Visualize result
    print("\nVisualizing partitioned network...")
    visualize_network(G_cut, "Partitioned Flight Network", partition)
    
    return cut_edges
