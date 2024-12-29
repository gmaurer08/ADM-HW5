import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from IPython.display import display
from heapq import heappop, heappush  # Import heap functions for priority queue operations
from itertools import count  # Import count to generate unique sequence numbers
import random
from modules.shortest_path import deploy_dijkstra
from folium.plugins import MarkerCluster

# Custom DiGraph Class
class CustomDiGraph:
    def __init__(self):
        '''Initialization of an empty graph'''
        self.nodes = set()  # set to store nodes
        self.edges = defaultdict(set) # default dictionary to store edges: key=node, value=set of neighbors

    def add_node(self, node):
        '''Function to add a node to the graph'''
        self.nodes.add(node)

    def add_edge(self, node1, node2):
        '''Function that adds an outgoing edge from node1 to node2'''
        # Add node1 and node2 to set of nodes in case they are not already in it
        self.add_node(node1)
        self.add_node(node2)
        self.edges[node1].add(node2) # add node2 to set of neighbors of node1

    def remove_node(self, node):
        '''Function that removes a node and all edges connected to it'''
        if node in self.nodes:
            self.nodes.remove(node)
            del self.edges[node]  # Remove the node from the edges dictionary
            # Remove any edges that point to this node
            for other_node in list(self.edges.keys()):
                self.edges[other_node].discard(node)

    def remove_edge(self, node1, node2):
        '''Function that removes an edge from node1 to node2'''
        self.edges[node1].discard(node2)

    def remove_all_edges(self, node1, node2):
        '''Function that removes any edge between node1 and node2'''
        self.edges[node1].discard(node2)
        self.edges[node2].discard(node1)
    
    def get_neighbors(self, node):
        '''Function that returns the neighbors of a node that can be reached'''
        return self.edges[node]

    def get_nodes(self):
        '''Function that returns a list of all nodes'''
        return list(self.nodes)

    def get_edges(self):
        '''Function that returns a set of all edges as tuples (node1, node2)'''
        edge_set = set()
        for node1 in self.nodes:
            for node2 in self.get_neighbors(node1):
                edge_set.add((node1, node2))
        return edge_set
    
    def has_edge(self, node1, node2):
        '''Function that returns True if there is an edge between node1 and node2'''
        return node2 in self.get_neighbors(node1)

    def out_degree(self, node):
        '''Returns the out-degree of a node (number of outgoing edges)'''
        if node in self.edges:
            return len(self.edges[node])
        else:
            return 0
    
    def in_degree(self, node):
        '''Returns the in-degree of a node (number of incoming edges)'''
        num_in_edges = 0
        for neighbor in self.nodes:
            if node in self.edges[neighbor]:
                num_in_edges += 1
        return num_in_edges

    def __str__(self):
        '''Function that returns a string representation of the graph'''
        return f'Nodes: {self.nodes}\nEdges: {self.get_edges()}'
    
    def to_networkx_digraph(self):
        '''Function that converts the custom graph to a NetworkX DiGraph'''
        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(node)
        for node1, neighbors in self.edges.items():
            for node2 in neighbors:
                G.add_edge(node1, node2)
        return G
    
    def num_nodes(self):
        '''Function that returns the number of nodes in the graph'''
        return len(self.nodes)
    
    def num_edges(self):
        '''Function that returns the number of edges in the graph'''
        return len(self.get_edges())



def analyze_graph_features(flight_network):
    '''
    Function that, given a directed input graph, computes:
    - Number of airports (nodes) and flights (edges) in the graph
    - Density of the (directed) graph
    - in-degree dictionary with key: node, value: flight_network.in_degree(node)
    - out-degree dictionary with key: node, value: flight_network.out_degree(node)
    - Hubs, airports with in-degrees or out-degrees higher than the respective 90th percentile
    - Whether the graph is sparse or dense
    Input:
    - flight_network(CustomDiGraph): custom directed graph of the flight network
    Output:
    - graph_features (dict): dictionary with information about the graph
    '''
    # Count number of airports (nodes)
    num_airports = flight_network.num_nodes()

    # Count number of flights (edges)
    num_flights = flight_network.num_edges()

    # Calculate density
    density = num_flights / (num_airports*(num_airports-1))

    # Calculate in-degrees
    in_degrees = {node: flight_network.in_degree(node) for node in flight_network.get_nodes()}

    # Calculate out-degrees
    out_degrees = {node: flight_network.out_degree(node) for node in flight_network.get_nodes()}

    # Plot histograms of in-degrees and out-degrees
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.hist(in_degrees.values(), bins=30, color='lightblue', edgecolor='black')
    plt.xlabel('In-degree')
    plt.ylabel('Frequency')
    plt.title('In-degree Distribution')
    plt.subplot(1, 2, 2)
    plt.hist(out_degrees.values(), bins=30, color='orange', edgecolor='black')
    plt.xlabel('Out-degree')
    plt.ylabel('Frequency')
    plt.title('Out-degree Distribution')
    plt.tight_layout()

    # Calculate 90th percentile of in_degrees
    percentile_90_in_degrees = np.percentile(list(in_degrees.values()), 90)

    # Calculate 90th percentile of out_degrees
    percentile_90_out_degrees = np.percentile(list(out_degrees.values()), 90)

    # Find airports with in-degree greater than 90th percentile
    in_degree_hubs = [node for node in flight_network.get_nodes() if flight_network.in_degree(node) > percentile_90_in_degrees]
    out_degree_hubs = [node for node in flight_network.get_nodes() if flight_network.out_degree(node) > percentile_90_out_degrees]

    # Determine hubs as airports with high in-degree or out-degree
    hubs = list(set(in_degree_hubs).union(set(out_degree_hubs)))

    # Determine if the graph is dense or sparse based on the density
    threshold = 0.1
    dense = False
    if density > threshold:
        dense = True
    
    print(f'Number of airports (nodes): {num_airports}')
    print(f'Number of flights (edges): {num_flights}')
    print(f'Graph Density: {density}')
    print(f'Graph is dense: {dense}')
    print('Hubs:')
    print(hubs)
    print('Degree distributions: ')
    plt.show()

    # Create a dictionary with information do return
    graph_features = {
        'num_airports': num_airports,
        'num_flights': num_flights,
        'in_degrees': in_degrees,
        'out_degrees': out_degrees,
        'density': density,
        'hubs': hubs
    }

    return graph_features


def summarize_graph_features(flight_network):
    '''
    Function that, given a flight network, writes a report about the graph features, including:
    - Number of nodes (airports)
    - Number of edges (flights)
    - Graph density
    - Degree distribution plots for in-degree and out-degree
    - Table of identified hubs
    Inputs:
    - flight_network (CustomDiGraph): custom directed graph of the flight network
    Outputs:
    - graph_summary (dict): dictionary with summary information about the graph
    '''

    # Count number of airports (nodes)
    num_airports = flight_network.num_nodes()

    # Count number of flights (edges)
    num_flights = flight_network.num_edges()

    # Calculate density
    density = num_flights / (num_airports*(num_airports-1))

    # Print number of nodes and edges
    print(f'The network contains {num_airports} nodes and {num_flights} edges.')

    # Determine if the graph is dense or sparse based on the density, and print the result
    threshold = 0.1
    if density > threshold:
        print(f'The graph is dense with density={density}')
    else:
        print(f'The graph is not dense and has density={density}')

    # Calculate in-degrees, store them in a defaultdict
    in_degrees = defaultdict(int, {node: flight_network.in_degree(node) for node in flight_network.get_nodes()})

    # Calculate out-degrees, store them in a defaultdict
    out_degrees = defaultdict(int, {node: flight_network.out_degree(node) for node in flight_network.get_nodes()})

    # Make a table with in_degrees and out_degrees for each airport
    airport_degree_data = []
    for node in flight_network.get_nodes():
        airport_degree_data.append((node,in_degrees[node],out_degrees[node]))

    # Create table of in_degrees and out_degrees for each airport
    degree_df = pd.DataFrame(airport_degree_data, columns=['airport', 'in-degree', 'out-degree'])

    # Display in-degree and out-degree
    print('Here is a table of in-degrees and out-degrees for every airport:')
    display(degree_df) 

    # Calculate 90th percentile of in_degrees
    percentile_90_in_degrees = np.percentile(list(in_degrees.values()), 90)

    # Calculate 90th percentile of out_degrees
    percentile_90_out_degrees = np.percentile(list(out_degrees.values()), 90)

    # Find airports with in-degree greater than 90th percentile
    in_degree_hubs = [node for node in flight_network.get_nodes() if flight_network.in_degree(node) > percentile_90_in_degrees]
    out_degree_hubs = [node for node in flight_network.get_nodes() if flight_network.out_degree(node) > percentile_90_out_degrees]

    # Determine hubs as airports with high in-degree or out-degree
    hubs = list(set(in_degree_hubs).union(set(out_degree_hubs)))

    # Create a table of hubs with their respective in-degrees and out-degrees
    hub_degree_df = degree_df[degree_df['airport'].isin(hubs)]
    hub_degree_df = hub_degree_df.rename(columns={'airport':'hub'})
    
    # Display hubs
    print('The following table contains the hubs, airports with in-degree or out-degree above the')
    print(f'90th percentile of in-degrees ({round(percentile_90_in_degrees,2)}) and out-degrees ({round(percentile_90_out_degrees,2)}) respectively:')
    display(hub_degree_df)

    print('In-degree and out-degree distributions:')

    # Plot histograms of in-degrees and out-degrees
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.hist(in_degrees.values(), bins=30, color='lightblue', edgecolor='black')
    plt.xlabel('In-degree')
    plt.ylabel('Frequency')
    plt.title('In-degree Distribution')
    plt.subplot(1, 2, 2)
    plt.hist(out_degrees.values(), bins=30, color='orange', edgecolor='black')
    plt.xlabel('Out-degree')
    plt.ylabel('Frequency')
    plt.title('Out-degree Distribution')
    plt.tight_layout()
    
    # Create a dictionary with information do return
    graph_summary = {
        'num_airports': num_airports,
        'num_flights': num_flights,
        'in_degrees': in_degrees,
        'out_degrees': out_degrees,
        'density': density,
        'hubs': hubs,
        'degree_df': degree_df,
        'hub_degree_df': hub_degree_df
    }

    return graph_summary

def total_passenger_flow(df):
    '''
    Function that, given the network DataFrame, computes the total passenger flow
    of each route.
    Input:
    - df (pd.DataFrame): data frame of the flight network
    Output:
    - city_route_flow (dict): dictionary of the passenger flows of routes
    '''
    # Create defaultdict to store passenger flows between origin and destination cities
    city_route_flow = defaultdict(int)
    
    # Iterate over rows in the df, add passenger flow
    for idx in range(df.shape[0]):
        origin = df['Origin_city'].iloc[idx]
        destination = df['Destination_city'].iloc[idx]
        # Add the min between number of passengers and seats
        if df['Seats'].iloc[idx] != 0 and df['Passengers'].iloc[idx] !=0:
            city_route_flow[(origin, destination)] += min(df['Passengers'].iloc[idx], df['Seats'].iloc[idx])
    
    return city_route_flow


def degree_centrality(G):
    '''
    Function that calculates the degree centrality of a node based on the number of neighbors
    Inputs:
    - G (nx.DiGraph): graph of the flight network
    Outputs:
    - dict: dictionary of degree centralities
    '''
    degree_centralities = {node: G.degree(node) / (G.number_of_nodes()-1) for node in G.nodes}
    return degree_centralities

def closeness_centrality(G):
    '''
    Function that calculates the closeness centrality of a node
    Inputs:
    - G (nx.DiGraph): graph of the flight network
    Outputs:
    - dict: dictionary of closeness centralities
    '''
    # Initialize dictionary of closeness centralities
    closeness_centralities = defaultdict()

    for node in G.nodes:
        # Compute distances from the source node
        distances, _ = deploy_dijkstra(G, node)
        
        # Filter reachable nodes
        reachable = [dist for dist in distances.values() if dist < float('inf')]
        
        # If only the node itself is reachable, centrality is 0
        if len(reachable) <= 1:
            closeness_centralities[node] = 0.0
        
        # Compute the total shortest path distances
        totsp = sum(reachable)
        
        # Normalize by the size of the reachable component
        reachable_count = len(reachable)
        normalization = (reachable_count-1) / (G.number_of_nodes()-1)
    
        # Closeness centrality formula
        closeness_centralities[node] = (reachable_count-1) / totsp * normalization if totsp!=0 else 0.0

    return closeness_centralities

def betweenness_centrality(G):
    '''
    Calculate the betweenness centrality for a directed, weighted G.

    Parameters:
        G (dict): A directed, weighted G represented as an adjacency list.

    Returns:
        dict: A dictionary with nodes as keys and their betweenness centrality as values.
    '''

    # Initialize betweenness centrality for each node to 0.0
    centrality = dict.fromkeys(G, 0.0)
    all_nodes = G  # Extract all nodes from the G

    # Iterate over all nodes in the G to calculate their contributions to centrality
    for start_node in all_nodes:

        # Dijkstra's algorithm setup
        visited_stack = []  # Stack to keep track of the nodes visited in order
        predecessors = {}  # Dictionary to store predecessors of each node
        for node in G:
            predecessors[node] = []

        path_count = dict.fromkeys(G, 0.0)  # Initialize path counts for each node
        shortest_distances = {}  # Dictionary to store shortest distances from the start node
        path_count[start_node] = 1.0  # There's one path to the start node itself

        # Priority queue for nodes to be explored; stores (distance, unique ID, predecessor, current node)
        push = heappush
        pop = heappop
        seen_distances = {start_node: 0}  # Dictionary to track the minimum distance seen for each node
        node_counter = count()  # Unique sequence numbers for heap operations
        priority_queue = []
        push(priority_queue, (0, next(node_counter), start_node, start_node))

        # Process the priority queue until empty
        while priority_queue:
            (current_distance, _, from_node, current_node) = pop(priority_queue)

            # Skip processing if the node is already finalized
            if current_node in shortest_distances:
                continue

            # Update the number of shortest paths to the current node
            path_count[current_node] += path_count[from_node]
            visited_stack.append(current_node)  # Add the current node to the stack

            # Finalize the shortest distance to the current node
            shortest_distances[current_node] = current_distance

            # Explore neighbors of the current node
            for neighbor, _ in G[current_node].items():
                distance_to_neighbor = current_distance + G.edges[current_node, neighbor]['distance']

                # If a shorter path to the neighbor is found, update the priority queue and path counts
                if neighbor not in shortest_distances and (neighbor not in seen_distances or distance_to_neighbor < seen_distances[neighbor]):
                    seen_distances[neighbor] = distance_to_neighbor
                    push(priority_queue, (distance_to_neighbor, next(node_counter), current_node, neighbor))
                    path_count[neighbor] = 0.0
                    predecessors[neighbor] = [current_node]

                # If another shortest path to the neighbor is found, update path counts and predecessors
                elif distance_to_neighbor == seen_distances[neighbor]:
                    path_count[neighbor] += path_count[current_node]
                    predecessors[neighbor].append(current_node)

        # Accumulate dependencies for betweenness centrality
        dependencies = dict.fromkeys(visited_stack, 0)  # Initialize dependency for each node in the stack
        while visited_stack:
            current_node = visited_stack.pop()  # Pop nodes in reverse order of finishing times
            coefficient = (1 + dependencies[current_node]) / path_count[current_node]
            for from_node in predecessors[current_node]:
                dependencies[from_node] += path_count[from_node] * coefficient

            # Accumulate betweenness centrality for nodes other than the start node
            if current_node != start_node:
                centrality[current_node] += dependencies[current_node]

    # Normalize the betweenness centrality values
    total_nodes = len(G)  # Total number of nodes in the G
    if total_nodes <= 2:
        scale_factor = None  # No normalization if there are less than 3 nodes
    else:
        scale_factor = 1 / ((total_nodes - 1) * (total_nodes - 2))

    if scale_factor is not None:
        for node in centrality:
            centrality[node] *= scale_factor

    return centrality

def pagerank(G, node=None, a=0.5, seed=42, T=100000):
    '''
    Function that calculates the betweenness centrality of a node
    Inputs:
    - G (nx.DiGraph): graph of the flight network
    - node (str): starting node
    - a (int): parameter in [0,1]
    - seed (int): random seed
    - T (int): nuber of steps
    Outputs:
    - int: betweenness centrality of the node
    '''
    if seed is not None:
        random.seed(seed)  # Set random seed for reproducibility

    t = 1
    if node is not None:
        current = random.choice(list(G.nodes)) # starting node
    else:
        current = node  # Starting node
    freq = defaultdict(int)  # initialize frequency counts

    # Random walk through the graph
    while t <= T:
        coin_flip = random.choices([0, 1], weights=[1 - a, a], k=1)[0]  # flip a coin

        if coin_flip == 0:  # follow a random out-neighbor
            successors = list(G.successors(current))
            if successors:  # Avoid errors in case the node has no successor
                current = random.choice(successors)
            else:  # If there are no out-neighbors, teleport
                current = random.choice(list(G.nodes))
        else:  # teleport to a random node
            current = random.choice(list(G.nodes))

        freq[current] += 1
        t += 1

    # Normalize frequencies to compute PageRank
    Pagerank = {u: freq[u] / T for u in G.nodes}

    return Pagerank  # Return the full PageRank distribution


def analyze_centrality(flight_network, airport):
    '''
    Function that computes betweenness, closeness, degree and pagerank centralities for a given airport in the flight_network
    Inputs:
    - flight_network (nx.DiGraph): graph of the flight network
    - airport (str): name of the airport
    Outputs:
    - None
    '''
    # Display centrality results for the given airport
    print(f"\n--- Centrality Measures for Airport: {airport} ---")
    print(f"Betweenness Centrality: {betweenness_centrality(flight_network)[airport]:.4f}")
    print(f"Closeness Centrality: {closeness_centrality(flight_network)[airport]:.4f}")
    print(f"Degree Centrality: {degree_centrality(flight_network)[airport]:.4f}")
    print(f"PageRank: {pagerank(flight_network)[airport]:.4f}")

def compare_centralities(flight_network):
    '''
    Function that compares betweenness, closeness, degree and pagerank centralities for all airports in the flight_network
    Inputs:
    - flight_network (nx.DiGraph): graph of the flight network
    Outputs:
    - None
    '''
    # Compute centrality measures
    betweenness = betweenness_centrality(flight_network)
    closeness = closeness_centrality(flight_network)
    degree_centr = degree_centrality(flight_network)
    pagerank_centr = pagerank(flight_network)

    # Plot histograms for centrality distributions
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.hist(betweenness.values(), bins=20, color='skyblue')
    plt.title('Betweenness Centrality')

    plt.subplot(2, 2, 2)
    plt.hist(closeness.values(), bins=20, color='lightgreen')
    plt.title('Closeness Centrality')

    plt.subplot(2, 2, 3)
    plt.hist(degree_centr.values(), bins=20, color='lightcoral')
    plt.title('Degree Centrality')

    plt.subplot(2, 2, 4)
    plt.hist(pagerank_centr.values(), bins=20, color='lightskyblue')
    plt.title('PageRank')

    plt.tight_layout()
    plt.show()

    # Identify top 5 nodes for each centrality measure
    print("\n--- Top 5 Airports by Centrality Measures ---")
    top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
    top_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]
    top_degree = sorted(degree_centr.items(), key=lambda x: x[1], reverse=True)[:5]
    top_pagerank = sorted(pagerank_centr.items(), key=lambda x: x[1], reverse=True)[:5]

    print("\nTop 5 by Betweenness Centrality:")
    for airport, value in top_betweenness:
        print(f"{airport}: {value:.4f}")

    print("\nTop 5 by Closeness Centrality:")
    for airport, value in top_closeness:
        print(f"{airport}: {value:.4f}")

    print("\nTop 5 by Degree Centrality:")
    for airport, value in top_degree:
        print(f"{airport}: {value:.4f}")

    print("\nTop 5 by PageRank:")
    for airport, value in top_pagerank:
        print(f"{airport}: {value:.4f}")


def eigenvector_centrality(flight_network, max_iter=1000, tol=1e-6):
    '''
    Function that computes eigenvector centrality of the flight_network
    Inputs:
        graph (nx.DiGraph or nx.Graph): input graph
        max_iter (int): max number of iterations
        tol (float): tolerance for convergence
    Outputs:
        dict: dictionary with nodes as keys and centrality scores as values
    '''
    # Convert the graph to an adjacency matrix
    adj_matrix = nx.to_numpy_array(flight_network)
    n = adj_matrix.shape[0]
    
    # Start with uniform centrality
    centrality = np.ones(n)
    centrality = centrality / np.linalg.norm(centrality)  # normalize 
    
    # Iterate until max_iter is reached
    for _ in range(max_iter):
        # update centrality
        new_centrality = np.dot(adj_matrix, centrality)
        # Normalize to prevent overflow or underflow
        new_centrality = new_centrality / np.linalg.norm(new_centrality)
        
        # Stop if convergence is reached within a tolerance
        if np.linalg.norm(new_centrality - centrality) < tol:
            break
        
        # update centrality
        centrality = new_centrality
    
    # map centrality values back to node labels
    centrality_dict = {node: score for node, score in zip(flight_network.nodes, centrality)}
    
    return centrality_dict


def analyze_eigenvector_centrality(flight_network):
    '''
    Function that analyzes the eigenvector centrality
    Inputs:
    - flight_network (nx.DiGraph): graph of the flight network
    Outputs:
    - None
    '''
    # Compute Eigenvector Centrality
    try:
        eigen_centrality = eigenvector_centrality(flight_network, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        print("Eigenvector centrality did not converge.")
        return

    # Display top 5 nodes by Eigenvector Centrality
    top_eigenvector = sorted(eigen_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\n--- Top 5 Airports by Eigenvector Centrality ---")
    for airport, value in top_eigenvector:
        print(f"{airport}: {value:.4f}")

    # Plot Eigenvector Centrality Distribution
    plt.figure(figsize=(8, 6))
    plt.hist(eigen_centrality.values(), bins=20, color='purple')
    plt.title('Eigenvector Centrality Distribution')
    plt.xlabel('Eigenvector Centrality')
    plt.ylabel('Frequency')
    plt.show()
