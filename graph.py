import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from IPython.display import display

# Custom DiGraph Class
from collections import defaultdict

class CustomDiGraph:
    def __init__(self):
        '''Initialization of an empty graph'''
        self.nodes = set()  # set to store nodes
        self.edges = defaultdict(dict)  # default dictionary to store edges with weights: key=node, value=dict of neighbors and weights

    def add_node(self, node):
        '''Function to add a node to the graph'''
        self.nodes.add(node)

    def add_edge(self, node1, node2, weight=1):
        '''Function that adds an outgoing edge from node1 to node2 with a specified weight (default is 1)'''
        self.add_node(node1)
        self.add_node(node2)
        self.edges[node1][node2] = weight

    def remove_node(self, node):
        '''Function that removes a node and all edges connected to it'''
        if node in self.nodes:
            self.nodes.remove(node)
            del self.edges[node]  # Remove the node from the edges dictionary
            # Remove any edges that point to this node
            for other_node in list(self.edges.keys()):
                if node in self.edges[other_node]:
                    del self.edges[other_node][node]

    def remove_edge(self, node1, node2):
        '''Function that removes an edge from node1 to node2'''
        if node2 in self.edges[node1]:
            del self.edges[node1][node2]

    def get_neighbors(self, node):
        '''Function that returns the neighbors of a node with weights'''
        return self.edges[node]

    def get_nodes(self):
        '''Function that returns a list of all nodes'''
        return list(self.nodes)

    def get_edges(self):
        '''Function that returns a set of all edges as tuples (node1, node2, weight)'''
        edge_set = set()
        for node1 in self.nodes:
            for node2, weight in self.edges[node1].items():
                edge_set.add((node1, node2, weight))
        return edge_set

    def has_edge(self, node1, node2):
        '''Function that returns True if there is an edge between node1 and node2'''
        return node2 in self.edges[node1]

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
        import networkx as nx
        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(node)
        for node1, neighbors in self.edges.items():
            for node2, weight in neighbors.items():
                G.add_edge(node1, node2, weight=weight)
        return G

    def num_nodes(self):
        '''Function that returns the number of nodes in the graph'''
        return len(self.nodes)

    def num_edges(self):
        '''Function that returns the number of edges in the graph'''
        return sum(len(neighbors) for neighbors in self.edges.values())



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