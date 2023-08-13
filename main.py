import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from colour.algebra import euclidean_distance
# import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import math

COLORS = ['r', 'b', 'g', 'brown', 'pink', 'gold', 'gray', 'silver', 'y', 'purple', 'magenta', 'orange', 'azure', 'green', 'teal', 'aqua', 'lime', 'olive', 'maroon', 'navy', 'yellow', 'violet', 'indigo', 'black', 'white', 'coral', 'plum', 'salmon', 'peach', 'cyan', 'tan', 'beige', 'khaki', 'ivory', 'lavender', 'turquoise', 'sienna', 'orchid', 'slate', 'mauve', 'tomato']
N = 200
NUM_OF_CLUSTERS = 40
K = 2
NUM_OF_RUN = 20


def euclidean_distance(a, b):
    a_coords = (a['x'], a['y'])
    b_coords = (b['x'], b['y'])
    return np.linalg.norm(np.array(a_coords) - np.array(b_coords))


def get_total_power(G1):
    """ return and print the total power in the graph """
    sum1 = 0
    for node in G1.nodes():
        x = G1.nodes[node]['x']
        y = G1.nodes[node]['y']
        power = G1.nodes[node]['power']
        sum1 += G1.nodes[node]['power']
        # print(f"Node {node}: (x={x}, y={y}), power={power}")

    print("The total power of all the nodes in the graph is ", sum1)
    return sum1


def get_num_of_broadcast(G1):
    """ return and print the number of broadcast message can pass through the graph"""
    max_power = 0
    max_node = None
    for j in G1.nodes:
        if G1.nodes[j]['power'] > max_power:
            max_power = G1.nodes[j]['power']
            max_node = j
    result = G1.nodes[max_node]['battery'] // max_power

    print("The max power in the graph is: ", max_power, " from node ", max_node)
    print("Number of broadcast message can pass through the graph is: ", result)

    return result


def get_diameter(G1):
    """return and print the Diameter of the graph"""
    diameter = nx.algorithms.distance_measures.diameter(G1)
    print("The diameter of the graph is:", diameter)
    return diameter


def set_power(G1):
    """ for every node set the node power according to the biggest edge"""
    for node in G1.nodes():
        max_weight = max([G1.edges[e]['weight'] for e in G1.edges(node)])
        G1.nodes[node]['power'] = max_weight


def add_all_edge_in_radius(G):
    """ for every node add the edges for all other node in the radius (distance < power) """
    for i in G.nodes():
        for j in G.nodes():
            if i != j:
                x1, y1 = G.nodes[i]['x'], G.nodes[i]['y']
                x2, y2 = G.nodes[j]['x'], G.nodes[j]['y']
                dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if dist < G.nodes[i]['power']:
                    G.add_edge(i, j)


def make_complete_graph(G1):
    # Add edges to the graph with weights equal to the Euclidean distance between nodes
    for i in range(N):
        for j in range(i + 1, N):
            x1, y1 = G1.nodes[i]['x'], G1.nodes[i]['y']
            x2, y2 = G1.nodes[j]['x'], G1.nodes[j]['y']
            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            G1.add_edge(i, j, weight=dist)


def make_connected(G):
    """Connects all the separate components in the graph with minimum power so that the graph is connected"""
    # Find all connected components in the graph
    components = list(nx.connected_components(G))
    while len(components) > 1:
        # If the graph is already connected, return
        if len(components) == 1:
            return

        # Find the two closest components
        min_dist = float('inf')
        closest = None
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                dist = min([euclidean_distance(G.nodes[u], G.nodes[v]) for u in components[i] for v in components[j]])
                if dist < min_dist:
                    min_dist = dist
                    closest = (i, j)

        # Connect the two closest components with the shortest edge
        u, v = None, None
        min_edge = None
        for i in components[closest[0]]:
            for j in components[closest[1]]:
                dist = euclidean_distance(G.nodes[i], G.nodes[j])
                if min_edge is None or dist < euclidean_distance(G.nodes[u], G.nodes[v]):
                    min_edge = (i, j)
                    u, v = i, j

        # Add the shortest edge to the graph and call the function recursively
        distance_squared = (G.nodes[u]['x'] - G.nodes[v]['x']) ** 2 + (
                G.nodes[u]['y'] - G.nodes[v]['y']) ** 2
        distance_squared = math.sqrt(distance_squared)
        G.add_edge(u, v, weight=distance_squared)
        components = list(nx.connected_components(G))


def degree_plus_1(G1, min_deg):
    # Loop over all nodes in the graph
    for node in G1.nodes():
        # Check if the node has degree num
        if G1.degree(node) < min_deg:
            # Find the closest node with no edge between them
            closest_dist = math.inf
            closest_node = None
            for other_node in G1.nodes():
                if other_node != node and not G1.has_edge(node, other_node):
                    dist = math.sqrt((G1.nodes[node]['x'] - G1.nodes[other_node]['x']) ** 2 +
                                     (G1.nodes[node]['y'] - G1.nodes[other_node]['y']) ** 2)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_node = other_node
            # Add an edge between the two closest nodes
            G1.add_edge(node, closest_node, weight=closest_dist)


def check_k_link_disjoint_paths(G, k):
    # check if there are at least two edge-disjoint paths between every pair of nodes
    for u in G.nodes():
        for v in G.nodes():
            if u != v:
                paths = list(nx.edge_disjoint_paths(G, u, v))
                if len(paths) < k:
                    return True
    return False


def check_k_node_disjoint_paths(G, k):
    # check if there are at least two edge-disjoint paths between every pair of nodes
    for u in G.nodes():
        for v in G.nodes():
            if u != v:
                paths = list(nx.node_disjoint_paths(G, u, v))
                if len(paths) < k:
                    return False
    return True


def cluster_with_center(G, kmeans, node_labels):
    """This method divide the node to cluster, find center node for each cluster and then connect all the centers to
    each other"""
    """find the center node of each group"""
    centers = []
    for i, center in enumerate(kmeans.cluster_centers_):
        # print(f"Center of cluster {i}: {center}")
        closest_node = min(G.nodes(),
                           key=lambda node: np.linalg.norm(
                               np.array([G.nodes[node]['x'], G.nodes[node]['y']]) - center))
        centers.append(closest_node)

        # Connect all nodes in the cluster to the center node
        for node in G.nodes():
            if node_labels[node] == i and node != closest_node:
                distance_squared = math.sqrt((G.nodes[node]['x'] - G.nodes[closest_node]['x']) ** 2 + (
                        G.nodes[node]['y'] - G.nodes[closest_node]['y']) ** 2)
                G.add_edge(node, closest_node, weight=distance_squared)

    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            distance_squared = math.sqrt((G.nodes[i]['x'] - G.nodes[j]['x']) ** 2 + (
                    G.nodes[i]['y'] - G.nodes[j]['y']) ** 2)
            G.add_edge(centers[i], centers[j], weight=distance_squared)

    set_power(G)
    print("CLUSTERS WITH CENTERS")
    total_power, diameter, num_of_broad = get_total_power(G), get_diameter(G), get_num_of_broadcast(G)

    """Draw the graph"""
    # draw_graph(G, 1, node_labels)
    # add_all_edge_in_radius(G)
    # draw_graph(G, 1, node_labels)

    return total_power, num_of_broad, diameter


def cluster_with_center_MST(G_mst1, kmeans, node_labels):
    """This method divide the node to cluster, find center node for each cluster and then build MST
    for all the center node - the results are a decrease in total power but an increase in the diameter of the graph """

    """find the center node of each group"""
    centers = []
    for i, center in enumerate(kmeans.cluster_centers_):
        # print(f"Center of cluster {i}: {center}")
        closest_node = min(G_mst1.nodes(),
                           key=lambda node: np.linalg.norm(
                               np.array([G_mst1.nodes[node]['x'], G_mst1.nodes[node]['y']]) - center))
        # print(f"Closest node to center of cluster {i}: {closest_node}")
        centers.append(closest_node)

        # Adjust the power for each node in the group
        for node in G_mst1.nodes():
            if node_labels[node] == i and node not in centers:
                distance_squared = (G_mst1.nodes[node]['x'] - G_mst1.nodes[closest_node]['x']) ** 2 + (
                        G_mst1.nodes[node]['y'] - G_mst1.nodes[closest_node]['y']) ** 2
                distance_squared = math.sqrt(distance_squared)
                G_mst1.nodes[node]['power'] = distance_squared

        # Connect all nodes in the cluster to the center node
        for node in G_mst1.nodes():
            if node_labels[node] == i and node != closest_node:
                G_mst1.add_edge(node, closest_node, weight=0)  # ???????????????????????????????????????????????
    # print(centers)

    # Add edges between the nodes in the center list
    for u in centers:
        for v in centers:
            if u < v and not G_mst1.has_edge(u, v):
                x1, y1 = G_mst1.nodes[u]['x'], G_mst1.nodes[u]['y']
                x2, y2 = G_mst1.nodes[v]['x'], G_mst1.nodes[v]['y']
                distance = (x1 - x2) ** 2 + (y1 - y2) ** 2
                distance = math.sqrt(distance)
                G_mst1.add_edge(u, v, weight=distance)

    mst = nx.minimum_spanning_tree(G_mst1)
    G_mst1.remove_edges_from(G_mst1.edges())
    # Add the edges of the minimum spanning tree to the graph
    for (u, v) in mst.edges():
        x1, y1 = G_mst1.nodes[u]['x'], G_mst1.nodes[u]['y']
        x2, y2 = G_mst1.nodes[v]['x'], G_mst1.nodes[v]['y']
        distance = (x1 - x2) ** 2 + (y1 - y2) ** 2
        distance = math.sqrt(distance)
        G_mst1.add_edge(u, v, weight=distance)

    # Adjust the power for each node in the group
    # Connect all nodes in the cluster to the borders node
    for node in G_mst1.nodes:
        min_power = 1000000
        node_to_connect = 0
        for bor in centers:
            if node_labels[node] == node_labels[bor] and node != bor:
                distance_squared = (G_mst1.nodes[node]['x'] - G_mst1.nodes[bor]['x']) ** 2 + (
                        G_mst1.nodes[node]['y'] - G_mst1.nodes[bor]['y']) ** 2
                distance_squared = math.sqrt(distance_squared)
                temp = bor
                if min_power > distance_squared:
                    node_to_connect = temp
                min_power = min(min_power, distance_squared)
        if min_power < 1000000:
            G_mst1.add_edge(node_to_connect, node, weight=min_power)

    G_mst1.remove_edges_from([(u, v) for u, v in G_mst1.edges() if u == v])

    for u in G_mst1.nodes:
        max_power = 0
        for v in G_mst1.nodes:
            if G_mst1.has_edge(u, v):
                weight = G_mst1.get_edge_data(u, v)['weight']
                # print("from ", u, " to ", v, " dist is: ", weight)
                max_power = max(max_power, weight)
        G_mst1.nodes[u]['power'] = max_power
    print("CLUSTERS WITH CENTERS WITH MST")
    total_power, diameter, num_of_broad = get_total_power(G_mst1), get_diameter(G_mst1), get_num_of_broadcast(G_mst1)

    """Draw the graph"""
    # draw_graph(G_mst1, 2, node_labels)
    # add_all_edge_in_radius(G_mst1)
    # draw_graph(G_mst1, 2, node_labels)

    return total_power, num_of_broad, diameter


def cluster_with_border(Q, node_labels1):
    """This method divide the node to cluster, find border node for each cluster. Then connect all the node in each
    cluster to the closest border node, and build an MST for all the border node in the graph - the results are a
    decrease in total power but an increase in the diameter of the graph """
    """find the border node for each cluster"""
    borders = []
    for i in range(NUM_OF_CLUSTERS):
        group_i_nodes = [node for node, label in enumerate(node_labels1) if label == i]
        other_groups = [j for j in range(NUM_OF_CLUSTERS) if j != i]

        # Find the closest node in each other group
        closest_nodes = {}
        for j in other_groups:
            group_j_nodes = [node for node, label in enumerate(node_labels1) if label == j]
            min_distance = math.inf
            closest_node = None
            for node_i in group_i_nodes:
                for node_j in group_j_nodes:
                    distance = (Q.nodes[node_i]['x'] - Q.nodes[node_j]['x']) ** 2 + (
                            Q.nodes[node_i]['y'] - Q.nodes[node_j]['y']) ** 2
                    distance = math.sqrt(distance)
                    if distance < min_distance:
                        min_distance = distance
                        closest_node = node_j
            closest_nodes[j] = closest_node
            if closest_node not in borders:
                borders.append(closest_node)

    # Add edges between the nodes in the borders list
    for u in borders:
        for v in borders:
            if u < v and not Q.has_edge(u, v):
                x1, y1 = Q.nodes[u]['x'], Q.nodes[u]['y']
                x2, y2 = Q.nodes[v]['x'], Q.nodes[v]['y']
                distance = (x1 - x2) ** 2 + (y1 - y2) ** 2
                distance = math.sqrt(distance)
                Q.add_edge(u, v, weight=distance)

    mst = nx.minimum_spanning_tree(Q)
    Q.remove_edges_from(Q.edges())
    # Add the edges of the minimum spanning tree to the graph
    for (u, v) in mst.edges():
        x1, y1 = Q.nodes[u]['x'], Q.nodes[u]['y']
        x2, y2 = Q.nodes[v]['x'], Q.nodes[v]['y']
        distance = (x1 - x2) ** 2 + (y1 - y2) ** 2
        distance = math.sqrt(distance)
        Q.add_edge(u, v, weight=distance)

    # Adjust the power for each node in the group
    # Connect all nodes in the cluster to the borders node
    for node in Q.nodes:
        min_power = 1000000
        node_to_connect = 0
        for bor in borders:
            if node_labels1[node] == node_labels1[bor] and node != bor:
                distance_squared = (Q.nodes[node]['x'] - Q.nodes[bor]['x']) ** 2 + (
                        Q.nodes[node]['y'] - Q.nodes[bor]['y']) ** 2
                distance_squared = math.sqrt(distance_squared)
                temp = bor
                if min_power > distance_squared:
                    node_to_connect = temp
                min_power = min(min_power, distance_squared)
        if min_power < 1000000:
            Q.add_edge(node_to_connect, node, weight=min_power)

    Q.remove_edges_from([(u, v) for u, v in Q.edges() if u == v])
    set_power(Q)

    print("CLUSTERS WITH BORDERS")
    total_power, diameter, num_of_broad = get_total_power(Q), get_diameter(Q), get_num_of_broadcast(Q)

    """Draw the graph"""
    # draw_graph(Q, 3, node_labels1)
    # add_all_edge_in_radius(Q)
    # draw_graph(Q, 3, node_labels1)

    return total_power, num_of_broad, diameter


def MST(G1):
    """This method build a MST - Our assumption is that this is the connection
    that will give the minimum amount of power possible """
    make_complete_graph(G1)
    # Build the minimum spanning tree
    T1 = nx.minimum_spanning_tree(G1)
    set_power(T1)
    print("MST")
    total_power, diameter, num_of_broad = get_total_power(T1), get_diameter(T1), get_num_of_broadcast(T1)

    """Draw the graph"""
    # draw_graph(T1, 4)
    # add_all_edge_in_radius(T1)
    # draw_graph(T1, 4)

    return total_power, num_of_broad, diameter


def k_link_disjoint_path(G1):
    """With this method we want to assign power so that between each pair of nodes there are K link disjoint path. We
    start by connecting each node to its two closest neighbors, if the condition is met then we are done, if not then
    we will raise the degree of all nodes to at least 3 and check again and so on until we meet the condition"""
    # Calculate pairwise distances between nodes
    nodes = np.array([(data["x"], data["y"]) for _, data in G1.nodes(data=True)])
    distances = euclidean_distances(nodes)
    indices = np.argpartition(distances, K + 1)[:, :K + 1]
    for i in range(len(indices)):
        for j in range(K + 1):
            if indices[i][j] != i and not G1.has_edge(i, indices[i][j]):
                temp = indices[i][j]
                distance_squared = (G1.nodes[i]['x'] - G1.nodes[temp]['x']) ** 2 + (
                        G1.nodes[i]['y'] - G1.nodes[temp]['y']) ** 2
                distance_squared = math.sqrt(distance_squared)
                G1.add_edge(i, indices[i][j], weight=distance_squared)

    make_connected(G1)
    deg = 3
    while check_k_link_disjoint_paths(G1, K):
        # print("make every node deg minimum ", deg)
        degree_plus_1(G1, deg)

        deg += 1
    set_power(G1)
    # add_all_edge_in_radius(G1)
    print("K link disjoint path")
    total_power, diameter, num_of_broad = get_total_power(G1), get_diameter(G1), get_num_of_broadcast(G1)

    """Draw the graph"""
    draw_graph(G1, 5)

    return total_power, num_of_broad, diameter


def k_node_disjoint_path(G1):
    """With this method we want to assign power so that between each pair of nodes there are K node disjoint path. We
    start by finding the MST of the graph, then for each pair of connected nodes U,V, we will connect all the
    neighbors of U to all the neighbors of V. This method will ensure that the condition is met."""
    make_complete_graph(G1)
    T_original = nx.minimum_spanning_tree(G1)
    T_copy = T_original.copy()
    # Iterate over all edges in the graph
    for u, v in T_original.edges():
        # Get the set of neighbors of node u and node v
        u_neighbors = set(T_original.neighbors(u))
        v_neighbors = set(T_original.neighbors(v))

        # Add edges between u's neighbors and v's neighbors
        for u_neighbor in u_neighbors:
            for v_neighbor in v_neighbors:
                # print("u_neighbor = ", u_neighbor)
                # print("v_neighbor = ", v_neighbor)
                # Add edge between u's neighbor and v's neighbor
                distance = (T_original.nodes[u_neighbor]['x'] - T_original.nodes[v_neighbor]['x']) ** 2 + (
                        T_original.nodes[u_neighbor]['y'] - T_original.nodes[v_neighbor]['y']) ** 2
                distance = math.sqrt(distance)
                if T_copy.has_edge(u_neighbor, v_neighbor):
                    # print("pass")
                    pass
                else:
                    T_copy.add_edge(u_neighbor, v_neighbor, weight=distance)
        if check_k_node_disjoint_paths(T_copy, K):
            break

    set_power(T_copy)
    print("K node disjoint path")
    total_power, diameter, num_of_broad = get_total_power(T_copy), get_diameter(T_copy), get_num_of_broadcast(T_copy)
    """Draw the graph"""
    # draw_graph(T_original, 6)
    draw_graph(T_copy, 6)

    return total_power, num_of_broad, diameter


def draw_graph(G2, idx, node_labels=[]):
    if idx == 1 or idx == 2 or idx == 3:
        plt.clf()
        # Define the layout
        pos = {i: (G2.nodes[i]['x'], G2.nodes[i]['y']) for i in G2.nodes}
        # Draw the nodes and edges with cluster COLORS
        node_colors = [COLORS[label] for label in node_labels]
        nx.draw_networkx_nodes(G2, pos, node_size=250, node_color=node_colors)
        nx.draw_networkx_edges(G2, pos, width=2, edge_color='gray')
        nx.draw_networkx_labels(G2, pos, font_size=7, font_color='black')
        if idx == 1:
            plt.title("CLUSTERS WITH CENTERS")
        elif idx == 2:
            plt.title("CLUSTERS WITH CENTERS WITH MST")
        else:
            plt.title("CLUSTERS WITH BORDERS")
        plt.show()
    elif idx == 4 or idx == 5 or idx == 6:
        # Draw the graph
        plt.clf()
        pos = {j: (G2.nodes[j]['x'], G2.nodes[j]['y']) for j in G2.nodes}
        nx.draw(G2, pos, with_labels=True)
        if idx == 4:
            plt.title("MST")
        elif idx == 5:
            plt.title("K link disjoint path")
        else:
            plt.title("K node disjoint path")
        plt.show()


def create_graph():
    """Create the graph"""
    G = nx.Graph()
    Q = nx.Graph()
    G_MST = nx.Graph()
    T = nx.Graph()
    k_link = nx.Graph()
    k_node = nx.Graph()
    for j in range(N):
        x1 = random.randint(0, 200)
        y1 = random.randint(0, 200)
        G.add_node(j, x=x1, y=y1, power=0, battery=1000)
        G_MST.add_node(j, x=x1, y=y1, power=0, battery=1000)
        Q.add_node(j, x=x1, y=y1, power=0, battery=1000)
        T.add_node(j, x=x1, y=y1, power=0, battery=1000)
        k_link.add_node(j, x=x1, y=y1, power=0, battery=1000)
        k_node.add_node(j, x=x1, y=y1, power=0, battery=1000)

    """Clustering:"""
    node_positions = np.array([[node[1]['x'], node[1]['y']] for node in G.nodes(data=True)])

    # Cluster the nodes into groups using the KMeans algorithm
    kmeans = KMeans(n_clusters=NUM_OF_CLUSTERS)
    kmeans.fit(node_positions)
    node_labels = kmeans.predict(node_positions)
    return kmeans, node_labels, G, Q, G_MST, T, k_link, k_node


def create_simple_graph():
    """Create the graph"""
    G_star = nx.Graph()
    Q_star = nx.Graph()
    G_ring = nx.Graph()
    Q_ring = nx.Graph()
    G_line = nx.Graph()
    Q_line = nx.Graph()

    """star"""
    # radius = 10
    # center = (50, 50)
    # for i in range(N):
    #     angle = i * 2 * 3.14159 / N
    #     x = center[0] + radius * math.cos(angle)
    #     y = center[1] + radius * math.sin(angle)
    #     G_star.add_node(i, x=x, y=y, power=0)
    # G_star.add_node(N, x=center[0], y=center[1], power=0)

    """line"""
    # x, y = 10, 10
    # for i in range(N):
    #     G_star.add_node(i, x=x, y=y, power=0)
    #     x = x + 5
    #     y = y + 5

    """our"""
    G_star.add_node(0, x=1, y=1, power=0)
    G_star.add_node(2, x=1, y=6, power=0)
    G_star.add_node(3, x=6, y=1, power=0)
    G_star.add_node(1, x=6, y=6, power=0)
    # G_star.add_node(1, x=5.9, y=5.9, power=0)
    # G_star.add_node(1, x=6.1, y=6.1, power=0)

    G_star.add_node(4, x=1, y=51, power=0)
    G_star.add_node(5, x=6, y=56, power=0)
    G_star.add_node(6, x=1, y=56, power=0)
    G_star.add_node(7, x=6, y=51, power=0)
    # G_star.add_node(7, x=5.9, y=51.1, power=0)
    # G_star.add_node(7, x=6.1, y=50.9, power=0)

    G_star.add_node(8, x=51, y=1, power=0)
    G_star.add_node(9, x=56, y=6, power=0)
    G_star.add_node(11, x=56, y=1, power=0)
    G_star.add_node(10, x=51, y=6, power=0)
    # G_star.add_node(10, x=51.1, y=5.9, power=0)
    # G_star.add_node(10, x=50.9, y=6.1, power=0)

    G_star.add_node(13, x=56, y=56, power=0)
    G_star.add_node(14, x=51, y=56, power=0)
    G_star.add_node(15, x=56, y=51, power=0)
    G_star.add_node(12, x=51, y=51, power=0)
    # G_star.add_node(12, x=51.1, y=51.1, power=0)
    # G_star.add_node(12, x=50.9, y=50.9, power=0)

    # G_star.add_edge(1,7)
    # G_star.add_edge(7, 12)
    # G_star.add_edge(12, 10)
    # G_star.add_edge(10, 1)
    # G_star.add_edge(0, 2)
    # G_star.add_edge(0, 3)
    # G_star.add_edge(1, 3)
    # G_star.add_edge(1, 2)
    # G_star.add_edge(4, 7)
    # G_star.add_edge(4, 6)
    # G_star.add_edge(5, 7)
    # G_star.add_edge(5, 6)
    # G_star.add_edge(12, 15)
    # G_star.add_edge(12, 14)
    # G_star.add_edge(13, 14)
    # G_star.add_edge(13, 15)
    # G_star.add_edge(8, 10)
    # G_star.add_edge(8, 11)
    # G_star.add_edge(10, 9)
    # G_star.add_edge(11, 9)
    # pos = {}
    # for node in G_star.nodes():
    #     pos[node] = (G_star.nodes[node]['x'], G_star.nodes[node]['y'])
    #
    # color_map = []
    # for node in G_star.nodes():
    #     if node < 4:
    #         color_map.append('red')
    #     elif node < 8:
    #         color_map.append('blue')
    #     elif node < 12:
    #         color_map.append('green')
    #     else:
    #         color_map.append('purple')
    # nx.draw(G_star, pos, node_color=color_map, with_labels=True)
    # plt.show()

    # G_star.add_node(16, x=26, y=26, power=0)
    # G_star.add_node(17, x=31, y=26, power=0)
    # G_star.add_node(18, x=26, y=31, power=0)
    # G_star.add_node(19, x=31, y=31, power=0)

    return G_star
    # ''', Q_star, G_ring, Q_ring, G_line, Q_line'''


if __name__ == '__main__':
    # Set a seed value
    # seed_value = 42
    # random.seed(seed_value)
    avg_power, avg_broad, avg_diameter = [0] * 6, [0] * 6, [0] * 6
    temp_power, temp_broadcast, temp_diam = 0, 0, 0
    for i in range(NUM_OF_RUN):
        kme, node_Lab, G, G1, G_mst, T, K_link, K_node = create_graph()
        # gsta = create_simple_graph()

        temp_power, temp_broadcast, temp_diam = cluster_with_center(G, kme, node_Lab)
        avg_power[0] += temp_power
        avg_broad[0] += temp_broadcast
        avg_diameter[0] += temp_diam
        print("============================================================================")
        temp_power, temp_broadcast, temp_diam = cluster_with_center_MST(G_mst, kme, node_Lab)
        avg_power[1] += temp_power
        avg_broad[1] += temp_broadcast
        avg_diameter[1] += temp_diam
        print("============================================================================")
        temp_power, temp_broadcast, temp_diam = cluster_with_border(G1, node_Lab)
        avg_power[2] += temp_power
        avg_broad[2] += temp_broadcast
        avg_diameter[2] += temp_diam
        print("============================================================================")
        # temp_power, temp_broadcast, temp_diam = k_link_disjoint_path(K_link)
        # avg_power[3] += temp_power
        # avg_broad[3] += temp_broadcast
        # avg_diameter[3] += temp_diam
        print("============================================================================")
        # temp_power, temp_broadcast, temp_diam = k_node_disjoint_path(K_node)
        # avg_power[4] += temp_power
        # avg_broad[4] += temp_broadcast
        # avg_diameter[4] += temp_diam
        print("============================================================================")
        temp_power, temp_broadcast, temp_diam = MST(T)
        avg_power[5] += temp_power
        avg_broad[5] += temp_broadcast
        avg_diameter[5] += temp_diam
        print("============================================================================")
        print("============================================================================")

    for i in range(len(avg_power)):
        avg_power[i] /= NUM_OF_RUN
        avg_broad[i] /= NUM_OF_RUN
        avg_diameter[i] /= NUM_OF_RUN
    print("Average total power: ", avg_power)
    print("Average diameter: ", avg_diameter)
    print("Average broadcast number: ", avg_broad)
