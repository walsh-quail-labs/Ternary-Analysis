import os
import numpy as np
import networkx as nx
import itertools
from matplotlib import pyplot as plt

class Graph:
    def __init__(self, img_size, threshold_distance):
        self.img_size = img_size 
        self.threshold_distance = threshold_distance
        self.G = nx.Graph()
    
    @staticmethod
    def ind2sub(array_shape, linear_indices):
        cols = ((linear_indices-1) / array_shape[0]).astype(int)
        rows = ((linear_indices-1) % array_shape[0]).astype(int)
        return cols,rows

    def initiate_base_graph(self,n_cells,img_size,reg_indices,celltypes,requested_celltypes):
        '''
        initiate a graph where
        - node: added when cell is of requested celltype; 
                has a property "type", which was used later to assign colors;
        - edge: added whenever two cells are touching;
                touching := center of two cells are closer than the distance_threshold;
        Return
        - G_base: a graph with nodes only
        - G: a graph with nodes and all touching edges
        - center_locations: center of every cell, used to make plots later
        '''

        cells = []
        center_locations = {}
        for idx in range(n_cells):
            cols,rows = self.ind2sub(img_size,reg_indices[idx][0])
            center_location = (int(np.mean(rows)),int(np.mean(cols)))
            if celltypes[idx][0].shape[0] == 0:
                continue
            celltype = celltypes[idx][0][0]
            if celltype in requested_celltypes:
                cells.append(idx)
                center_locations[idx] =  center_location
                self.G.add_node(idx, location=center_location, type=celltypes[idx][0][0])

        G_base = self.G.copy()

        # loop through all the edges and draw an edge if they're close enough 
        for idx1, idx2 in list(itertools.product(cells, cells)):
            if idx1 == idx2:
                continue
            loc1 = center_locations[idx1]
            loc2 = center_locations[idx2]
            distance = np.linalg.norm(np.array(loc1) - np.array(loc2))
            if distance <= self.threshold_distance:
                self.G.add_edge(idx1, idx2, weight=int(distance))

        return G_base, self.G, center_locations
       
    def build_filtered_graph(self, G_base, connected_triplets):
        """
        Creates a graph with only fully connected triplet edges.
        """
        G_filtered = G_base.copy()
        for triplet in connected_triplets:
            G_filtered.add_edge(triplet[0], triplet[1])
            G_filtered.add_edge(triplet[0], triplet[2])
            G_filtered.add_edge(triplet[1], triplet[2])
        return G_filtered

    def refine_graph(self, G_filtered_pairs, center_locations):
        """
        Adds edges between nearby nodes to form connected clusters.
        """
        G_filtered_pairs_nodes = G_filtered_pairs.copy()
        isolated_nodes = list(nx.isolates(G_filtered_pairs_nodes))
        G_filtered_pairs_nodes.remove_nodes_from(isolated_nodes)

        for idx1, idx2 in itertools.combinations(G_filtered_pairs_nodes.nodes(), 2):
            if idx1 == idx2:
                continue
            loc1 = center_locations[idx1]
            loc2 = center_locations[idx2]
            distance = np.linalg.norm(np.array(loc1) - np.array(loc2))
            
            if not G_filtered_pairs_nodes.has_edge(idx1, idx2) and distance <= self.threshold_distance + 5:
                G_filtered_pairs_nodes.add_edge(idx1, idx2, weight=int(distance), type="colored")

            if not G_filtered_pairs.has_edge(idx1, idx2) and distance <= self.threshold_distance + 5:
                G_filtered_pairs.add_edge(idx1, idx2, weight=int(distance), type="colored")

        return G_filtered_pairs,G_filtered_pairs_nodes
    
    def save_graph_to_pdf(self,graph,img_size,center_locations,node_colors,save_path):
        '''
        save the graph plot to pdf 
        '''
        # the plotted need to be rotate 90 deg counter-clockwise to align with the original scan
        # -- looks like a behavior of networkx
        print(save_path)

        fixed_positions = {node: center_locations[node] for node in graph.nodes}
        rotated_positions = {node: (y, -x) for node, (x, y) in fixed_positions.items()}
        type_colors = [node_colors[graph.nodes[node]['type']] for node in graph.nodes]

        # stretch the size of the plot according to size of the scan
        fig = plt.figure(
                figsize=(12,int(12*(img_size[0]/img_size[1])))
            )
        nx.draw(graph, 
                rotated_positions, 
                with_labels=False, 
                node_color=type_colors, 
                node_size=3, 
                font_size=8, 
                width=1,
                edge_color=["#D1C780" if 'type' in graph.edges[edge] else "black" for edge in graph.edges]) 
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')