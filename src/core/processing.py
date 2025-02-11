
import os
import sys
import numpy as np
import scipy.io
import itertools
import networkx as nx
from src.core.graph import Graph

class Processing:
    def __init__(self, segmentation_dir, celltype_dir, scan, threshold_distance, node_colors, save_dir):
        self.segmentation_dir = segmentation_dir
        self.celltype_dir = celltype_dir
        self.scan = scan
        self.threshold_distance = threshold_distance
        self.node_colors = node_colors
        self.save_dir = save_dir
    
        # Attributes to be populated later
        self.img_size = None
        self.bd_indices = None
        self.reg_indices = None
        self.celltype = None 
        self.all_labels = None
        self.graph = Graph(self.img_size, threshold_distance)

    def read_mat_from_source(self):  
        """
        Reads .mat files and extracts necessary data.
        Stores extracted values as instance attributes.
        """
        f_nuclei = os.path.join(self.segmentation_dir,self.scan,"nuclei_multiscale.mat")
        f_region = os.path.join(self.segmentation_dir,self.scan,"allRegionIndices.mat")
        f_celltype = os.path.join(self.celltype_dir,self.scan+".mat")

        mat = scipy.io.loadmat(f_nuclei)
        self.img_size = mat['nucleiImage'].shape 
        self.bd_indices = mat['Boundaries'][0]

        mat = scipy.io.loadmat(f_region)
        self.reg_indices = mat['allRegInds']

        mat = scipy.io.loadmat(f_celltype)
        self.celltypes = mat['cellTypes']
        self.all_labels = mat['allLabels']
    
    def flatten_celltypes(self,requested_celltypes):
        """
        Converts requested_celltypes (which might contain dictionaries) into a flat list.
        """
        return [item for sublist in requested_celltypes for item in 
                (list(itertools.chain.from_iterable(sublist.values())) if isinstance(sublist, dict) else [sublist])]

        
    def count_unique_fully_connected_paths(self, graph, start_nodes, middle_nodes, end_nodes):
        '''
        count cell pairs from the graph built already
        a pair := a (start_node, middle_node, end_node) loop
        '''
        unique_paths = set()
        for start_node in start_nodes:
            for middle_node in middle_nodes:
                if graph.has_edge(start_node, middle_node):
                    for end_node in end_nodes:
                        # make sure end_node is also connected back to  start_node
                        if graph.has_edge(middle_node, end_node) and graph.has_edge(start_node, end_node):
                            # avoid duplicates by sorting the cells in every pair and add to a set 
                            path = tuple(sorted([start_node, middle_node, end_node]))
                            unique_paths.add(path)
        return unique_paths

    def process_scan(self, requested_celltypes):
        """
        Main function that processes the scan.
        """
        self.read_mat_from_source()
        linearized_celltypes = self.flatten_celltypes(requested_celltypes)
        n_cells = self.reg_indices.shape[0]

        # Generate graphs
        G_base, G, center_locations = self.graph.initiate_base_graph(n_cells, self.img_size, self.reg_indices, self.celltypes, linearized_celltypes)
        print("base_initiated")
        self.graph.save_graph_to_pdf(G, self.img_size, center_locations, self.node_colors, os.path.join(self.save_dir, self.scan, "all_requested_cells.pdf"))
        print("base save done")
        # Identify cell triplets
        counts = []
        for ct in requested_celltypes:
            # handle the case where we have a subtype (for "MAC")
            if isinstance(ct,dict):
                count = 0 
                for c in list(ct.values()):
                    count += np.count_nonzero(self.celltypes == c)
                counts.append([count,list(ct.keys())[0]])
            else:
                counts.append([np.count_nonzero(self.celltypes[0] == ct),ct]) 

        sorted_counts = sorted(counts)
        start_type, middle_type, end_type = sorted_counts[0][1], sorted_counts[1][1], sorted_counts[2][1]

        start_nodes = [node for node, data in G.nodes(data=True) if data['type'].startswith(start_type) and G.degree(node) > 0]
        middle_nodes = [node for node, data in G.nodes(data=True) if data['type'].startswith(middle_type) and G.degree(node) > 0]
        end_nodes = [node for node, data in G.nodes(data=True) if data['type'].startswith(end_type) and G.degree(node) > 0]
        print("start counting loops")
        # Count fully connected paths
        rst = self.count_unique_fully_connected_paths(G, start_nodes, middle_nodes, end_nodes)
        print("counting done")
        # Build filtered graphs
        self.graph.save_graph_to_pdf(G_base, self.img_size, center_locations, self.node_colors, os.path.join(self.save_dir, self.scan, "test.pdf"))
        G_filtered_pairs = self.graph.build_filtered_graph(G_base, rst)
        self.graph.save_graph_to_pdf(G_filtered_pairs,self.img_size, center_locations, self.node_colors, os.path.join(self.save_dir, self.scan, "pairs_found.pdf"))

        G_filtered_pairs,G_filtered_pairs_nodes = self.graph.refine_graph(G_filtered_pairs, center_locations)
        self.graph.save_graph_to_pdf(G_filtered_pairs,self.img_size, center_locations, self.node_colors, os.path.join(self.save_dir, self.scan, "groups_found.pdf"))
        components = list(nx.connected_components(G_filtered_pairs_nodes))
        num_isolated_groups = len(components)

        np.savez(os.path.join(self.save_dir,self.scan,"cell_pairs"),pairs = np.array(list(rst)), components=components,count = num_isolated_groups)


