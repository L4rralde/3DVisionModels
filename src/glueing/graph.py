import os
from typing import List, Hashable, Tuple, Dict
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from glob import glob

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import dijkstra


@dataclass
class Node:
    def __init__(self) -> None:
        predictions: dict
        id: Hashable
    
    @staticmethod
    def from_npz(path: os.PathLike) -> "Node":
        preds = np.load(path, allow_pickle=True)
        hash = os.path.splitext(os.path.basename(path))[0]
        return Node(preds, hash)

@dataclass
class Edge:
    src: Hashable
    dst: Hashable
    cost: float


def nodes_and_edges(
    preds_folder: os.PathLike,
    connections: List[Tuple]
) -> Tuple[List[Node], List[Edge]]:
    basenames = glob("*.npz", root_dir=preds_folder)
    preds_paths = [os.path.join(preds_folder, name) for name in basenames]
    nodes = [Node.from_npz(path) for path in preds_paths]
    edges = [Edge(*connection) for connection in connections]

    return nodes, edges


@dataclass
class Path:
    path: List[Hashable]
    cost: float


class Graph:
    def __init__(
            self,
            nodes: List[Node],
            edges: List[Edge]
        ) -> None:
        self.nodes: OrderedDict[Node] = {node.id: node for node in nodes}
        self.node_ids = list(self.nodes.keys())
        self.edges: List[Edge] = edges
        self.incidence = self.__incidence_matrix()

    @staticmethod
    def from_preds_folder(
        preds_folder: os.PathLike,
        connections: Tuple[Hashable, Hashable, float]
    ) -> "Graph":
        nodes, edges = nodes_and_edges(preds_folder, connections)
        graph = Graph(nodes, edges)
        return graph

    def __incidence_matrix(self) -> np.ndarray:
        matrix = np.inf * np.ones((len(self.nodes), len(self.nodes)))
        for edge in self.edges:
            idx_src = self.node_ids.index(edge.src)
            idx_dst = self.node_ids.index(edge.dst)
            matrix[idx_src, idx_dst] = edge.cost
            matrix[idx_dst, idx_src] = edge.cost

        matrix = np.triu(matrix)
        return matrix

    def paths(self, src: Hashable, dst: Hashable) -> Dict[Hashable, List[Path]]:
        raise NotImplementedError("TODO")

    def __path_from_dijkstra(
        self,
        src: Hashable,
        dst: Hashable,
        dist_matrix: np.ndarray,
        predecessors: List[int],
    ) -> Path:
        src_idx = self.node_ids.index(src)
        dst_idx = self.node_ids.index(dst)

        cost = dist_matrix[dst_idx]
        path_nodes_list: List[Node] = []
        current = dst_idx
        while current != src_idx:
            current_id = self.node_ids[current]
            current_node = self.nodes[current_id]
            path_nodes_list.append(current_node)
            current = predecessors[current]

        return Path(path_nodes_list, cost)

    @lru_cache
    def __shortest_paths(self, src: Hashable) -> Dict[Hashable, Path]:
        graph = csr_array(self.incidence)

        src_idx = self.node_ids.index(src)
        dist_matrix, predecessors = dijkstra(
            csgraph=graph,
            directed=False,
            indices = src_idx,
            return_predecessors=True
        )
        return {
            id: self.__path_from_dijkstra(src, id, dist_matrix, predecessors)
            for id in self.node_ids
        }

    def shortest_path(self, src: Hashable, dst: Hashable) -> Path:
        shortest_paths = self.__shortest_paths(src)
        
        path = shortest_paths[dst]
        if path.cost == np.inf:
            raise RuntimeError("No path found from {src} to {dst}")
        
        return path


class Tree(Graph):
    def __init__(
        self,
        root: Hashable,
        nodes: List[Node],
        edges: List[Edge]
    ) -> None:
        super().__init__(nodes, edges)
        self.root = root

    @staticmethod
    def from_preds_folder(
        root: Hashable, 
        preds_folder: os.PathLike,
        connections: Tuple[Hashable, Hashable, float]
    ) -> "Graph":
        nodes, edges = nodes_and_edges(preds_folder, connections)
        return Tree(root, nodes, edges)

    def shortest_path(self, dst: Hashable) -> Path:
        return super().shortest_path(self.root, dst)
