import os
from typing import List, Hashable, Tuple, Dict, Callable
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from glob import glob

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import dijkstra


@dataclass
class Node:
    predictions: dict
    id: Hashable
    
    @staticmethod
    def from_npz(path: os.PathLike) -> "Node":
        preds = np.load(path, allow_pickle=True)
        hash = os.path.splitext(os.path.basename(path))[0]
        return Node(preds, hash)

    def __repr__(self) -> str:
        return self.id

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
    path: List[dict]
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

    @property
    def predictions(self) -> Dict[Hashable, dict]:
        return {
            hash: node.predictions
            for hash, node in self.nodes.items()
        }

    def __incidence_matrix(self) -> np.ndarray:
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for edge in self.edges:
            idx_src = self.node_ids.index(edge.src)
            idx_dst = self.node_ids.index(edge.dst)
            matrix[idx_src, idx_dst] = edge.cost
            matrix[idx_dst, idx_src] = edge.cost

        matrix = np.triu(matrix)
        return matrix

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
            path_nodes_list.append(current_node.predictions)
            current = predecessors[current]
        path_nodes_list.append(self.nodes[src].predictions)

        return Path(path_nodes_list, cost)

    @lru_cache
    def _shortest_paths(self, src: Hashable) -> Dict[Hashable, Path]:
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
        shortest_paths = self._shortest_paths(src)
        
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
    ) -> "Tree":
        nodes, edges = nodes_and_edges(preds_folder, connections)
        return Tree(root, nodes, edges)

    def shortest_path(self, dst: Hashable) -> Path:
        return super().shortest_path(self.root, dst)

    def align(
        self,
        align_function: Callable,
        current: Hashable|None = None,
        aligned: Dict[Hashable, bool]|None = None,
    ) -> Dict[Hashable, bool]:
        if current is None:
            current = self.root

        if aligned is None:
            aligned = {hash: False for hash in self.node_ids}
        aligned[current] = True

        incidence = self.incidence + self.incidence.transpose()
        current_i = self.node_ids.index(current)
        children_idcs,  = np.where(incidence[current_i] > 0)
        for child_i in children_idcs:
            child_hash = self.node_ids[child_i]
            if aligned[child_hash]:
                continue
            current_pred = self.nodes[current].predictions
            child_pred = self.nodes[child_hash].predictions

            aligned_child_pred = align_function(child_pred, current_pred)
            self.nodes[child_hash].predictions = aligned_child_pred

            aligned = self.align(align_function, child_hash, aligned)

        return aligned
