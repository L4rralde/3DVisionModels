import os

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

from .types import FeaturesDf

    
def hdbscan_clustering(
    feats_df: FeaturesDf,
    min_cluster_size: int=3,
    max_cluster_size: int=15
) -> FeaturesDf:
    feats_mat = feats_df.feats_mat
    hdbscan = HDBSCAN(
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        metric=lambda x, y: 1- np.dot(x, y)
    )
    hdbscan = hdbscan.fit(feats_mat)

    new_df = pd.DataFrame(feats_df.df)
    new_df['clusters'] = hdbscan.labels_

    return FeaturesDf(new_df, feats_df.dir_path)


class ClusterGraph:
    def __init__(self, feats_df: FeaturesDf) -> None:
        df = pd.DataFrame(feats_df.df)
        df = df[df['clusters'] >= 0]
        self.feats_df = FeaturesDf(df, feats_df.dir_path)

        if not "clusters" in self.feats_df.df.columns:
            raise ValueError("The dataframe must include the 'clusters' column")
        
        hierarchy_df, self.connections = self.build()
        self.hierarchy_feats_df = FeaturesDf(hierarchy_df, feats_df.dir_path)

    def __str__(self) -> str:
        return str(self.edges)

    @property
    def edges(self) -> list:
        df = self.hierarchy_feats_df.df
        edges = []
        for edge in self.connections:
            i = edge['src_image_idx']
            j = edge['dst_image_idx']
            label_src = df['clusters'].iloc[i].item()
            label_dst = df['clusters'].iloc[j].item()
            weight = edge['weight']
            edges.append((label_src, label_dst, weight))

        return edges

    @property
    def key_photos(self) -> list:
        key_photos = []
        for connection in self.connections:
            key_photos += [connection['src_image_path'], connection['dst_image_path']]
        
        return list(set(key_photos))

    def build(self) -> tuple:
        all_connections = []
        tmp_df = pd.DataFrame(self.feats_df.df)

        groups = tmp_df['clusters']
        level = 0
        while True:
            connections, edges = self.get_connections(groups)
            all_connections += connections
            graphs = ClusterGraph.connect_edges(edges)
            if len(graphs) == 1:
                break
            groups = groups.apply(lambda label: ClusterGraph.get_graph_idx(label, graphs))
            tmp_df[f'graph_{level}'] = groups
            level += 1

        return tmp_df, all_connections

    def get_group_sim_matrix(self, label_a: int, df_groups: pd.Series) -> np.array:
        idcs = np.where(df_groups==label_a)[0]
        mask_a = np.zeros(len(df_groups))
        mask_a[idcs] = 1
        mask_a = mask_a.reshape(1, -1)

        idcs = np.where(df_groups!=label_a)[0]
        mask_b = np.zeros(len(df_groups))
        mask_b[idcs] = 1
        mask_b = mask_b.reshape(1, -1)

        mask = np.dot(mask_a.T, mask_b)

        return mask * self.feats_df.sims_mat

    def find_group_connection(self, label: int, df_groups: pd.Series) -> dict:
        df = self.feats_df.df
        sim_mat = self.get_group_sim_matrix(label, df_groups)
        weight = sim_mat.max()
        i, j = np.where(sim_mat == weight)
        i = i.item()
        j = j.item()
        cluster_i = df['clusters'].iloc[i].item()
        cluster_j = df['clusters'].iloc[j].item()
        image_path_i = df['image_paths'].iloc[i]
        image_path_j = df['image_paths'].iloc[j]
        return {
            'src_image_idx': i,
            'dst_image_idx': j,
            'src_cluster': cluster_i,
            'dst_cluster': cluster_j,
            'src_image_path': image_path_i,
            'dst_image_path': image_path_j,
            'weight': weight.item()
        }

    def get_connections(self, df_groups: pd.Series) -> tuple:
        connections = [
            self.find_group_connection(i, df_groups)
            for i in range(df_groups.max() + 1)
        ]
        
        edges = [
            (i, df_groups.iloc[connection['dst_image_idx']].item())
            for i, connection in enumerate(connections)
        ]

        return connections, edges

    @staticmethod
    def connect_edges(edges) -> list:
        prev_graphs = list(edges)
        while True:
            graphs = []
            for edge in prev_graphs:
                edge = set(edge)
                found = False
                for i in range(len(graphs)):
                    if edge & graphs[i]:
                        graphs[i] = graphs[i] | edge
                        found = True
                        break
                if not found:
                    graphs.append(edge)
            if graphs == prev_graphs:
                return graphs
            prev_graphs = list(graphs)

    @staticmethod
    def get_graph_idx(label, graphs) -> int:
        for i, graph in enumerate(graphs):
            if label in graph:
                return i
        return -1

    def save_photos(self, dir_path: os.PathLike="graph") -> None:
        graph_levels = sorted([
            column
            for column in self.hierarchy_feats_df.df.columns
            if 'graph_' in column
        ], reverse=True)

        for i, row in enumerate(self.hierarchy_feats_df):
            graphs = [
                str(value)
                for value in row[graph_levels].values
            ]
            image = self.hierarchy_feats_df.get_image(i)
            path = os.path.join(
                dir_path,
                *graphs,
                str(row['clusters']),
                row['image_paths']
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            image.save(path)

    def save_connections(self, path: os.PathLike) -> None:
        df = pd.DataFrame(self.connections)
        df.to_csv(path)

    def save_hierarchy(self, path: os.PathLike) -> None:
        self.hierarchy_feats_df.save(path)

    def save(self, dir_path: os.PathLike) -> None:
        os.makedirs(dir_path, exist_ok=True)
        #self.save_photos(os.path.join(dir_path, 'photos'))
        self.save_connections(os.path.join(dir_path, 'connections.csv'))
        self.save_hierarchy(os.path.join(dir_path, 'hierarcy_feats.json'))


def hierarchical_clustering(feats_df: FeaturesDf):
    aux_df = pd.DataFrame(feats_df.df)
    aux_df['clusters'] = aux_df.reset_index(drop=True).index

    aux_feats_df = FeaturesDf(aux_df, feats_df.dir_path)
    graph = ClusterGraph(aux_feats_df)

    aux_df = pd.DataFrame(feats_df.df)
    aux_df['clusters'] = graph.hierarchy_feats_df.df['graph_1']
    aux_feats_df = FeaturesDf(aux_df, feats_df.dir_path)
    graph = ClusterGraph(aux_feats_df)

    return graph
