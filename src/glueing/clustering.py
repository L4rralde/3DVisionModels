import os
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image

from .types import FeaturesDf


def hdbscan_clustering(
    feats_df: FeaturesDf,
    min_cluster_size: int=3,
    max_cluster_size: int=15
) -> FeaturesDf:
    from sklearn.cluster import HDBSCAN

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


class ConnectedClusterTree:
    def __init__(self) -> None:
        self.connections: List[tuple] = []
        self.df: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def from_feature_df(feats_df: FeaturesDf, depth: int=-1) -> "ConnectedClusterTree":
        if not 'clusters' in feats_df.df.columns:
            raise ValueError("Input FeaturesDf does not include clusters")
        df = pd.DataFrame(feats_df.df)
        df = df[df['clusters'] >= 0]
        clustered_feats_df = FeaturesDf(df, feats_df.dir_path)

        tree = ConnectedClusterTree()
        tree.connections, tree.df = ConnectedClusterTree.build(clustered_feats_df, depth)
        return tree

    @staticmethod
    def from_dir_output(dir_path: os.PathLike) -> "ConnectedClusterTree":
        tree = ConnectedClusterTree()
        tree.load(dir_path)
        return tree

    @property
    def edges(self) -> list:
        edges = []
        for edge in self.connections:
            label_src = edge['src_cluster']
            label_dst = edge['dst_cluster']
            weight = edge['weight']
            edges.append((label_src, label_dst, weight))

        return edges

    @property
    def key_photos(self) -> list:
        key_photos = []
        for connection in self.connections:
            key_photos += [connection['src_image_path'], connection['dst_image_path']]
        
        return list(set(key_photos))

    @property
    def clusters(self) -> Dict[str, set]:
        df = self.df
        clusters = {
            label: set(df[df['clusters'] == label]['image_paths'])
            for label in df['clusters'].unique()
        }
        return clusters

    @property
    def joint_clusters(self) -> Dict[str, set]:
        disjoint_clusters = self.clusters
        for connection in self.connections:
            src_img = connection['src_image_path']
            dst_img = connection['dst_image_path']
            src_label = connection['src_cluster']
            dst_label = connection['dst_cluster']

            disjoint_clusters[dst_label].add(src_img)
            disjoint_clusters[src_label].add(dst_img)
        
        return disjoint_clusters
    
    def __str__(self) -> str:
        return str(self.edges)

    def save_connections(self, path: os.PathLike) -> None:
        df = pd.DataFrame(self.connections)
        df.to_csv(path)

    def load_connections(self, path: os.PathLike) -> None:
        df_connections = pd.read_csv(path).drop(columns='Unnamed: 0')
        self.connections = df_connections.to_dict(orient='records')

    def save_hierarchy(self, path: os.PathLike) -> None:
        self.df.to_csv(path)

    def load_hierarchy(self, path: os.PathLike) -> None:
        self.df = pd.read_csv(path).drop(columns='Unnamed: 0')

    def save(self, dir_path: os.PathLike) -> None:
        os.makedirs(dir_path, exist_ok=True)
        self.save_connections(os.path.join(dir_path, 'connections.csv'))
        self.save_hierarchy(os.path.join(dir_path, 'hierarchy_feats.csv'))

    def load(self, dir_path: os.PathLike) -> None:
        self.load_connections(os.path.join(dir_path, 'connections.csv'))
        self.load_hierarchy(os.path.join(dir_path, 'hierarchy_feats.csv'))

    def save_photos(self, photos_dir: os.PathLike=".", output_dir: os.PathLike="tree") -> None:
        tree_levels = sorted([
            column
            for column in self.df.columns
            if 'tree_' in column
        ], reverse=True)

        for i, row in self.df.iterrows():
            trees = [
                str(value)
                for value in row[tree_levels].values
            ]
            image = Image.open(os.path.join(photos_dir, row['image_paths']))
            path = os.path.join(
                output_dir,
                *trees,
                str(row['clusters']),
                row['image_paths']
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            image.save(path)

    @staticmethod
    def build(clustered_feats_df: FeaturesDf, depth: int=-1) -> tuple:
        all_connections = []
        tmp_df = pd.DataFrame(clustered_feats_df.df)

        groups = tmp_df['clusters']
        level = 0
        while True:
            connections, edges = ConnectedClusterTree.get_connections(clustered_feats_df, groups)
            all_connections += connections
            trees = ConnectedClusterTree.connect_edges(edges)
            if len(trees) == 1:
                break
            groups = groups.apply(lambda label: ConnectedClusterTree.get_tree_idx(label, trees))
            tmp_df[f'tree_{level}'] = groups

            level += 1

            if depth >= 2 and (level+1) == depth:
                break

        connections = all_connections
        clustered_feats_df = FeaturesDf(tmp_df, clustered_feats_df.dir_path)
        df = clustered_feats_df.df.drop(columns=['global_descriptors'])

        return connections, df

    @staticmethod
    def get_group_sim_matrix(clustered_feats_df: FeaturesDf, label_a: int, df_groups: pd.Series) -> np.array:
        idcs = np.where(df_groups==label_a)[0]
        mask_a = np.zeros(len(df_groups))
        mask_a[idcs] = 1
        mask_a = mask_a.reshape(1, -1)

        idcs = np.where(df_groups!=label_a)[0]
        mask_b = np.zeros(len(df_groups))
        mask_b[idcs] = 1
        mask_b = mask_b.reshape(1, -1)

        mask = np.dot(mask_a.T, mask_b)

        return mask * clustered_feats_df.sims_mat

    @staticmethod
    def find_group_connection(clustered_feats_df: FeaturesDf, label: int, df_groups: pd.Series) -> dict:
        df = clustered_feats_df.df
        sim_mat = ConnectedClusterTree.get_group_sim_matrix(clustered_feats_df, label, df_groups)
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

    @staticmethod
    def get_connections(clustered_feats_df: FeaturesDf, df_groups: pd.Series) -> tuple:
        connections = [
            ConnectedClusterTree.find_group_connection(clustered_feats_df, i, df_groups)
            for i in range(df_groups.max() + 1)
        ]
        
        edges = [
            (i, df_groups.iloc[connection['dst_image_idx']].item())
            for i, connection in enumerate(connections)
        ]

        return connections, edges

    @staticmethod
    def connect_edges(edges) -> list:
        prev_trees = list(edges)
        while True:
            trees = []
            for edge in prev_trees:
                edge = set(edge)
                found = False
                for i in range(len(trees)):
                    if edge & trees[i]:
                        trees[i] = trees[i] | edge
                        found = True
                        break
                if not found:
                    trees.append(edge)
            if trees == prev_trees:
                return trees
            prev_trees = list(trees)

    @staticmethod
    def get_tree_idx(label, trees) -> int:
        for i, tree in enumerate(trees):
            if label in tree:
                return i
        return -1


def hierarchical_clustering(feats_df: FeaturesDf):
    aux_df = pd.DataFrame(feats_df.df)
    aux_df['clusters'] = aux_df.reset_index(drop=True).index

    aux_feats_df = FeaturesDf(aux_df, feats_df.dir_path)
    tree = ConnectedClusterTree.from_feature_df(aux_feats_df, depth=3)

    aux_df = pd.DataFrame(feats_df.df)

    aux_df['clusters'] = tree.df['tree_1']
    aux_feats_df = FeaturesDf(aux_df, feats_df.dir_path)
    tree = ConnectedClusterTree.from_feature_df(aux_feats_df)

    return tree
