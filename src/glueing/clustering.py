import os
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

from .types import FeaturesDf, ClusteredFeaturesDf


def hdbscan_clustering(
    feats_df: FeaturesDf,
    min_cluster_size: int=3,
    max_cluster_size: int=15
) -> ClusteredFeaturesDf:
    feats_mat = feats_df.feats_mat
    hdbscan = HDBSCAN(
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        metric=lambda x, y: 1- np.dot(x, y)
    )
    hdbscan = hdbscan.fit(feats_mat)

    new_df = pd.DataFrame(feats_df.df)
    new_df['clusters'] = hdbscan.labels_

    return ClusteredFeaturesDf(new_df, feats_df.dir_path)


class ConnectedClusterTree:
    def __init__(self, feats_df: FeaturesDf, depth: int=-1) -> None:
        df = pd.DataFrame(feats_df.df)
        df = df[df['clusters'] >= 0]
        self.clustered_feats_df = ClusteredFeaturesDf(df, feats_df.dir_path)
        self.depth = depth
        
        self.connections: list = []
        self.build()

    def __str__(self) -> str:
        return str(self.edges)

    @property
    def clusters(self) -> Dict[str, set]:
        df_clusters =  self.clustered_feats_df.clusters
        clusters = {
            label: set(df_clusters['image_paths'])
            for label in df_clusters.keys()
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

    @property
    def edges(self) -> list:
        df = self.clustered_feats_df.df
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

    def build(self) -> "ConnectedClusterTree":
        all_connections = []
        tmp_df = pd.DataFrame(self.clustered_feats_df.df)

        groups = tmp_df['clusters']
        level = 0
        while True:
            connections, edges = self.get_connections(groups)
            all_connections += connections
            trees = ConnectedClusterTree.connect_edges(edges)
            if len(trees) == 1:
                break
            groups = groups.apply(lambda label: ConnectedClusterTree.get_tree_idx(label, trees))
            tmp_df[f'tree_{level}'] = groups

            level += 1

            if self.depth >= 2 and (level+1) == self.depth:
                break

        self.connections = all_connections
        self.clustered_feats_df = ClusteredFeaturesDf(tmp_df, self.clustered_feats_df.dir_path)

        return self

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

        return mask * self.clustered_feats_df.sims_mat

    def find_group_connection(self, label: int, df_groups: pd.Series) -> dict:
        df = self.clustered_feats_df.df
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

    def save_photos(self, dir_path: os.PathLike="tree") -> None:
        tree_levels = sorted([
            column
            for column in self.clustered_feats_df.df.columns
            if 'tree_' in column
        ], reverse=True)

        for i, row in enumerate(self.clustered_feats_df):
            trees = [
                str(value)
                for value in row[tree_levels].values
            ]
            image = self.clustered_feats_df.get_image(i)
            path = os.path.join(
                dir_path,
                *trees,
                str(row['clusters']),
                row['image_paths']
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            image.save(path)

    def save_connections(self, path: os.PathLike) -> None:
        df = pd.DataFrame(self.connections)
        df.to_csv(path)

    def save_hierarchy(self, path: os.PathLike) -> None:
        self.clustered_feats_df.save(path)

    def save(self, dir_path: os.PathLike) -> None:
        os.makedirs(dir_path, exist_ok=True)
        #self.save_photos(os.path.join(dir_path, 'photos'))
        self.save_connections(os.path.join(dir_path, 'connections.csv'))
        self.save_hierarchy(os.path.join(dir_path, 'hierarcy_feats.json'))


def hierarchical_clustering(feats_df: FeaturesDf):
    aux_df = pd.DataFrame(feats_df.df)
    aux_df['clusters'] = aux_df.reset_index(drop=True).index

    aux_feats_df = FeaturesDf(aux_df, feats_df.dir_path)
    tree = ConnectedClusterTree(aux_feats_df, depth=3)

    aux_df = pd.DataFrame(feats_df.df)

    aux_df['clusters'] = tree.clustered_feats_df.df['tree_1']
    aux_feats_df = FeaturesDf(aux_df, feats_df.dir_path)
    tree = ConnectedClusterTree(aux_feats_df)

    return tree

