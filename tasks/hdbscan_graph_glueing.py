import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse

import src.glueing as glueing


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--global_features', required=True)
    parser.add_argument('--photos_dir', default='')
    parser.add_argument('--output_dir', required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    features_df = glueing.FeaturesDf.from_json(
        args.global_features,
        args.photos_dir
    )

    filtered_feats = glueing.filter_features_df(features_df, 0.05, 0.7)

    clustered_feats_df = glueing.hdbscan_clustering(filtered_feats, 3, 15)
    graph = glueing.ClusterGraph(clustered_feats_df)

    graph.save(args.output_dir)
    key_photos = graph.key_photos
    key_photos_path = os.path.join(args.output_dir, 'key_photos.txt')
    with open(key_photos_path, 'w') as file:
        for item in key_photos:
            file.write(item + '\n')


if __name__ == '__main__':
    main()
