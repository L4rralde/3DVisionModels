import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
from PIL import Image

import src.glueing as glueing


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--global_features', required=True)
    parser.add_argument('--photos_dir', default='')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--filter_l', default=0.3)
    parser.add_argument('--filter_h', default=0.8)
    parser.add_argument('--min_elements', default=3)
    parser.add_argument('--max_elements', default=10)
    parser.add_argument('--save_photos', action='store_true')

    return parser.parse_args()


def main():
    args = parse_args()

    features_df = glueing.FeaturesDf.from_json(
        args.global_features,
        args.photos_dir
    )

    filtered_feats = glueing.filter_features_df(features_df, args.filter_l, args.filter_h)

    clustered_feats_df = glueing.hdbscan_clustering(filtered_feats, args.min_elements, args.max_elements)
    graph = glueing.ClusterGraph(clustered_feats_df)

    graph.save(args.output_dir)
    key_photos = graph.key_photos
    key_photos_path = os.path.join(args.output_dir, 'key_photos.txt')
    with open(key_photos_path, 'w') as file:
        for item in key_photos:
            file.write(item + '\n')

    if args.save_photos:
        graph.save_photos(args.output_dir)
    
    os.makedirs(os.path.join(args.output_dir, 'key_photos'))
    for key_photo in key_photos:
        input_path = os.path.join(args.photos_dir, key_photo)
        img = Image.open(input_path)

        output_path = os.path.join(args.output_dir, 'key_photos', key_photo)
        img.save(output_path)


if __name__ == '__main__':
    main()
