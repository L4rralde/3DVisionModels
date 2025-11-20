import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
from PIL import Image

import src.glueing as glueing


def check_args(args):
    if args.save_photos and args.photos_dir == '':
        raise ValueError("To save photos, --photos_dir must be provided")
    if args.save_photos and not os.path.exists(args.photos_dir):
        raise ValueError(f"--photos_dir:{args.photos_dir} does not exist.")

    return args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--global_features', required=True)
    parser.add_argument('--photos_dir', default='')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--filter_l', default=0.3)
    parser.add_argument('--filter_h', default=0.8)
    parser.add_argument('--save_photos', action='store_true')

    args = parser.parse_args()
    args = check_args(args)

    return args



def main():
    args = parse_args()

    #1. Read global features
    features_df = glueing.FeaturesDf.from_json(
        args.global_features,
        args.photos_dir
    )

    #2. Filter out duplicates and outliers
    filtered_feats = glueing.filter_features_df(features_df, args.filter_l, args.filter_h)

    #3. Build tree/graph
    graph = glueing.hierarchical_clustering(filtered_feats)

    #4. Dump results
    #4.1 Dump cliustering info
    graph.save(args.output_dir)
    #4.2 Dump key photos file
    key_photos = graph.key_photos
    key_photos_path = os.path.join(args.output_dir, 'key_photos.txt')
    with open(key_photos_path, 'w') as file:
        for item in key_photos:
            file.write(item + '\n')

    #(Optional)4.3 Dump photos
    if not args.save_photos:
        return

    #4.3.1 Dump photos hierarchy
    graph.save_photos(args.output_dir)
    
    #4.3.2 Dump key photos.
    os.makedirs(os.path.join(args.output_dir, 'key_photos'))
    for key_photo in key_photos:
        input_path = os.path.join(args.photos_dir, key_photo)
        img = Image.open(input_path)

        output_path = os.path.join(args.output_dir, 'key_photos', key_photo)
        img.save(output_path)


if __name__ == '__main__':
    main()
