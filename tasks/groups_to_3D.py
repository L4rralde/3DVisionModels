import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse

from tqdm import tqdm
import numpy as np

from src.recons3d import Client


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--from_path_list',
        nargs='*',
        default=[],
        help='List of image paths'
    )
    parser.add_argument(
        '--from_hierarchical_clustering',
        type=str,
        default='',
        help='Path to hierarchical clustering dir output'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='',
        help='Path to dir of set of photos used by any connecting method'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Path to directory to dump predictions'
    )
    
    args = parser.parse_args()
    check_args(args)
    return args


def check_args(args: argparse.Namespace) -> None:
    if args.from_path_list != []:
        if args.from_hierarchical_clustering or args.dataset_path:
            raise ValueError("arg --from_path_list is exclusive. No more images must be processed")
    
    elif not args.dataset_path:
        raise ValueError("When image paths are not passed explicitly, path to set of photos used must be passed with '--dataset_path'")



def main():
    args = parse_args()
    model = Client()

    if args.from_path_list != []:
        img_groups = {'all': args.from_path_list}
    elif args.from_hierarchical_clustering:
        from src.glueing import ConnectedClusterTree

        tree = ConnectedClusterTree.from_dir_output(args.from_hierarchical_clustering)
        clusters = tree.joint_clusters
        img_groups = {
            str(label): {
                os.path.join(args.dataset_path, path)
                for path in cluster_img_paths
            }
            for label, cluster_img_paths in clusters.items()
        }


    os.makedirs(args.output_dir, exist_ok=True)
    for group, img_path_list in tqdm(img_groups.items()):
        tqdm.write(f"Processing group {group} comprised of {len(img_path_list)} photos")
        prediction = model.infer(img_path_list)
        path = os.path.join(args.output_dir, f'{group}.npz')
        tqdm.write(f"Dumping predictions to {path}")
        np.savez(path, **prediction)


if __name__ == '__main__':
    main()