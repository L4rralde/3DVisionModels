import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse

from src.recons3d import get_model_cls, make_server, available_models


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, choices=available_models)
    return parser.parse_args()


def main():
    args = parse_args()
    model_cls = get_model_cls(args.name)
    model = model_cls()

    server = make_server(model)
    server.start()
    input("Press enter to exit...\n")


if __name__ == '__main__':
    main()

