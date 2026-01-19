#!/bin/bash

if test "$#" -ne 2; then
    echo "Illegal number of parameters"
    exit 2
fi

echo "Extracting global features"
python tasks/global_feature_extraction.py --input_dir $1 --output_path "$2/features.json"

echo "Connecting photos"
python tasks/hierarchical_clustering.py --global_features "$2/features.json" --output_dir "$2/clustering"

echo "Predicting 3D points"
python tasks/groups_to_3D.py --from_hierarchical_clustering "$2/clustering" --dataset_path $1 --output_dir "$2/preds"

echo "Aligning scenes"
python tasks/hierarchical_align.py --preds "$2/preds/" --clusters "$2/clustering/"

echo "Done!"
