import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from dataset import CIRCODataset

base_path = Path(__file__).absolute().parents[1].absolute()  # Getting the path to the base directory


def compute_metrics(data_path: Path, predictions_dict: Dict[int, List[int]], ranks: List[int]) -> Tuple[
    Dict[int, float], Dict[int, float]]:
    """Computes the Average Precision (AP) and Recall for a given set of predictions.

    Args:
        data_path (Path): Path where the CIRCO datasset is located
        predictions_dict (Dict[int, List[int]]): Predictions of image ids for each query id
        ranks (List[int]): Ranks to consider in the evaluation (e.g., [5, 10, 20])

    Returns:
        Tuple[Dict[int, float], Dict[int, float]]: Dictionaries with the AP and Recall for each rank
    """

    relative_val_dataset = CIRCODataset(data_path, split='val', mode='relative', preprocess=None)

    # Initialize empty dictionaries to store the AP and Recall values for each rank
    aps_atk = defaultdict(list)
    recalls_atk = defaultdict(list)

    # Iterate through each query id and its corresponding predictions
    for query_id, predictions in predictions_dict.items():
        target = relative_val_dataset.get_target_img_ids(int(query_id))
        gt_img_ids = target['gt_img_ids']
        target_img_id = target['target_img_id']

        # gt_img_ids = np.trim_zeros(gt_img_ids)  # remove trailing zeros added for collate_fn (when using dataloader)

        predictions = np.array(predictions, dtype=int)
        ap_labels = np.isin(predictions, gt_img_ids)
        precisions = np.cumsum(ap_labels, axis=0) * ap_labels  # Consider only positions corresponding to GTs
        precisions = precisions / np.arange(1, ap_labels.shape[0] + 1)  # Compute precision for each position

        # Compute the AP and Recall for the given ranks
        for rank in ranks:
            aps_atk[rank].append(float(np.sum(precisions[:rank]) / min(len(gt_img_ids), rank)))

        recall_labels = (predictions == target_img_id)
        for rank in ranks:
            recalls_atk[rank].append(float(np.sum(recall_labels[:rank])))

    # Compute the mean AP and Recall for each rank and store them in a dictionary
    ap_atk = {}
    recall_atk = {}
    for rank in ranks:
        ap_atk[rank] = float(np.mean(aps_atk[rank]))
        recall_atk[rank] = float(np.mean(recalls_atk[rank]))
    return ap_atk, recall_atk


def main():
    # Parse the arguments
    args = ArgumentParser()
    args.add_argument('--data_path', type=Path, default=base_path)
    args.add_argument('--ranks', type=int, nargs='+', default=[5, 10, 25, 50])
    args.add_argument('--predictions_file_path', type=Path,
                      default=base_path / 'submission_examples' / 'submission_val.json')
    args = args.parse_args()

    # Load the predictions from the given file
    try:
        with open(args.predictions_file_path, 'r') as f:
            predictions_dict = json.load(f)
    except FileNotFoundError as e:
        raise Exception("predictions_file_path must be a valid path to a json file")

    # Ensure that the query ids are consecutive and start from zero
    assert np.all(np.sort(np.array(list(predictions_dict.keys()), dtype=int)) == np.arange(
        len(predictions_dict.keys()))), "The keys of the predictions dictionary must be all the query ids"

    # Compute the metrics and print them
    ap_atk, recall_atk = compute_metrics(args.data_path, predictions_dict, args.ranks)

    print("\nWe remind that the mAP@k metrics are computed considering all the ground truth images for each query, the "
          "Recall@k metrics are computed considering only the target image for each query (the one we used to write "
          "the relative caption)")

    print("\nmAP@k metrics")
    for rank in args.ranks:
        print(f"mAP@{rank}: {ap_atk[rank] * 100:.2f}")
    print("\nRecall@k metrics")
    for rank in args.ranks:
        print(f"Recall@{rank}: {recall_atk[rank] * 100:.2f}")


if __name__ == '__main__':
    main()
