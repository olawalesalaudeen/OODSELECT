#!/usr/bin/env python
# coding: utf-8

import os
import sys
sys.path.insert(0, '../src/')
import argparse
import pandas as pd
from tqdm.auto import tqdm

from compare_ID_OOD_samples_utils import create_splits_and_samples, generate_captions, generate_difference_captions, compute_similarity_deltas
from class_names import get_class_names_dict
from prompts import get_caption_prompt, get_difference_caption_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Combine caption generation, difference captioning, and ranker steps.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name.")
    parser.add_argument("--train_envs", type=int, nargs="+", required=True,
                        help="Indices of train environments to use for ID.")
    parser.add_argument("--test_envs", type=int, nargs="+", required=True,
                        help="Index of test environment to use for OOD selection.")
    parser.add_argument("--selection_path", type=str, required=True,
                        help="Path to OOD selection vector.")

    parser.add_argument("--trial_seed", type=int, default=0,
                        help="Random seed for trial.")
    parser.add_argument("--holdout_fraction", type=float, default=0.2,
                        help="Fraction of dataset to hold out for validation.")
    parser.add_argument("--num_ID_samples", type=int, default=200,
                        help="Number of in-distribution samples to take from the start of CSV rows.")
    parser.add_argument("--num_OOD_samples", type=int, default=200,
                        help="Number of OOD samples to take from the start of CSV rows.")
    parser.add_argument("--label_idx", type=int, default=None,
                        help="Label to compare; if None, all labels are used.")
    parser.add_argument("--num_difference_captions", type=int, default=25,
                        help="Number of difference captions to generate.")
    parser.add_argument("--max_ID_captions", type=int, default=25,
                        help="Number of ID captions to generate.")
    parser.add_argument("--max_OOD_captions", type=int, default=25,
                        help="Number of OOD captions to generate.")

    parser.add_argument("--output_dir", type=str, default="./compare_selected_OOD_samples",
                        help="Directory to save intermediate CSVs and final result CSV.")
    parser.add_argument("--data_dir", type=str, default="",
                        help="Directory to save intermediate CSVs and final result CSV.")
    parser.add_argument("--caption_model_id", type=str, default="Salesforce/blip2-flan-t5-xl",
                        help="HuggingFace model ID for the BLIP2 (or other) captioning model.")
    parser.add_argument("--proposer_model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="HuggingFace model ID for the LLM used to generate difference captions.")
    parser.add_argument("--ranker_model_id", type=str, default="openai/clip-vit-base-patch32",
                        help="HuggingFace model ID for the CLIP ranker.")
    return parser.parse_args()


HPARAMS = {
    "data_augmentation": False,
    "weights": "IMAGENET1K_V1",
}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)


    train_sorted_file_label_pairs, test_sorted_file_label_pairs = create_splits_and_samples(
        args.dataset,
        args.data_dir,
        args.test_envs,
        HPARAMS,
        args.trial_seed,
        args.holdout_fraction,
        args.selection_path,
        args.num_OOD_samples,
        args.num_ID_samples,
        args.label_idx
    )

    train_img_paths = [i[0] for i in train_sorted_file_label_pairs]
    test_img_paths = [i[0] for i in test_sorted_file_label_pairs]

    file_label_pairs = [[i, j] for i, j in train_sorted_file_label_pairs + test_sorted_file_label_pairs]
    # Step 1: Generate captions
    print("[INFO] Generating captions...")
    id_captions = generate_captions(train_img_paths, args.caption_model_id, prompt=get_caption_prompt(args.dataset))
    ood_captions = generate_captions(test_img_paths, args.caption_model_id, prompt=get_caption_prompt(args.dataset))

    df = pd.DataFrame(file_label_pairs, columns=["img_path", "label"])
    df["caption"] = id_captions + ood_captions

    # Step 2: Generate difference captions (only for one particular class, or for all?)
    # Here we'll just do one set vs. the rest as an example.
    # If you have multiple classes, you might want to loop over classes or subsets.

    # In many experiments, you might want to check a particular class: e.g., class == 0
    # For simplicity, we'll just use the entire set. Adjust to your needs.
    print("[INFO] Generating difference captions via LLM...")
    difference_captions = generate_difference_captions(
        id_captions,
        ood_captions,
        proposer_model_id=args.proposer_model_id,
        num_difference_captions=args.num_difference_captions,
        max_ID_captions=args.max_ID_captions,
        max_OOD_captions=args.max_OOD_captions
    )


    # Step 3: Compute similarity deltas for each difference_ column
    id_img_paths = [i[0] for i in train_sorted_file_label_pairs]
    ood_img_paths = [i[0] for i in test_sorted_file_label_pairs]
    print("[INFO] Computing similarity deltas with CLIP ranker...")
    diffs_df = compute_similarity_deltas(id_img_paths, ood_img_paths,
                                         difference_captions, args.ranker_model_id)

    class_names_dict = get_class_names_dict(args.dataset)
    # Write final result
    final_csv = os.path.join(args.output_dir,
                             f"ID_OOD_samples_captions_{args.dataset}_{'-'.join(map(str, args.train_envs))}_{args.num_ID_samples}_{'-'.join(map(str, args.test_envs))}_{args.num_OOD_samples}_{class_names_dict[args.label_idx] if args.label_idx is not None else 'all'}.csv")
    df.to_csv(final_csv, index=False)
    diffs_csv = os.path.join(args.output_dir, f"ID_OOD_difference_similarities_{args.dataset}_{'-'.join(map(str, args.train_envs))}_{args.num_ID_samples}_{'-'.join(map(str, args.test_envs))}_{args.num_OOD_samples}_{class_names_dict[args.label_idx] if args.label_idx is not None else 'all'}.csv")
    diffs_df.to_csv(diffs_csv, index=False)
    print(f"[INFO] Wrote final similarity differences to {final_csv}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
