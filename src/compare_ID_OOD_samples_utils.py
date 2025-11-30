from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image
import torch

from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
    CLIPProcessor,
    CLIPModel,
)
from datasets import get_dataset_class
from utils import seed_hash
from data import split_dataset


def create_splits_and_samples(
    dataset_name: str,
    root: str,
    test_envs: List[int],
    hparams: Dict,
    trial_seed: int,
    holdout_fraction: float,
    selection_path: str,
    num_OOD_samples: int,
    num_ID_samples: int,
    label_idx: Optional[int] = None,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Create dataset splits and sample ID/OOD image-label pairs for comparison.

    This function:
    1. Loads the dataset class and creates training/validation splits (out_envs)
    2. Loads a PyTorch .pt file of OOD scores (selection)
    3. Creates test_sorted_file_label_pairs for each environment in test_envs
    4. Generates a list of in-distribution samples by subsampling each env
    5. Returns both the ID samples and OOD samples as lists of (image_path, label) tuples

    Args:
        dataset_name: Name of the dataset to load
        root: Root directory for the dataset
        test_envs: List of environment indices that are test environments
        hparams: Hyperparameters dictionary for dataset initialization
        trial_seed: Random seed for reproducibility
        holdout_fraction: Fraction of data to hold out for validation
        selection_path: Path to PyTorch .pt file containing OOD scores
        num_OOD_samples: Maximum number of OOD samples to return
        num_ID_samples: Target number of ID samples to return
        label_idx: Optional label index to filter samples (if None, uses all labels)

    Returns:
        Tuple containing:
            - train_sorted_file_label_pairs: List of (image_path, label) tuples for ID samples
            - test_sorted_file_label_pairs: List of (image_path, label) tuples for OOD samples,
              sorted by OOD score (highest first)
    """

    # 1) Create the dataset and environment splits
    dataset_class = get_dataset_class(dataset_name)
    dataset = dataset_class(root, test_envs, hparams)

    out_envs = []
    orders = []
    for env_i, env in enumerate(dataset):
        # Split into "out" (validation) and "in" (training) sets
        out, _ = split_dataset(
            env,
            int(len(env) * holdout_fraction),
            seed_hash(trial_seed, env_i)
        )
        # Store a permutation of indices for reproducibility
        keys = list(range(len(env)))
        np.random.RandomState(seed_hash(trial_seed, env_i)).shuffle(keys)
        orders.append(keys.copy())
        out_envs.append(out)

    # 2) Load OOD scores (selection), sort descending
    selection = torch.load(selection_path)
    sorted_ood_indices = np.argsort(-selection)  # Sort descending (large->small)
    num_ood_samples = min(num_OOD_samples, len(sorted_ood_indices))

    # 3) Build file_label_pairs for each environment
    file_label_pairs: Dict[int, List[Tuple[str, int]]] = {}
    test_sorted_file_label_pairs: List[Tuple[str, int]] = []
    for i, env in enumerate(out_envs):
        # Build file_label_pairs for all examples in out_env: (img_path, label)
        file_label_pairs[i] = [
            env.underlying_dataset.samples[k] for k in orders[i][: len(env)]
        ]

        # For test env, produce the sorted list by OOD selection score
        if i in test_envs:
            test_sorted_file_label_pairs = [
                env.underlying_dataset.samples[orders[i][: len(env)][k]]
                for k in sorted_ood_indices
            ][:num_ood_samples]

    # 4) Collect "in-distribution" samples from all non-test envs
    samples = []
    for env_idx, pairs in file_label_pairs.items():
        if env_idx in test_envs:
            continue
        # Attach environment index so we know from which env it came
        samples += [(img_path, label, env_idx) for (img_path, label) in pairs]

    samples_df = pd.DataFrame(samples, columns=["img_path", "label", "env"])
    samples_df = samples_df.sample(n=samples_df.shape[0], random_state=trial_seed)

    if label_idx is not None:
        samples_df = samples_df[samples_df["label"] == label_idx]

    # 5) Sample equally from each environment to get num_ID_samples
    num_envs_for_ID = len(file_label_pairs) - len(test_envs)
    n_per_env = min(num_ID_samples, samples_df.shape[0]) // max(num_envs_for_ID, 1)

    # Sample from each env: randomly pick n_per_env from each
    final_samples_df = (
        samples_df.groupby("env", group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), n_per_env), random_state=trial_seed))
        .reset_index(drop=True)
    )

    train_sorted_file_label_pairs = list(
        zip(final_samples_df["img_path"], final_samples_df["label"])
    )

    # 6) Return the final in-distribution samples plus the test-sorted OOD info
    return train_sorted_file_label_pairs, test_sorted_file_label_pairs


def generate_captions(
    image_paths: List[str],
    caption_model_id: str,
    prompt: str,
    batch_size: int = 32,
    max_new_tokens: int = 512,
) -> List[str]:
    """
    Generate captions for a list of image paths using a BLIP2-based model.

    Processes images in batches for efficiency. Handles broken images gracefully
    by returning "CAPTION_ERROR" as a placeholder.

    Args:
        image_paths: List of paths to image files
        caption_model_id: HuggingFace model ID for BLIP2 captioning model
        prompt: Text prompt to condition caption generation
        batch_size: Number of images to process per batch
        max_new_tokens: Maximum number of tokens to generate per caption

    Returns:
        List of generated captions (one per image_path). Failed images return "CAPTION_ERROR"
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and processor just once
    processor = Blip2Processor.from_pretrained(caption_model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(caption_model_id)
    model.eval().to(device)

    captions = []
    # Process image_paths in batches
    for start_idx in tqdm(range(0, len(image_paths), batch_size), desc="Generating captions"):
        batch_paths = image_paths[start_idx : start_idx + batch_size]

        # Load images into a list
        images = []
        for path in batch_paths:
            try:
                images.append(Image.open(path).convert("RGB"))
            except Exception as e:
                print(f"Warning: Failed to open {path}: {e}")
                # Insert a placeholder so we can keep alignment in captions
                images.append(None)

        # Preprocess images in one go, ignoring any broken images
        valid_imgs = [img for img in images if img is not None]
        inputs = processor(
            images=valid_imgs,
            text=[prompt] * len(valid_imgs),
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            # Generate outputs (length = number of valid images)
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )

        # Decode each caption
        batch_captions = []
        valid_idx = 0  # Index into generated_ids
        for img in images:
            if img is None:
                # Broken image -> placeholder caption
                batch_captions.append("CAPTION_ERROR")
            else:
                cap = processor.tokenizer.decode(
                    generated_ids[valid_idx], skip_special_tokens=True
                )
                batch_captions.append(cap)
                valid_idx += 1

        captions.extend(batch_captions)

    return captions


def generate_difference_captions(
    ID_captions: List[str],
    OOD_captions: List[str],
    proposer_model_id: str,
    dataset_prompt: str = "",
    max_new_tokens: int = 500,
    num_difference_captions: int = 25,
    max_ID_captions: int = 10,
    max_OOD_captions: int = 10,
    seed: int = 0,
) -> List[str]:
    """
    Compare ID and OOD caption sets using an LLM to find bullet-point differences.

    Uses a language model to identify properties that are more likely to be true
    for OOD samples compared to ID samples. Returns a list of difference captions
    in bullet-point format.

    Args:
        ID_captions: List of captions from in-distribution samples
        OOD_captions: List of captions from out-of-distribution samples
        proposer_model_id: HuggingFace model ID for the LLM used to propose differences
        dataset_prompt: Optional additional context about the dataset
        max_new_tokens: Maximum number of tokens to generate
        num_difference_captions: Target number of difference captions to generate
        max_ID_captions: Maximum number of ID captions to use in the prompt
        max_OOD_captions: Maximum number of OOD captions to use in the prompt
        seed: Random seed for shuffling captions

    Returns:
        List of difference captions (strings) extracted from the model's bullet-point response
    """
    np.random.RandomState(seed).shuffle(ID_captions)
    ID_captions = ID_captions[:max_ID_captions]
    np.random.RandomState(seed).shuffle(OOD_captions)
    OOD_captions = OOD_captions[:max_OOD_captions]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(proposer_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        proposer_model_id,
        torch_dtype=(torch.float16 if device == "cuda" else torch.float32),
        device_map="auto",
    )
    model.eval()

    prompt = f"""
I am a machine learning researcher trying to figure out the major differences between these two groups so I can better understand my data. {dataset_prompt}
Come up with {num_difference_captions} distinct concepts that are more likely to be true for Out-of-Distribution Group compared to In-Distribution Group. Please write a list of captions (separated by bullet points "*"). For example:
* "unusual lighting conditions"
* "visual distortions"
* "complex backgrounds"
* "non-standard object poses"
* "uncommon viewing angles"
* "partial views of objects"
* "objects in unexpected contexts"
* "scenes with high visual clutter"
* "images with unusual color schemes"
* "low-resolution images"
Do not talk about the caption, e.g., "caption with one word" and do not list more than one concept. The hypothesis should be a caption, so hypotheses like "more of ...", "presence of ...", "images with ..." are incorrect. Also do not enumerate possibilities within parentheses. Here are examples of bad outputs and their corrections:
* INCORRECT: "various nature environments like lakes, forests, and mountains" CORRECTED: "nature"
* INCORRECT: "images of household object (e.g. bowl, vacuum, lamp)" CORRECTED: "household objects"
* INCORRECT: "Presence of baby animals" CORRECTED: "baby animals"
* INCORRECT: "Images involving interaction between humans and animals" CORRECTED: "interaction between humans and animals"
* INCORRECT: "More realistic images" CORRECTED: "realistic images"
Again, I want to figure out what kind of distribution shift are there. {dataset_prompt} List {num_difference_captions} properties that hold more often for the images (not captions) in Out-of-Distribution Group compared to In-Distribution Group. Answer with a list
(separated by bullet points "*").

In-Distribution Group:
{chr(10).join(ID_captions)}

Out-of-Distribution Group:
{chr(10).join(OOD_captions)}

Your response:
    """.strip()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the difference text after "Your response:"
    diffs_raw = response.split("Your response:")[-1].strip()

    # Look for lines that start with "*" to interpret them as bullet points
    difference_captions = []
    for line in diffs_raw.splitlines():
        line = line.strip()
        if line.startswith("*"):
            # Remove the "*" and any leading space
            line = line.lstrip("*").strip()
            difference_captions.append(line)

    # Clean up GPU memory
    del model, tokenizer
    torch.cuda.empty_cache()
    return difference_captions


def compute_similarity_deltas(
    image_paths_ID: List[str],
    image_paths_OOD: List[str],
    difference_captions: List[str],
    ranker_model_id: str,
) -> pd.DataFrame:
    """
    Compute similarity deltas between ID and OOD images for each difference caption.

    Uses CLIP to compute cosine similarity between each difference caption and the
    image sets. Returns a DataFrame sorted by relative difference (descending).

    Args:
        image_paths_ID: List of paths to in-distribution images
        image_paths_OOD: List of paths to out-of-distribution images
        difference_captions: List of difference captions to evaluate
        ranker_model_id: HuggingFace model ID for CLIP model

    Returns:
        DataFrame with columns:
            - difference_caption: The difference caption text
            - avg_ID: Average similarity between caption and ID images
            - avg_OOD: Average similarity between caption and OOD images
            - sim_diff: Absolute difference in similarities (|avg_ID - avg_OOD|)
            - rel_diff: Relative difference (sim_diff / avg_ID)
        Sorted by rel_diff in descending order.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and processor
    clip_model = CLIPModel.from_pretrained(ranker_model_id).to(device)
    clip_processor = CLIPProcessor.from_pretrained(ranker_model_id)
    clip_model.eval()

    # Helper: compute average similarity of a given text vs. a list of images
    def get_avg_similarity(img_paths: List[str], text: str) -> float:
        """Compute average cosine similarity between text and images."""
        images = []
        for path in img_paths:
            try:
                images.append(Image.open(path).convert("RGB"))
            except Exception as e:
                print(f"Warning: failed to load {path}: {e}")
        if not images:
            return 0.0

        inputs = clip_processor(
            text=[text] * len(images),
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = clip_model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

        # L2-normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # Cosine similarity is just dot product after normalization
        sims = (image_embeds * text_embeds).sum(dim=-1)
        return sims.mean().item()

    # Compute absolute sim-diffs for each difference caption
    results = []
    for diff_cap in tqdm(difference_captions, desc="Computing similarity deltas"):
        avg_ID = get_avg_similarity(image_paths_ID, diff_cap)
        avg_OOD = get_avg_similarity(image_paths_OOD, diff_cap)
        sim_diff = abs(avg_ID - avg_OOD)
        rel_diff = sim_diff / max(avg_ID, 1e-8)  # Avoid division by zero
        results.append((diff_cap, avg_ID, avg_OOD, sim_diff, rel_diff))

    df = pd.DataFrame(
        results,
        columns=["difference_caption", "avg_ID", "avg_OOD", "sim_diff", "rel_diff"],
    )
    df = df.sort_values(by="rel_diff", ascending=False)

    # Clean up GPU memory
    del clip_model, clip_processor
    torch.cuda.empty_cache()

    return df
