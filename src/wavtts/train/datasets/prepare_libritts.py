import os
import sys


sys.path.append(os.getcwd())

import json
from concurrent.futures import ProcessPoolExecutor
from importlib.resources import files
from pathlib import Path

import soundfile as sf
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm


def deal_with_audio_dir(audio_dir):
    sub_result, durations = [], []
    vocab_set = set()
    missing_text = 0
    audio_lists = list(audio_dir.rglob("*.wav"))

    for line in audio_lists:
        text_path = line.with_suffix(".normalized.txt")
        # LibriTTS_R ships some clips without their transcript. Skipping them rather than
        # writing a blank one: nothing in the unconditional model reads the text column,
        # but a text-conditioned reader of this arrow would get silent garbage instead of
        # an obvious gap. The count is reported at the end so the loss is never invisible.
        if not text_path.exists():
            missing_text += 1
            continue
        text = open(text_path, "r").read().strip()
        duration = sf.info(str(line)).duration
        if duration < 0.4 or duration > 30:
            continue
        sub_result.append({"audio_path": str(line), "text": text, "duration": duration})
        durations.append(duration)
        vocab_set.update(list(text))
    return sub_result, durations, vocab_set, missing_text


def main():
    result = []
    duration_list = []
    text_vocab_set = set()

    # process raw data
    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures = []

    for subset in tqdm(SUB_SET):
        dataset_path = Path(os.path.join(dataset_dir, subset))
        [
            futures.append(executor.submit(deal_with_audio_dir, audio_dir))
            for audio_dir in dataset_path.iterdir()
            if audio_dir.is_dir()
        ]
    missing_text = 0
    for future in tqdm(futures, total=len(futures)):
        sub_result, durations, vocab_set, missing = future.result()
        result.extend(sub_result)
        duration_list.extend(durations)
        text_vocab_set.update(vocab_set)
        missing_text += missing
    executor.shutdown()

    # save preprocessed dataset to disk
    if not os.path.exists(f"{save_dir}"):
        os.makedirs(f"{save_dir}")
    print(f"\nSaving to {save_dir} ...")

    with ArrowWriter(path=f"{save_dir}/raw.arrow") as writer:
        for line in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(line)
        writer.finalize()

    # dup a json separately saving duration in case for DynamicBatchSampler ease
    with open(f"{save_dir}/duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # vocab map, i.e. tokenizer
    with open(f"{save_dir}/vocab.txt", "w") as f:
        for vocab in sorted(text_vocab_set):
            f.write(vocab + "\n")

    print(f"\nFor {dataset_name}, sample count: {len(result)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}")
    print(f"For {dataset_name}, total {sum(duration_list) / 3600:.2f} hours")
    if missing_text:
        print(f"For {dataset_name}, skipped {missing_text} clips with no .normalized.txt")


if __name__ == "__main__":
    max_workers = 36

    tokenizer = "char"  # "pinyin" | "char"

    SUB_SET = ["train-clean-100", "train-clean-360"]
    # Corpus root and the name to save under, both overridable:
    #   python src/wavtts/train/datasets/prepare_libritts.py <dataset_dir> <dataset_name>
    # LibriTTS_R is LibriTTS restored — same segmentation, same filenames, same subset
    # layout — so it prepares through this script unchanged, just from another root.
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/public_datasets/LibriTTS/LibriTTS"
    # clean-100 + clean-360 = the standard "clean-460" pool; keep the name short so it
    # reads as one dataset in ckpt/run dir names rather than a list of subsets
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else "LibriTTS_460"
    save_dir = str(files("wavtts").joinpath("../../")) + f"/data/{dataset_name}"
    print(f"\nPrepare for {dataset_name}, will save to {save_dir}\n")
    main()

    # For LibriTTS_100_360_500_char, sample count: 354218
    # For LibriTTS_100_360_500_char, vocab size is: 78
    # For LibriTTS_100_360_500_char, total 554.09 hours
