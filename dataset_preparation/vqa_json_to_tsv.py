import argparse
import base64
import json
import pickle
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


def encode_img(path):
    img = Image.open(path)
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)  # bytes
    return base64_str.decode("utf-8")  # str


def main(args):
    assert len(args.path_to_json) == len(args.path_to_dataset), "Must give equal amount of json files and dataset paths"

    # Accumulate vocabulary across files for final trainval_ans2label.pkl file
    vocab = []

    # Save stats
    statistics = {"datasets": {}}

    # Support processing multiple json files
    for json_idx, json_path in enumerate(args.path_to_json):
        dataset_path = Path(args.path_to_dataset[json_idx])
        json_path = Path(json_path)
        assert json_path.is_file() and json_path.suffix == ".json", f"{json_path} is not a valid json file."

        print(f"Processing {json_path.name} with dataset at {str(dataset_path)}:")
        name = json_path.stem
        content = pd.read_json(json_path)
        content.rename(columns={"sent": "question"}, inplace=True)
        content[["img_id", "question_id"]] = content[["img_id", "question_id"]].astype(str)
        output_file = Path(args.output_dir) / f"{name}.tsv"
        # Handle file existence
        if output_file.exists():
            print(f"- Output file {str(output_file)} already exists. Overwrite?")
            ans = input("y/n: ") if not args.y else "y"
            if ans == "n":
                print(f"- Cancelled processing {str(output_file)}")
                continue
            elif ans == "y":
                print(f"- Overwriting.")
            else:
                print(f"- Invalid input. Cancelled processing {str(output_file)}")
                continue

        output_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"- Writing output to {str(output_file)}")
        total_lines = 0
        with open(output_file, 'w') as outfile:
            # Iterate through data as namedtuples
            data_iterator = tqdm(content.itertuples(), desc="- Processing dataset rows", total=content.shape[0])
            for row in data_iterator:
                img_path = dataset_path / f"{row.img_id}.jpg"
                # Encode img to base64 string
                encoded_img = encode_img(img_path)
                # Generate one data point for each possible label
                for ans, conf in row.label.items():
                    # First explanation (add multiple for val_set?)
                    explanation = row.explanation[0]
                    explanation = explanation.rstrip(".")
                    # Add ans to vocab
                    vocab += ans.split()
                    vocab += explanation.split()
                    row_data = [
                        row.question_id,        # Question id
                        row.img_id,             # Image id
                        row.question,           # Question
                        f"{conf:.1f}|!+{ans}",  # Confidence and answer separated with |?+
                        explanation,            # Explanation
                        "",                     # Empty (instead of VinVL object labels)
                        encoded_img             # base64 encoded image
                    ]
                    row_string = "\t".join(row_data) + "\n"
                    outfile.write(row_string)
                    total_lines += 1
        print(f"- Completed processing {json_path.stem}")
        print(f"- Final dataset has a total of {total_lines} data points")
        statistics["datasets"][name] = {
            "samples_processed": content.shape[0],
            "total_rows": total_lines
        }

    # If answers were added to vocab, some datasets were processed
    if vocab:
        ans2label_file = Path(args.output_dir) / "trainval_ans2label.pkl"
        print(f"Generating ans2label mapping at {str(ans2label_file)}.")
        vocab = set(vocab)
        vocab = {ans: label for label, ans in enumerate(vocab)}
        if args.src_ans2label:
            print(f"Loading source ans2label from {args.src_ans2label}")
            with open(args.src_ans2label, "rb") as f:
                src_vocab = pickle.load(f)
            print(f"Merging vocabularies")
            vocab.update(src_vocab)
        if ans2label_file.exists():
            print(f"Ans2label file {str(ans2label_file)} already exists. Overwrite?")
            ans = input("y/n: ") if not args.y else "y"
            if ans == "n":
                print(f"Cancelled writing {str(ans2label_file)}")
                return
            elif ans == "y":
                print(f"Continuing.")
            else:
                print(f"Invalid input. Cancelled writing {str(ans2label_file)}")
                return
        pickle.dump(vocab, ans2label_file.open("wb"))
        print(f"Final vocabulary size: {len(vocab)}")
        statistics["vocab_length"] = len(vocab)
        stats_file = Path(args.output_dir) / "stats.json"
        json.dump(statistics, stats_file.open("w"))
        print("Success!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_json", type=str, default=["./"], nargs="*",
                        help="The path(s) to the json file(s)")
    parser.add_argument("--path_to_dataset", type=str, default=["./"], nargs="*",
                        help="The path(s) to the corresponding dataset(s)")
    parser.add_argument("--src_ans2label", type=str, default=None,
                        help="The path to the original vqa trainval_ans2label.pkl")
    parser.add_argument("--output_dir", type=str, default="./",
                        help="The directory to output files")
    parser.add_argument("-y", action="store_true", help="Answer yes to all overwrite asks")
    main(parser.parse_args())
