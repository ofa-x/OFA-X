import argparse
import base64
import json
import pickle
from collections import Counter
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

    # Save stats
    statistics = {"datasets": {}}

    # Support processing multiple json files
    for json_idx, json_path in enumerate(args.path_to_json):
        dataset_path = Path(args.path_to_dataset[json_idx])
        json_path = Path(json_path)
        assert json_path.is_file() and json_path.suffix == ".json", f"{json_path} is not a valid json file."

        print(f"Processing {json_path.name} with dataset at {str(dataset_path)}:")
        name = json_path.stem
        data_list = json.loads(json_path.read_text())
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
            data_iterator = tqdm(data_list, desc="- Processing dataset rows", total=len(data_list))
            for obj in data_iterator:
                img_name = obj["question_id"].split("#")[0]
                img_path = dataset_path / img_name
                # Encode img to base64 string
                encoded_img = encode_img(img_path)
                label = obj["label"]
                explanation = obj["explanation"][0].rstrip(".").lower()
                row_data = [
                    obj["question_id"],     # Question id
                    obj["img_id"],          # Image id
                    encoded_img,            # base64 encoded image
                    obj["sent"],            # Hypothesis
                    explanation,            # Explanation
                    label,                  # Label (Entailment, Neutral, Contradiction)
                ]
                # remove tabs or double spaces from items with regex
                for i, item in enumerate(row_data):
                    if isinstance(item, str):
                        row_data[i] = item.replace("\t", " ").replace("  ", " ")
                row_string = "\t".join(row_data) + "\n"
                outfile.write(row_string)
                total_lines += 1
        print(f"- Completed processing {json_path.stem}")
        print(f"- Final dataset has a total of {total_lines} data points")
        statistics["datasets"][name] = {
            "samples_processed": len(data_list),
            "total_rows": total_lines
        }

    stats_file = Path(args.output_dir) / "stats.json"
    json.dump(statistics, stats_file.open("w"))
    print("Success!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_json", type=str, default=["./"], nargs="*",
                        help="The path(s) to the json file(s)")
    parser.add_argument("--path_to_dataset", type=str, default=["./"], nargs="*",
                        help="The path(s) to the corresponding dataset(s)")
    parser.add_argument("--output_dir", type=str, default="./",
                        help="The directory to output files")
    parser.add_argument("-y", action="store_true", help="Answer yes to all overwrite asks")
    main(parser.parse_args())
