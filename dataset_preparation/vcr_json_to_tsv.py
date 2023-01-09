import argparse
import base64
import json
import pickle
from collections import Counter
from io import BytesIO
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import utils.transforms as T


def box_transform(target, max_size=480):
    h, w = target["size"][0], target["size"][1]
    boxes = target["boxes"]
    boxes = boxes / max_size
    scaled_boxes = boxes * torch.as_tensor([max_size / w, max_size / h, max_size / w, max_size / h])
    target["boxes"] = scaled_boxes
    return target

def encode_img(img):
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)  # bytes
    return base64_str.decode("utf-8")  # str


def count_labels(sample):
    labels = []
    counter = {}
    for obj in sample["objects"]:
        num = counter.get(obj, 0)
        counter[obj] = num + 1
        labels.append(f"{obj}{num}")
    return labels


def replace_objects(vals, labels, boxes):
    out = []
    for word in vals:
        if type(word) is list:
            objs = []
            for obj in word:
                label = labels[obj]
                box = boxes[obj]
                obj_str = f"{label} ( {box} )"
                objs.append(obj_str)
            out.append(" and ".join(objs))
        else:
            out.append(word)
    o = " ".join(out)
    o = o.replace(" , ", ", ").replace(" ' ", "'").replace(" .", "")
    return o


def to_int(x):
    return int((x * (1000 - 1)).round())


def to_bin(x):
    return f"<bin_{to_int(x)}>"


def coords_to_bins(coords, image):
    boxes_target = {"boxes": torch.tensor([[float(x) for x in box[:-1]] for box in coords]),
                    "size": image.size[::-1]}
    patch_boxes = box_transform(boxes_target, max_size=480)
    output_boxes = [" ".join(to_bin(val) for val in box) for box in patch_boxes['boxes']]
    output_boxes_int = [[to_int(val) for val in box] for box in patch_boxes['boxes']]

    return output_boxes, output_boxes_int


def main(args):
    assert len(args.path_to_json) == len(args.path_to_dataset), "Must give equal amount of json files and dataset paths"

    # Save stats
    statistics = {"datasets": {}}

    # Support processing multiple json files
    for json_idx, json_path in enumerate(args.path_to_json):
        dataset_path = Path(args.path_to_dataset[json_idx])
        json_path = Path(json_path)
        assert json_path.is_file() and json_path.suffix == ".jsonl", f"{json_path} is not a valid jsonl file."

        print(f"Processing {json_path.name} with dataset at {str(dataset_path)}:")
        name = json_path.stem
        data_list = []
        with open(json_path) as f:
            for line in tqdm(f):
                data_list.append(json.loads(line))
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
            for sample in data_iterator:
                fn = sample["metadata_fn"]
                with open(f"{dataset_path}/{fn}") as meta_file:
                    a = json.load(meta_file)
                image = Image.open(f"{dataset_path}/{sample['img_fn']}")
                boxes, box_ints = coords_to_bins(a["boxes"], image)
                labels = count_labels(sample)
                question = replace_objects(sample["question"], labels, boxes)
                answers = [replace_objects(ans, labels, boxes) for ans in sample["answer_choices"]]
                answer_string = ". ".join([f"answer{i}: {answer}" for i, answer in enumerate(answers)])
                net_input = f"{question} {answer_string}"
                correct_answer = f"answer{sample['answer_label']}"
                explanation = sample['rationale_choices'][sample['rationale_label']]
                expl_string = replace_objects(explanation, labels, boxes)
                encoded_image = encode_img(image)
                row_data = [
                    sample["annot_id"],  # Question id
                    sample["img_id"],  # Image id
                    encoded_image,  # base64 encoded image
                    net_input,  # Hypothesis
                    expl_string,  # Explanation
                    correct_answer,  # answer0, answer1, answer2, answer3
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
