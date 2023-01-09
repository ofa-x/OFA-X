from tqdm import tqdm

for file_name in ["train", "val", "test"]:
    with open(f"./data/vqax/{file_name}_x.tsv", "r") as f:
        with open(f"./data/reshaped_vqax/{file_name}_x.tsv", "w") as of:
            for line in tqdm(f):
                indices = [0, 6, 2, 3, 4, 5]
                columns = line.split("\t")
                uniq_id, _, question, ref, explanation, _, image = columns
                out_line = "\t".join([uniq_id, "_", image.strip(), question, explanation, ref])
                of.write(out_line + "\n")
