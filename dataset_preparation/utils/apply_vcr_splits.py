import json
from tqdm import tqdm

# VCR does not have a test set, so we use the validation set as the test set
# we follow the same split  for train and val as the e-ViL authors

# Read split data from csv from:
# https://drive.google.com/drive/folders/1REopdRzF1tgik22LHf2i85MMLXjconQK
# ( https://github.com/maximek3/e-ViL )
with open("../data_raw/vcr/vcr_train_split.json") as f:
    vcr_train = json.load(f)
with open("../data_raw/vcr/vcr_dev_split.json") as f:
    vcr_dev = json.load(f)
with open("../data_raw/vcr/vcr_valtest.json") as f:
    vcr_test = json.load(f)


train_ids = {sample["question_id"]: True for sample in vcr_train}
val_ids = {sample["question_id"]: True for sample in vcr_dev}
test_ids = {sample["question_id"]: True for sample in vcr_test}

# Prepare write files
new_train = open("../../vcr/train_split.tsv", "w")
new_val = open("../../data/vcr/val_split.tsv", "w")
new_test = open("../../data/vcr/test_split.tsv", "w")

# Open processed train dataset
with open("../../data/vcr/train.tsv") as f:
    for line in tqdm(f, desc="Processing train dataset"):
        vals = line.split("\t")
        id = vals[0]
        if id in train_ids:
            new_train.writelines([line])
        elif id in val_ids:
            new_val.writelines([line])
        elif id in test_ids:
            new_test.writelines([line])

# Open processed val dataset
with open("../../data/vcr/val.tsv") as f:
    for line in tqdm(f, desc="Processing val dataset"):
        vals = line.split("\t")
        id = vals[0]
        if id in train_ids:
            new_train.writelines([line])
        elif id in val_ids:
            new_val.writelines([line])
        elif id in test_ids:
            new_test.writelines([line])
new_train.close()
new_val.close()
new_test.close()
