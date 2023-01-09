# Dataset Preparation
## VQA-X
1. Download MSCOCO 2014 Images from https://cocodataset.org/#download (train2014, val2014, test2014) and extract to `./dataset_preparation/data_raw/coco2014`.
2. Download VQA-X data from [Google drive](https://drive.google.com/drive/folders/1zPexyNo_W8L-FYq6iPcERQ5cJUUJzYhl).
3. Generate .tsv files according to [OFA documentation](https://github.com/OFA-Sys/OFA#visual-question-answering) by
running the `vqa_json_to_tsv.py` script. This will generate three .tsv files containing the dataset info in the format:
```
question_id img_id  question    conf|!+answer   explanation [empty] base64_encoded_image
```
Example:
```
79459   79459   is this person wearing shorts?  0.6|!+no    their pants are long    ""  /9j/4AAQS...tigZ/9k=
``` 
Note that there will be more dataset rows generated than there are samples in the VQA-X files.
We choose to create a separate datapoint for each possible 
answer option, following [OFA documentation](https://github.com/OFA-Sys/OFA#visual-question-answering).
```
python vqa_json_to_tsv.py \
--path_to_json ./train_x.json ./val_x.json ./test_x.json \
--path_to_dataset ./data_raw/coco2014/train2014 ./data_raw/coco2014/val2014 ./data_raw/coco2014/test2014 \
--output_dir ../data/vqax
```

## e-SNLI-VE
1. Download Flickr30k Images from https://www.kaggle.com/hsankesara/flickr-image-dataset.
2. Download e-SNLI-VE data according [e-ViL GitHub](https://github.com/maximek3/e-ViL#e-snli-ve-1).
3. Generate .tsv files according to [OFA documentation](https://github.com/OFA-Sys/OFA#visual-entailment) by
running the `esnlive_json_to_tsv.py` script. This will generate three .tsv files containing the dataset info in the format:
```
question_id img_id  base64_encoded_image   statement   explanation answer
```
Example:
```
4465359505.jpg#2r1c   flickr30k_004465359505.npz   /9j/4AAQS...tigZ/9k=  The old man is chopping down a tree in his yard.   	hair and tree are two different things  contradiction
```
Run the script:
```
python esnlive_json_to_tsv.py \
--path_to_json ./esnlive_train.json ./esnlive_dev.json ./esnlive_test.json \
--path_to_dataset ./data_raw/esnlive/flickr30k_images/flickr30k_images ./data_raw/esnlive/flickr30k_images/flickr30k_images ./data_raw/esnlive/flickr30k_images/flickr30k_images \
--output_dir ../data/esnlive
```

## VCR
1. Download VCR Images and annotations from https://visualcommonsense.com/download.html.
2. Generate .tsv files according to in the same format as e-SNLI-VE by 
running the `vcr_json_to_tsv.py` script. This will generate three .tsv files containing the dataset info in the format:
```
question_id img_id  base64_encoded_image    statement   explanation  answer
```
3. Split the dataset into train, val and test sets according to the split used by the e-ViL authors by first 
downloading the split files from [e-ViL GitHub](https://github.com/maximek3/e-ViL#vcr) and then running `utils/apply_vcr_splits.py`.

Example:
```
python vcr_json_to_tsv.py \
--path_to_json ./data_raw/vcr/train.jsonl ./data_raw/vcr/val.jsonl ./data_raw/vcr/test.jsonl \
--path_to_dataset ./data_raw/vcr/vcr1images ./data_raw/vcr/vcr1images ./data_raw/vcr/vcr1images \
--output_dir ../data/vcr

cd utils
python apply_vcr_splits.py
```

## Unifying the datasets
To train our model on the unified task of all three datasets, we need to combine the .tsv files into one file. For this,
we first need to reshape the VQA-X dataset to match the format of the other two datasets. This is done by running
`utils/reshape_vqax.py`. 
Afterwards, we must create a single .tsv file containing all the data from all three datasets. This is done by running
`utils/unify_datasets.py`. This will generate three .tsv files containing the dataset info in the format of e-SNLI-VE and VCR.
Optionally shuffle the data by running `utils/shuffle_dataset.py`. Adjust these files according to where your data is located.