cat data/esnlive/esnlive_train.tsv data/vqax/train_x.tsv data/vcr/vcr_train_split.tsv > data/unified_train.tsv
cat data/esnlive/esnlive_dev.tsv data/vqax/val_x.tsv data/vcr/vcr_val_split.tsv > data/unified_val.tsv
cat data/esnlive/esnlive_test.tsv data/vqax/test_x.tsv data/vcr/vcr_test_split.tsv > data/unified_test.tsv