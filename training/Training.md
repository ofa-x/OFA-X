# How to run experiments with OFA-X

1. Start your development server, navigate to your desired working directory and download this repository
2. Prepare datasets according to the [dataset preparation README](./dataset_preparation/Datasets.md)
3. Install the requirements
```
# [optional] create a virtual environment: conda create -n ofa-x python=3.10 -y && conda activate ofa-x
cd OFA
pip install -r requirements.txt
cd ..
```
4. Download the checkpoint for fine-tuning (see [OFA README](./OFA/checkpoints.md) for options)
5. Run the training script for the desired task
```
cd training
bash ./run_scripts/run_<task>_training.sh <config_file> <run_name> <checkpoint_path>
```
6. Download cococaptions for evaluation. We use the coco captions
repo provided by [e-ViL](https://github.com/maximek3/e-ViL).
```
wget -O ../OFA/cococaption.zip https://drive.google.com/uc?export\=download\&id\=1nLlrtQlsP5kSeB9L0PTle4PeGL9CJg29\&confirm\=t\&uuid=5c455b29-2b83-4490-af3a-cec071b3b40b
unzip ../OFA/cococaption.zip
```
7. After training was finished, you can run the evaluation script
```
bash evaluation/<task>/evaluate.sh <checkpoint_path> <run_name> <data>
sudo apt-get install default-jdk # for meteor to work properly
bash evaluation/<task>/evaluate_from_predictions.sh <run_name> <data>
```