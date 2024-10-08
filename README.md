# Better Safe than Sorry: Pre-training CLIP against Targeted Data Poisoning and Backdoor Attacks

[[Paper](https://arxiv.org/abs/2310.05862)]

## Overview
SafeCLIP is a strong defensive method for robust pre-training multimodal vision-language models against targeted data poisoning and backdoor attacks without any performance compromises.

### Training 

```
python -m src.main --name exp1 --train_data <path to train csv file> --validation_data <path to valid csv file>
--image_key <column name of the image paths in the train/validation csv file> --caption_key <column name of the captions
in the train/validation csv file> --device_ids 0 1 2 3 --distributed --memory_bank_size <memory bank size> --memory_bank --filter_lr <lr for cross modal warm up> --inmodal_warmup <inmodal warmup epochs> --multimodal_warmup <crossmodal warmup epochs> 
```

Your train/validation csv/tsv file should have 2 columns containing captions and the path to corresponding images on the machine. this script does not download the images for the captions directly. To download the images from their URL for CC3M and/or CC12M, use our `utils/download.py` script.

### Inference - ImageNet1K

```
python -m src.main --name <eval_imagenet_1k> --eval_data_type <dataset> --eval_test_data_dir data/ImageNet1K/validation/ --device_id 0 --checkpoint <ckpts/epoch_64.pt> 
```

For ImageNet1K: There should be a labels.csv in the test data directory that contains 2 columns -- image, label. image should have the location to the image in the local machine.

## Acknowledgements
Some code in this repo comes from the following repositories:

[RoCLIP](https://github.com/BigML-CS-UCLA/RoCLIP)

[CyCLIP](https://github.com/goel-shashank/CyCLIP)

[mlfoundations](https://github.com/mlfoundations/open_clip)  

[openai](https://github.com/openai/CLIP)

## Setup Environment and Install dependencies

The following commands create a conda environment inside the repository with the dependencies.

```bash
conda env create --prefix ./env -f environment.yml
source activate ./env
```

Then, to download the missing nltk_data

```
python -m nltk.downloader all

```

# Citation

Please cite our paper if you find the results or our code useful.
```

@article{yang2023better,
  title={Better safe than sorry: Pre-training clip against targeted data poisoning and backdoor attacks},
  author={Yang, Wenhan and Gao, Jingdong and Mirzasoleiman, Baharan},
  journal={arXiv preprint arXiv:2310.05862},
  year={2023}
}
```