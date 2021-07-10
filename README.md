# start_follow_read_py3

This repository is the implementation of the methods described in the paper [Start, Follow, Read: Full-Page End-to-end Handwriting Recognition](http://openaccess.thecvf.com/content_ECCV_2018/html/Curtis_Wigington_Start_Follow_Read_ECCV_2018_paper.html). The code is heavily based on the original code as released by the authors. Our repository has some minor changes to make it run with python3 and pytorch=1.3.

All steps in this repository are shown with the [ICDAR2017 Competition on Handwritten Text Recognition on the READ Dataset](https://scriptnet.iit.demokritos.gr/competitions/8/) and are the same as mentioned in the Author's repository.
This code is free for academic and research use. For commercial use of our code and methods please contact [BYU Tech Transfer](techtransfer.byu.edu).
The steps presented in this repository are to make the code running on single GPU. If you have high GPU resources available, please follow the base repository by authors to efficiently use the same.

## Dependencies

The dependencies are all found in `sfr_env_environment.yaml`. They are installed as follows.
```
conda env create -f sfr_env_environment.yaml
```

The environment is activated as `source activate sfr_env`.

As opposed to original SFR repository, this repository does not use LM decoding.

## Prepare Data

Download Train-A and Train-B from the competition [website](https://scriptnet.iit.demokritos.gr/competitions/8/). You need `Train-A.tbz2`, `Train-B_batch1.tbz2`, `Train-B_batch2.tbz2`. Put them in the data folder.

#### Extract Files

```
mkdir data
cd data
tar jxf Train-A.tbz2
tar jxf Train-B_batch1.tbz2
tar jxf Train-B_batch2.tbz2
cd ..
```
The data folder will have following contents:
Data
|-Train-A
|-Train-B_batch1
|-Train-B_batch2

#### Prepare Train-A

train-A is the subset of dataset with start-of-line positions and line follower target annotations.
The following script extracts start-of-line positions, line follower targets, and normalized line images.

```
python preprocessing/prep_train_a.py data/Train-A/page data/Train-A data/train_a data/train_a_training_set.json data/train_a_validation_set.json  
```

#### Prepare Train-B

This extracts only the GT lines from the XML.

```
python preprocessing/prep_train_b.py data/Train-B data/Train-B data/train_b data/train_b_training_set.json data/train_b_validation_set.json
```

#### Prepare Test data

Currently we only support running the tests for the Test-B task, not Test-A. When we compute the results for the Test-B while fully exploiting the competition provided regions-of-interest (ROI) we have to do a preprocessing step. This process masks out parts of the image that are not contained in the ROI.

```
python preprocessing/prep_test_b_with_regions.py data/Test-B data/Test-B data/test_b_roi
```

#### Generate Character Settings

This will generate a character set based on the lines in both Train-A and Train-B.
There should 196 unique characters.
This means the network will output 197 characters to include the CTC blank character.

```
python utils/character_set.py data/train_a_training_set.json data/train_a_validation_set.json data/train_b_training_set.json data/train_b_validation_set.json data/char_set.json
```


## Pretraining

Pretraining is performed using 32 pixel tall images. One can change this value and experiment.
During final training the line-level HWR network is retrained afterwards at a high resolution.
The 32 pixel network is recommended by SFR authors which trains faster and is good enough for the alignment.


All three networks can fit on a 12 GB GPU for pretraining. Each network will stop training after 10 epochs without any improvement.


#### Start of Line

You should expect to be done when the validation loss is around 50-60.

```
python sol_pretraining.py sample_config.yaml  
```

#### Line Follower

You should expect to be done when the validation loss is around 40-50.

```
python lf_pretraining.py sample_config.yaml  
```

#### Handwriting Recognition

You should expect to be done when the validation CER is around 0.50 to 0.55.

```
python hw_pretraining.py sample_config.yaml  
```

#### Copy Weights

After pretraining you need to copy the initial weights into the `best_overall`, `best_validation`, and `current` folders.

```
cp -r data/snapshots/init data/snapshots/best_overall
cp -r data/snapshots/init data/snapshots/best_validation
cp -r data/snapshots/init data/snapshots/current
```

## Training

Training of each component and alignment can be performed independently.
I have run using 4 GPUs.
You could do it on a single GPU but you would have to adapt the code to do that.

For BYU's super computer I run the following SLURM files for 4 GPUs.
You can run the python files independent of the SLURM scripts.

#### Initial Alignment

Before you can train, you have to first run the alignment so there are targets for the validation and the training set.
It will perform alignment over the validation set and the first training group (2000 images total)

A sample SLURM file can be found in `slurm_examples/init_alignment.sh`.

```
python continuous_validation.py sample_config.yaml init

```

#### Training

All of the following are designed to be run concurrently on 4 GPUs. They could be modified to run sequentially, but this would slow training time. If you more GPUs `continuous_validation.py` can be set to run over specific subsets of the dataset so more validation can happen in parallel. We did our experiments using 4 GPUs.

A sample SLURM file can be found in `slurm_examples/training.sh`.

```
CUDA_VISIBLE_DEVICES=0 python continuous_validation.py sample_config.yaml
CUDA_VISIBLE_DEVICES=1 python continuous_sol_training.py sample_config.yaml
CUDA_VISIBLE_DEVICES=2 python continuous_lf_training.py sample_config.yaml
CUDA_VISIBLE_DEVICES=3 python continuous_hw_training.py sample_config.yaml
```

## Retraining

Because we trained the handwriting recognition network at a lower resolution, we need to retrain it. First, we need to segment our line-level images at a high resolution.

A sample SLURM file can be found in `slurm_examples/resegment.sh`.

```
python resegment_images.py sample_config_60.yaml

```

After segmenting we need to retrain the network. We can just use the pretraining code to this.

A sample SLURM file can be found in `slurm_examples/retrain_hwr.sh`.

```
python hw_pretraining.py sample_config_60.yaml  
```

## Validation (Competition)

This section covers reproducing the results for the competition data specifically. The next section will explain it more generally.



#### With using the competition regions-of-interest.

```
python run_hwr.py data/test_b_roi sample_config_60.yaml data/test_b_roi_results
```

```
python run_decode.py sample_config_60.yaml data/test_b_roi_results --in_xml_folder data/Test-B --out_xml_folder data/test_b_roi_xml --roi --aug --lm
```

#### Without using the competition regions-of-interest

```
python run_hwr.py data/test_b sample_config_60.yaml data/test_b_results
```

```
python run_decode.py sample_config_60.yaml data/test_b_results --in_xml_folder data/Test-B --out_xml_folder data/test_b_xml --aug --lm
```

#### Submission

The xml folder needs to be compressed to a `.tar` and then can be submitted to the [online evaluation system](https://scriptnet.iit.demokritos.gr/competitions/8/).

We also include the xml files from our system so you can compute the error with regards those predictions instead of submitting to the online system. This will give you a rough idea of how your results compare to other results. The error rate is not computed the same as on the evaluation server.

``
python compare_results.py <path/to/xml1> <path/to/xml2>
``

## Validation (General)

The network can be run on a collection of images as follows. This process produces intermediate results. The post processing to these results are applied in a separate script.

```
python run_hwr.py <path/to/image/directory> sample_config_60.yaml <path/to/output/directory>
```

The postprocessing has number of different parameters. The most simple version is as follows. The `<path/to/output/directory>` is the same path in the previous command. It will save a text file with the transcription and an image to visualize the segmentation.

```
python run_decode.py sample_config_60.yaml <path/to/output/directory>
```

Run it with test-side augmentation:

```
python run_decode.py sample_config_60.yaml <path/to/output/directory> --aug
```

Run it with the language model:

```
python run_decode.py sample_config_60.yaml <path/to/output/directory> --lm
```

Run it with both the language model and test-side augmentation:

```
python run_decode.py sample_config_60.yaml <path/to/output/directory> --aug --lm
```
