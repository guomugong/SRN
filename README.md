# SRN
Enhancing Cross-Domain Generalization in Retinal Image Segmentation via Style Randomization and Style Normalization
Please read my [paper](https://doi.org/) for more details!
### Introduction:
Retinal image segmentation is a crucial procedure for automatically diagnosing ophthalmic diseases. However, existing deep learning-based segmentation models suffer from the domain shift issue, i.e., the segmentation accuracy decreases significantly when the test and training images are sampled from different distributions. To overcome this issue, we focus on the challenging single-source domain generalization scenario, where we expect to train a well-generalized segmentation model on unseen test domains with only access to one domain during training. In this paper, we present a style randomization method, which performs random scaling transformation to the LAB components of the training image, to enrich the style diversity. Furthermore, we present a style normalization method to effectively normalize style information while preserving content by channel-wise feature standardization and dynamic feature affine transformation. Our approach is evaluated on four types of retinal image segmentation tasks, including retinal vessel, optic cup, optic disc, and hard exudate. Experimental results demonstrate that our method achieves competitive or superior performance compared to state-of-the-art approaches. Specifically, it outperforms the second-best method by 3.9%, 2.6%, and 4.8% on vessel, optic cup, and hard exudate segmentation tasks, respectively.

## Training Pipeline
To start training, you need to modify the dataset path in `train.py` to point to your local data directory.  
You can launch training using the following command:

```bash
python3 train.py <source_dataset> <random_seed>
```

For example, to train on the DRIVE dataset with random seed 1:

```bash
python3 train.py drive 1
```

After training, the model checkpoints will be saved in the `snapshot/` directory.

### 📦 Pretrained Models

We also provide several pretrained checkpoints for direct evaluation.  
You can download them from [Google Drive](https://xxxx).

## 🔍 Testing

Once the training is complete, you can test the model on a different dataset using the following command:

```bash
python3 predict.py <source_dataset> <target_dataset> <random_seed>
```

For example, to test a model trained on DRIVE and evaluated on STARE:

```bash
python3 predict.py drive stare 1
```

## 📁 Project Structure

```
.
├── train.py            # Training script
├── predict.py          # Testing script
├── snapshot/           # Saved model checkpoints
├── data/               # Dataset folder (path needs to be set manually)
└── ...
```
