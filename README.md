# High-content-screening-pipeline
## What is it?
This is an unsupervised activity assessment tool based on cell phenotypes and established an analysis pipeline for single-cell morphological profile data.<br>

## Requirements 
MoGCN is a Python scirpt tool, your Python environment need:<br>
Python 3.8 or above <br>
Pytorch 1.7.0 or above <br>
scikit-image 0.21.0 or above <br>


## Usage
The whole workflow is divided into follow steps: <br>
* Use NUSeg for nucleus semantic segmentation. <br>
* Use condition erosion and marker based watershed to identify single cell.<br>
* Extract phenotypic features for each cell using scikit-image. <br>
The sample data is in the data folder, which contains the CNV, mRNA and RPPA data of BRCA. <br>
### Command Line Tool
```Python
#### Train
python train.py --action train_valid --epoch xx --arch xx --encoder xx --batch_size xx --dataset xx --data_path xx
#### Predict
python train.py --action predict --arch xx --encoder xx --attention scse --dataset xx --data_path xx --model_path xx
#### Watershed and extract phenotypic features
python cal_features.py --data_path xx --out_path xx
```

## License
MIT License