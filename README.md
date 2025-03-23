# π-PhenoDrug
## What is it?
This is a comprehensive deep learning-based pipeline for phenotypic drug activity screening in high content analysis. <br>

## Requirements 
π-PhenoDrug is a Python scirpt tool, your Python environment need:<br>
Python 3.8 or above <br>
Pytorch 1.7.0 or above <br>
scikit-image 0.21.0 or above <br>


## Usage
The whole workflow is divided into follow steps: <br>
* Use NUSeg for nucleus semantic segmentation. <br>
* Use condition erosion and marker based watershed to identify single cell.<br>
* Extract phenotypic features for each cell using scikit-image. <br>

### Command Line Tool
```Python
#### Train
python train.py --action train_valid --epoch xx --arch xx --encoder xx --batch_size xx --dataset xx --data_path xx
#### python train.py --action train_valid --epoch 1 --arch Unet++ --encoder xception --attention scse --batch_size 2 --dataset BBBC039 --data_path /data01/lixiao/pre/BBBC039
#### Predict
python train.py --action predict --arch xx --encoder xx --attention scse --dataset xx --data_path xx --model_path xx
####python train.py --action predict --arch Unet++ --encoder xception --attention scse --dataset HCS --data_path /data01/lixiao/pre/hcs --model_path /data01/lixiao/pre/saved_model/Unet++_xception_16_BBBC039_100_scse.pth 
#### Watershed and extract phenotypic features
python cal_features.py --data_path xx --out_path xx
```

## Contact
For any questions please contact Dr. Xiao Li (Email: lixiaoBioinfo@163.com).

## License
MIT License

## Citation
Li X, Ouyang Q, Han M, et al. π‐PhenoDrug: A Comprehensive Deep Learning‐Based Pipeline for Phenotypic Drug Screening in High‐Content Analysis[J]. Advanced Intelligent Systems, 2025: 2400635.
