## Motion Prediction Using Keyposes
This is the code for the 3DV 2022 paper _Long Term Motion Prediction Using Keyposes._ 

If you find our work useful, please cite it as:

    @inproceedings{kiciroglu2022keyposes,
      author = {Kiciroglu, Sena and Wang, Wei and Salzmann, Mathieu and Fua, Pascal},
      booktitle = {3DV},
      title = {Long Term Motion Prediction Using Keyposes},
      year = {2022}
    }

### Dependencies

We have the current setup:

* cuda 9.0
* Python 3.6
* [Pytorch](https://github.com/pytorch/pytorch) 0.3.1.
* [progress 1.5](https://pypi.org/project/progress/)
* [pygifsicle 1.0.7] (https://pypi.org/project/pygifsicle/)

### Get the data
[Human3.6m](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).

[CMU-Mocap](http://mocap.cs.cmu.edu/) was obtained from the [repo](https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics) of ConvSeq2Seq paper.

#### Keypose Extraction:
If you want to skip this step, you can download the already extracted keyposes from [this link]()

We first run the keypose extraction (of both the training data and test data.)
Note: Since we run the keypose extraction code for all 64 test sequences of each action of subject 5, this takes a while (about 30 minutes). 
```bash
python main_keypose.py --dataset="h36m" --kp_threshold=500 --kp_suffix="3dv2022" --reevaluate_keyposes=True --load_clusters=False --cluster_n=1000
```

### Evaluation commands:

For the "greedy" single future prediction:
```bash
python main_3d.py --is_eval=True --is_diverse=False --cluster_n=1000 --kp_threshold=500 --kp_suffix="3dv2022" --data_dir [Path To Your H36M data]/h3.6m/dataset/ --kp_model_path= [Path to your trained model]
```

You can set the kp_model_path to "pretrained/kp_model" if you'd like to run the code using our pretrained model.

To predict 100 futures:
```bash
python main_3d.py --is_eval=True --is_diverse=True --diverse_seq_num=100 --cluster_n=1000 --kp_threshold=500 --kp_suffix="3dv2022" --data_dir [Path To Your H36M data]/h3.6m/dataset/ --kp_model_path= [Path to your trained model]
```
To use the interpolator network instead of linear interpolation, you can use
```bash
python main_3d.py --is_eval=True --is_diverse=False --cluster_n=1000 --kp_threshold=500 --kp_suffix="3dv2022" --data_dir [Path To Your H36M data]/h3.6m/dataset/ --kp_model_path= [Path to your trained model] --use_interpolator=True --interpolator_num_stage=10 --interpolator_hidden_nodes=512
```

### Training commands:
All the running args are defined in [opt.py](utils/opt.py). We use following commands to train on Human3.6m.

```bash
python main_3d.py --is_eval=False --epoch 100 --cluster_n=1000 --kp_threshold=500 --kp_suffix="3dv2022" --data_dir [Path To Your H36M data]/h3.6m/dataset/
```

### Acknowledgements
This code is adapted from https://github.com/wei-mao-2019/LearnTrajDep, the code for the paper [_Learning Trajectory Dependencies for Human Motion Prediction_](https://arxiv.org/abs/1908.05436), in ICCV 2019. The dataloader is adapted from https://github.com/wei-mao-2019/HisRepItself, the code for the paper [_History Repeats Itself: Human Motion Prediction via Motion Attention_](https://arxiv.org/abs/2007.11755), in ECCV 2020. The action recognition model is adapted from https://github.com/huguyuehuhu/HCN-pytorch, the code for the paper [_Co-occurrence Feature Learning from Skeleton Data for Action Recognition and Detection with Hierarchical Aggregation_](http://arxiv.org/pdf/1804.06055.pdf), in IJCAI 2018.
