# Learning to Detect Violent Videos using Convolutional Long Short-Term Memory

Pytorch version can be found [here](https://github.com/swathikirans/violence-recognition-pytorch)

The source code associated with the paper [Learning to Detect Violent Videos using Convolutional Long Short-Term Memory](https://arxiv.org/abs/1709.06531), published in AVSS-2017. 

The code for evaluating the performance of the model proposed in the paper is given. The convLSTM model is based on the source provided by the authors of the paper [Spatio-temporal video autoencoder with differentiable memory](https://github.com/viorik/ConvLSTM)

#### Prerequisites
* Torch7(http://torch.ch/docs/getting-started.html)
 
#### How to run
The pre-trained models can be downloaded from the following links:
* [Hockey dataset](https://drive.google.com/open?id=0Bwd9CvJBXhj4aDVieGlUVzNaZnc)
* [Movies dataset](https://drive.google.com/open?id=0Bwd9CvJBXhj4SWNibHBUOGJHLW8)
* [Violentflows dataset](https://drive.google.com/open?id=0Bwd9CvJBXhj4NHhlQV9ZZVAxNlk)

To evaluate the performance of the model trained run the following command,
th main-run.lua -FightsDataset path-to-fights-video-file -noFightsDataset path-to-nonfights-video-file -model path-to-model -nSeq number-of-frames

The dataset should be in the format number_of_videosxnumber_of_framesx3xwidthxheight

To cite our paper/code:

```
@article{sudhakaran2017learning,
  title={Learning to Detect Violent Videos using Convolutional Long Short-Term Memory},
  author={Sudhakaran, Swathikiran and Lanz, Oswald},
  journal={arXiv preprint arXiv:1709.06531},
  year={2017}
}
```
 


