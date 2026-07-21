## Please cite our paper
@article{perez2022,
  title		= {Skeleton-based Relational Reasoning for Group Activity Analysis},
  author	= {Perez, Mauricio and Liu, Jun and Kot, Alex C},
  journal	= {Elsevier Pattern Recognition},
  volume	= {122},
  year		= {2022}
  doi		= {https://doi.org/10.1016/j.patcog.2021.108360},
  url		= {https://www.sciencedirect.com/science/article/pii/S0031320321005409},
}

## Original Dataset
You can download the original dataset from the following page:
https://github.com/mostafa-saad/deep-activity-rec#dataset
@inproceedings{msibrahiCVPR16deepactivity,
  author    = {Mostafa S. Ibrahim and Srikanth Muralidharan and Zhiwei Deng and Arash Vahdat and Greg Mori},
  title     = {A Hierarchical Deep Temporal Model for Group Activity Recognition.},
  booktitle = {2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2016}
}

## Annotated data
- We annotated the ball location in all 4830 sequences, from all 41 images available per sequence.
- Inside each video directory, a text file corresponds to the annotated sequence
	- e.g. volleyball_ball_annotation/39/29885.txt
- Each file has 41 annotation lines, one for each frame
- Each annotation line contains the center coordinates of the ball at that frame: {x} {y}
- If both values are 0, it means the ball is out of sight
- At some cases the ball is occluded, but its location can be reasonably estimated from previous/next frames, thus its location is provided
- The same holds true at some cases where the ball motion is so fast that it disappears for a few frames, before appearing again
- At some sequences there is no ball visible, so the file contains only zeros
- At some sequences there is more than a single ball visible, we annotated only the volleyball being used in the play
