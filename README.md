This is the code implementation of the paper "Deep-generative-adversarial-reinforcement-learning-for-semi-supervised-segmentation-in-medical-image
", please run main.py for training and testing. This is an implementation using 30% additional data (50% of which is unlabeled data). If you want to adjust the proportion of training data or unlabeled data, please adjust train_rate and label_rate in main.py. If you wish to use a custom dataset, please replace the current dataset and dataloader in main.py and create corresponding folders to store the results.
For example, 03_05 means using 30% additional data (50% of which is label data).

Please download the pancreas data set at https://academictorrents.com/details/80ecfefcabede760cdbdf63e38986501f7becd49. Please put the data and labels in the image and label of the pancreas folder and ensure that the suffixes correspond.
