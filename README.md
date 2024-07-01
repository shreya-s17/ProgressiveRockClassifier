# Music Genre Classifier: Progressive Rock vs. Non-Progressive Rock

### Description

PROJECT RUBRIC:

(30 points) Discussion of supervenience question based on David Chalmers’ target chapter and the Stanford Encyclopedia exposition.
1.	(10 points) Give a summary of local and global supervenience and then natural vs. logical supervenience.
2.	(15 points) Discussion of the core issue of logical supervenience of consciousness on the physical.
3.	(5 points) Discuss the relevance of the logical supervenience concept for present day AI.

(30 points) Performance of the classifiers on the Non-Prog and Prog training and validation sets.
1.	(10 points) Discussion of the techniques underpinning your classifiers.
2.	(20 points) Discussion and explanation of the performance of the classifiers on the training set.

(40 points) Performance of the classifiers on the test set.
1.	(15 points) Demo of performance of classifier on test set.
2.	(25 points) Discussion and explanation of the performance of the classifiers on the test set.




### Layout

- `data/`: Dataset creation scripts
- `EncoderDecoder/`: Windowed MFCC features -> convolutional dual-RNN encoder-decoder, based on in-class lecture.  
- `FullyConnected NN/`: Fully connected neural network.

### Dependencies (add as needed)

- tqdm: `pip install tqdm`
- ffmpy: `pip install ffmpy`
- pytorch: https://pytorch.org/get-started/locally/
- librosa: `conda install -c conda-forge librosa`
- pydub: `pip install pydub`
