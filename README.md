## K-nearest neighbours algorithm for ML-lab

Firstly run `pip install tensorflow` in your virtual environment, since `tensorflow-gpu` is not available for MacOS.

### Results analysis

Limited amount of test samples (only 5k) is used to reduce total calculation time.

L1-distance is used as metric for distance between neighbor samples.

Graph for K-accuracy dependency with k from 1 to k_max=30:
![accuracy-k](https://user-images.githubusercontent.com/20597105/67742504-4f27ee00-fa2d-11e9-9892-46e918d7fc67.png)

Calculation speed:
![progress](https://user-images.githubusercontent.com/20597105/67742549-6c5cbc80-fa2d-11e9-977b-6a3d961be397.png)
