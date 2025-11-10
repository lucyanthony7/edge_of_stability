# edge_of_stability
This project aims to recreate empirical results described in the paper Adaptive Gradient Methods at the Edge of Stability by Cohen et al [1](https://arxiv.org/abs/2207.14484). Edge of Stability (EoS) is a phenomenon observed particularly in Deep Learning where the stability (the largest eigenvalue of the Hessian of the loss function) interacts with the training loss in surprising ways.

The Python script [cifar10_SGD.py](cifar10_SGD.py) runs a Gradient Descent algorithm and estimates the sharpness, while also tracking the loss over a specified numer of training epochs. Thus we are able to produce two plots showing the evolution of the loss function and the sharpness over the different training epochs. We simulate results for different learning rates in order to observe empirically the progressive sharpnening and EoS phenomenon as the sharpness approaches the EoS limit.

We use the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) to produce these results. Our observations are the following:
- If the learning rate is too small, then progressive sharpening does not occur fast enough and the EoS is not obtained.
- As the learning rate increases, the rate of progressive sharpening increases
- Once the EoS limit is reached, the loss function decreases non-monotonically, and the sharpness oscillates around this EoS limit (typically $2/\eta$ where $\eta$ is the learning rate).
