# Conv-Transformer using slding window attention

* This project provides a Conv-Transformer model with sliding window attention written in 
PyTorch and cuda extension.
* Achieves an accuracy of about 81.7% on the validation set of ImageNet.
* The forward pass of the model is about 16.7ms on a 1660 graphics card with a 2*3*224*224 input,
whilt the Swin-Transformer is about 18.2ms.
* All the accuracy and speed are based on the tiny architecture of the model -- [2, 2, 6, 2] with 
window size of 7.