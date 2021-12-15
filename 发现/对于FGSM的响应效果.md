相比于FGSM在MNIST数据集demo的情况，似乎过拟合情况下的对抗攻击会更容易？

![example_fashion_mnist_VGG13_t1=10_R=1](对于FGSM的响应效果\example_fashion_mnist_VGG13_t1=10_R=1.png)

![fashion_mnist_VGG13_t1=10_R=1](对于FGSM的响应效果\fashion_mnist_VGG13_t1=10_R=1.png)

数据集不一样，所以基本数值还没办法比对，但是可以感觉到模式有所不同。

但是我的代码中data加载包含一个标准化，所以还得谨慎确认一下。

![](https://pytorch.org/tutorials/_images/sphx_glr_fgsm_tutorial_001.png)