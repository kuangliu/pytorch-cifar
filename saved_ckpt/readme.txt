ckpt1: 用shufflenet v2训练的, 20个epoch, 80%, 主要用来试验一下代码之类的, SGD(lr=0.1, momentum=0.9, weight_decay=5e-4) (default)
ckpt2: 用shufflenet v2训练了两百个epoch, 80%, 貌似极限就是这样了, SGD(lr=0.1, momentum=0.9, weight_decay=5e-4) (default)
ckpt3: 用DenseNet训练到三十个epoch左右开始卡住了, 87%, SGD(lr=0.1, momentum=0.9, weight_decay=5e-4) (default)
ckpt4: 用DenseNet训练了……三百个epoch。最后也只能到88%, 而且我觉得是巧合...SGD(lr=0.1, momentum=0.9, weight_decay=5e-4) (default)
