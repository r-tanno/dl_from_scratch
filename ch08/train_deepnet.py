# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from deep_convnet import DeepConvNet
from common.trainer import Trainer

opt = sys.argv[1]
print(opt)
if opt is None:
    opt = 'Adam'

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()  
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer=opt, optimizer_param={},
                  evaluate_sample_num_per_epoch=1000, verbose2=True)
trainer.train()

# パラメータの保存
network.save_params(opt + "_deep_convnet_params.pkl")
print("Saved Network Parameters!")
