# zemia-art
this is a reposiitory for zemi A with Adversarial Robustness toolbox.

# TODO
- [x] explaining and harnesting Fig 4の再現実験
- [x] random な 場合のperturbationでもFig 4と同じ図を書いてみる

ランダムノイズは, epsilon * np.random.randn(28,28,1)とした.
理由は, 乱数が0から1の値をとるとしたのは, FGSMにおいてgradientのsignをとる, つまり-1,1という小さな値しか取らないためだ.

- [x] transferabilityが説明がつく理由になっていないのではないか考察する.

hoge_plot.pngとhoge_plot_random_ver.pngを比較するとわかるように, FGSMはターゲットのpredictionがrandom verに比べて傾きが大きいことがわかる. これにより, epsilonが小さい範囲で誤りを最大化させるというミッションにおいてFGSMが有効であることがわかる.

[0_7_plot.png](0_7_plot.png)


[0_7_plot_random_ver.png](0_7_plot_random_ver.png)

## FGSM, random noiseのAccuracyの比較
eps = 0.1 + 0.1 * i (i=0,..,9)としてFGSM及びrandom noiseによるaccuracyの比較を行った.
以下に結果を示す.
```
eps=0.1
Accuracy on adversarial test examples: 80.47%
Accuracy on random noise test exampels: 97.76%
eps=0.2
Accuracy on adversarial test examples: 57.709999999999994%
Accuracy on random noise test exampels: 96.05%
eps=0.30000000000000004
Accuracy on adversarial test examples: 43.82%
Accuracy on random noise test exampels: 87.81%
eps=0.4
Accuracy on adversarial test examples: 36.49%
Accuracy on random noise test exampels: 68.97999999999999%
eps=0.5
Accuracy on adversarial test examples: 32.83%
Accuracy on random noise test exampels: 50.14999999999999%
eps=0.6
Accuracy on adversarial test examples: 30.990000000000002%
Accuracy on random noise test exampels: 36.35%
eps=0.7000000000000001
Accuracy on adversarial test examples: 30.470000000000002%
Accuracy on random noise test exampels: 27.98%
eps=0.8
Accuracy on adversarial test examples: 29.75%
Accuracy on random noise test exampels: 22.8%
eps=0.9
Accuracy on adversarial test examples: 29.360000000000003%
Accuracy on random noise test exampels: 19.54%
eps=1.0
Accuracy on adversarial test examples: 29.160000000000004%
Accuracy on random noise test exampels: 17.28%
```
やはり, epsilonが小さい値において, FGSMの攻撃力の高さが窺える. 逆に言えば, Accuracyの観点ではFGSMは一定の範囲においてのみ特筆している.

# Usage
```
$ git clone https://github.com/speed1313/zemia-art
$ cd zemia-art
$ python3 experiment.py
```


# Reference
- https://github.com/Trusted-AI/adversarial-robustness-toolbox
- Intriguing property of Machine Learning, https://arxiv.org/abs/1312.6199
- Explaining and Harnessing Adversarial Examples, https://arxiv.org/abs/1412.6572?context=cs