# matsumura-picking
松村さん作成のばら積みピッキング手法
CNNによる学習が用いられており、pytorchで動くように手直ししたもの

inputフォルダにout.plyとno-obj.plyの２つを入れる

$ python3 ESTIMATION_pytorch.py 0

or

$ python3 ESTIMATION_pytorch.py 1

0ならGraspabilityのみの把持位置検出
1ならCNNを用いた絡み位置の予測を加えた把持位置検出

なお、学習の仕方は不明
