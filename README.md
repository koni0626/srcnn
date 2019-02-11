# srcnn
srcnnの実験

# 環境
ubuntu18.04

keras+tensorflowを使用。


# 説明
## 学習
カレントディレクトリをsrcディレクトリに移動します。

cd ~srcnn/src


移動した後、

python3 train.py

を実行すれば学習を行います。

学習済みファイルはweightディレクトリに出力されます。

## 予測
python3 test.pyで予測します。

test.py実行前に39行目のweightのファイル名を学習したファイルに変更する必要があります。

まだ汎化性能などは詳しく調べていません。
