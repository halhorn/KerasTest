from seq2seq.models import SimpleSeq2Seq
import numpy as np
import matplotlib.pylab as plt

# シンプルな Seq2Seq モデルを構築
model = SimpleSeq2Seq(input_dim=1, hidden_dim=10, output_length=8, output_dim=1)

# 学習の設定
model.compile(loss='mse', optimizer='rmsprop')

# データ作成
# 入力：1000パターンの位相を持つ一次元のサイン波
# 出力：各入力の逆位相のサイン波
a = np.random.random(1000)
x = np.array([np.sin([[p] for p in np.arange(0, 0.8, 0.1)] + aa) for aa in a])
y = -x

# 学習
model.fit(x, y, nb_epoch=5, batch_size=32)

# 未学習のデータでテスト
x_test = np.array([np.sin([[p] for p in np.arange(0, 0.8, 0.1)] + aa) for aa in np.arange(0, 1.0, 0.1)])
y_test = -x_test
print(model.evaluate(x_test, y, batch_size=32))

# 未学習のデータで生成
predicted = model.predict(x_test, batch_size=32)

plt.plot(np.arange(0, 0.8, 0.1), [xx[0] for xx in x_test[9]])
plt.plot(np.arange(0, 0.8, 0.1), [xx[0] for xx in predicted[9]])
plt.show()

