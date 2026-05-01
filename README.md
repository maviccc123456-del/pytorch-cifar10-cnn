# CNN Image Classification with PyTorch

## Language / 言語

- [日本語](#日本語)
- [English](#english)

---

# 日本語

## 概要

本プロジェクトでは、PyTorch を用いて画像分類のためのシンプルな畳み込みニューラルネットワーク CNN を実装しました。

データセットには CIFAR-10 を使用しています。CIFAR-10 は、32×32 ピクセルのカラー画像を 10 クラスに分類するための代表的な画像分類データセットです。

本プロジェクトの目的は、CNN を用いた画像分類の基本的な流れを理解することです。

具体的には、以下の内容を扱います。

- 画像データセットの読み込み
- CNN モデルの構築
- モデルの学習
- 分類精度の評価
- 特徴マップサイズの計算
- ネットワーク構造の可視化

---

## 構造図

以下の図は、本プロジェクトで設計した CNN の構造を示しています。

![CNN Architecture](./image/cnn_architecture.png)

---

## データセット

本プロジェクトでは、`torchvision.datasets` から提供されている CIFAR-10 データセットを使用します。

CIFAR-10 には以下の 10 クラスが含まれています。

| ラベル | クラス |
|---|---|
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

入力画像の形状は以下の通りです。

```text
C × H × W = 3 × 32 × 32
```

---

## ネットワーク構造

本モデルは、LeNet に近いシンプルな CNN 構造です。

```text
入力画像
3 × 32 × 32
        ↓
畳み込み層 Conv2d: 3 → 6, kernel size = 3
ReLU
特徴マップ: 6 × 30 × 30
        ↓
最大値プーリング MaxPool2d: 2 × 2
特徴マップ: 6 × 15 × 15
        ↓
畳み込み層 Conv2d: 6 → 16, kernel size = 3
ReLU
特徴マップ: 16 × 13 × 13
        ↓
最大値プーリング MaxPool2d: 2 × 2
特徴マップ: 16 × 6 × 6
        ↓
平坦化 Flatten
576
        ↓
全結合層
576 → 120
        ↓
全結合層
120 → 84
        ↓
出力層
84 → 10
```

最終出力は 10 次元であり、CIFAR-10 の 10 クラスに対応しています。

---

## モデル構成

本モデルは以下の層で構成されています。

- 2 つの畳み込み層
- 2 つの最大値プーリング層
- 3 つの全結合層
- ReLU 活性化関数
- 多クラス分類用の CrossEntropyLoss

主なモデル構造は以下の通りです。

```python
Conv2d(3, 6, kernel_size=3)
MaxPool2d(2, 2)

Conv2d(6, 16, kernel_size=3)
MaxPool2d(2, 2)

Linear(576, 120)
Linear(120, 84)
Linear(84, 10)
```

---

## 特徴マップサイズの変化

各層における特徴マップサイズの変化は以下の通りです。

| 層 | 出力サイズ |
|---|---|
| 入力 | 3 × 32 × 32 |
| Conv1 | 6 × 30 × 30 |
| Pool1 | 6 × 15 × 15 |
| Conv2 | 16 × 13 × 13 |
| Pool2 | 16 × 6 × 6 |
| Flatten | 576 |
| FC1 | 120 |
| FC2 | 84 |
| Output | 10 |

平坦化後の特徴量数は以下のように計算されます。

```text
16 × 6 × 6 = 576
```

---

## 学習設定

| 項目 | 値 |
|---|---|
| データセット | CIFAR-10 |
| バッチサイズ | 8 |
| エポック数 | 10 |
| 最適化手法 | Adam |
| 学習率 | 0.001 |
| 損失関数 | CrossEntropyLoss |

---

## プロジェクト構成

```text
.
├── main.py
├── image
│   └── cnn_architecture.png
├── model
│   └── image_model.pth
└── README.md
```

---

## 実行環境

本プロジェクトでは、以下のライブラリを使用します。

```bash
pip install torch torchvision matplotlib torchsummary
```

---

## 実行方法

### 1. 必要なライブラリのインストール

```bash
pip install torch torchvision matplotlib torchsummary
```

### 2. モデル保存用フォルダの作成

学習済みモデルは `model` フォルダに保存されるため、事前にフォルダを作成します。

```bash
mkdir model
```

### 3. モデルの学習

`main` 関数内の以下の行を有効にします。

```python
train(train_dataset)
```

その後、以下のコマンドを実行します。

```bash
python main.py
```

学習後、モデルのパラメータは以下のパスに保存されます。

```text
./model/image_model.pth
```

### 4. モデルの評価

学習後、以下の関数を実行することでテストデータに対する評価を行います。

```python
evaluate(test_dataset)
```

実行すると、分類精度が表示されます。

```text
ACC: xx.xx
```

---

## 注意点

現在のコードでは、学習用の関数 `train(train_dataset)` がコメントアウトされており、評価用の `evaluate(test_dataset)` のみが実行される場合があります。

その場合、`./model/image_model.pth` が存在しないとエラーになります。

そのため、初回実行時には必ず先にモデルを学習してください。

---

## 学習したこと

本プロジェクトを通して、CNN を用いた画像分類の基本的な流れを学習しました。

具体的には、以下の内容を学習しました。

- PyTorch による画像データセットの読み込み
- 畳み込み層による画像特徴の抽出
- プーリング層による特徴マップサイズの削減
- 特徴マップサイズの計算方法
- 全結合層に入力するための平坦化処理
- CNN モデルの学習と評価
- 多クラス分類における CrossEntropyLoss の使い方

---

## 今後の改善案

今後の改善案として、以下が考えられます。

- 畳み込み層の追加
- データ拡張の導入
- Batch Normalization の追加
- Dropout による過学習対策
- VGG や ResNet などの深いモデルとの比較
- 学習時の loss と accuracy の可視化

---

# English

## Overview

This project implements a simple Convolutional Neural Network CNN for image classification using PyTorch.

The model is trained and evaluated on the CIFAR-10 dataset. CIFAR-10 is a standard image classification dataset consisting of 32×32 color images from 10 different classes.

The main purpose of this project is to understand the basic workflow of CNN-based image classification, including dataset loading, model construction, training, evaluation, and feature map size calculation.

---

## Architecture Figure

The following figure shows the CNN architecture designed in this project.

![CNN Architecture](./image/cnn_architecture.png)

---

## Dataset

This project uses the CIFAR-10 dataset provided by `torchvision.datasets`.

CIFAR-10 contains 10 classes:

| Label | Class |
|---|---|
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

The input image shape is:

```text
C × H × W = 3 × 32 × 32
```

---

## Network Architecture

The model follows a simple LeNet-style CNN structure.

```text
Input Image
3 × 32 × 32
        ↓
Conv2d: 3 → 6, kernel size = 3
ReLU
Feature Map: 6 × 30 × 30
        ↓
MaxPool2d: 2 × 2
Feature Map: 6 × 15 × 15
        ↓
Conv2d: 6 → 16, kernel size = 3
ReLU
Feature Map: 16 × 13 × 13
        ↓
MaxPool2d: 2 × 2
Feature Map: 16 × 6 × 6
        ↓
Flatten
576
        ↓
Fully Connected Layer
576 → 120
        ↓
Fully Connected Layer
120 → 84
        ↓
Output Layer
84 → 10
```

The final output has 10 dimensions, corresponding to the 10 classes of CIFAR-10.

---

## Model Details

The model consists of:

- Two convolutional layers
- Two max pooling layers
- Three fully connected layers
- ReLU activation functions
- Cross entropy loss for multi-class classification

The main model structure is:

```python
Conv2d(3, 6, kernel_size=3)
MaxPool2d(2, 2)

Conv2d(6, 16, kernel_size=3)
MaxPool2d(2, 2)

Linear(576, 120)
Linear(120, 84)
Linear(84, 10)
```

---

## Shape Transformation

The feature map size changes as follows:

| Layer | Output Shape |
|---|---|
| Input | 3 × 32 × 32 |
| Conv1 | 6 × 30 × 30 |
| Pool1 | 6 × 15 × 15 |
| Conv2 | 16 × 13 × 13 |
| Pool2 | 16 × 6 × 6 |
| Flatten | 576 |
| FC1 | 120 |
| FC2 | 84 |
| Output | 10 |

The flattened feature size is calculated as:

```text
16 × 6 × 6 = 576
```

---

## Training Settings

| Item | Value |
|---|---|
| Dataset | CIFAR-10 |
| Batch size | 8 |
| Epochs | 10 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | CrossEntropyLoss |

---

## Project Structure

```text
.
├── main.py
├── image
│   └── cnn_architecture.png
├── model
│   └── image_model.pth
└── README.md
```

---

## Environment

This project uses the following libraries:

```bash
pip install torch torchvision matplotlib torchsummary
```

---

## How to Run

### 1. Install dependencies

```bash
pip install torch torchvision matplotlib torchsummary
```

### 2. Prepare the model directory

Since the trained model is saved in the `model` directory, create the directory before training.

```bash
mkdir model
```

### 3. Train the model

In the `main` function, uncomment the following line:

```python
train(train_dataset)
```

Then run:

```bash
python main.py
```

After training, the model parameters will be saved as:

```text
./model/image_model.pth
```

### 4. Evaluate the model

After training, run the evaluation function:

```python
evaluate(test_dataset)
```

The program will output the classification accuracy:

```text
ACC: xx.xx
```

---

## Important Note

In the current code, the training function may be commented out and only the evaluation function may be executed.

If `./model/image_model.pth` does not exist, running only the evaluation function will cause an error.

Therefore, please train the model first before evaluation.

---

## What I Learned

Through this project, I learned the basic process of image classification using CNNs.

Specifically, I learned:

- How to load image datasets using PyTorch
- How convolutional layers extract image features
- How pooling layers reduce feature map size
- How to calculate the shape of feature maps
- How to flatten feature maps before fully connected layers
- How to train and evaluate a CNN model
- How to use CrossEntropyLoss for multi-class classification

---

## Future Improvements

Possible future improvements include:

- Adding more convolutional layers
- Using data augmentation
- Adding batch normalization
- Adding dropout to reduce overfitting
- Comparing this model with deeper models such as VGG or ResNet
- Visualizing training loss and accuracy curves

---
