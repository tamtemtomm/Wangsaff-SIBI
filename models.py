# import tensorflow as tf
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T

# @title <p> Model Architecture
class SIBIConv(nn.Module):
  def __init__(self, in_ch, out_ch, kernel_size=5, pooling_size=2, dilation=1):
    super().__init__()
    padding = (kernel_size - 1)*dilation
    self.conv = nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size, padding=2, stride=1, dilation=dilation),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(),
        nn.Conv1d(out_ch, out_ch, kernel_size, padding=2, stride=1, dilation=dilation),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(),
        nn.MaxPool1d(pooling_size)
    )

  def forward(self, x):
    return self.conv(x)

class SIBILinear(nn.Module):
  def __init__(self, in_ch, out_ch, dropout=0.2):
    super().__init__()
    self.linear = nn.Sequential(
        nn.Linear(in_ch, out_ch),
        nn.LayerNorm(out_ch),
        nn.ReLU(),
        nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.linear(x)

class SIBIModelTorch(nn.Module):

  def __init__(self, num_class, dropout=0.2):
    super().__init__()
    self.conv = nn.Sequential(
        SIBIConv(1, 32),
        SIBIConv(32, 64),
        SIBIConv(64, 128),
        SIBIConv(128, 256),
    )

    self.dropout = nn.Dropout(dropout)
    self.flatten = nn.Flatten()
    self.linear = nn.Sequential(
        SIBILinear(768, 512),
        nn.Linear(512, num_class),
        nn.Softmax(dim=1)
    )

  def forward(self, x):
    x = self.conv(x)
    x = self.dropout(x)
    x = self.flatten(x)
    x = self.linear(x)
    return x

# SIBIModelKeras = tf.keras.models.Sequential([
#     tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding="causal", activation="relu", input_shape=(63, 1)),
#     tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding="causal", activation="relu"),
#     tf.keras.layers.MaxPooling1D(pool_size=2),
#     tf.keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding="causal", activation="relu"),
#     tf.keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding="causal", activation="relu"),
#     tf.keras.layers.MaxPooling1D(pool_size=2),
#     tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding="causal", activation="relu"),
#     tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding="causal", activation="relu"),
#     tf.keras.layers.MaxPooling1D(pool_size=2),
#     tf.keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, padding="causal", activation="relu"),
#     tf.keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, padding="causal", activation="relu"),
#     tf.keras.layers.MaxPooling1D(pool_size=2),
#     tf.keras.layers.Dropout(rate=0.2),
#     # Flatten the results to feed into a DNN
#     tf.keras.layers.Flatten(),
#     # 512 neuron hidden layer
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(26, activation='softmax')])

