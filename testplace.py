import os

os.add_dll_directory("D:/CUDA/bin")
os.add_dll_directory("D:/CUDnn/cudnn-windows-x86_64-8.6.0.163_cuda11-archive/bin")

import tensorflow as tf


print(tf.__version__)

print(tf.config.list_physical_devices('GPU'))

# Wczytywanie modelu wraz z historią
loaded_model = tf.keras.models.load_model('F16/model_E1_S600-50_BS128_SS3')

# Wczytywanie historii uczenia
loaded_history = loaded_model.history

