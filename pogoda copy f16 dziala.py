import os
os.add_dll_directory("D:/CUDA/bin")
os.add_dll_directory("D:/CUDnn/cudnn-windows-x86_64-8.6.0.163_cuda11-archive/bin")
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# Wczytanie danych
data = pd.read_csv(r"D:\Users\User\Desktop\Studia\MGR\SEM 1\IMS\F16Data_SineSw_Level3.csv") #zwraca Data Frame

# Konwersja do typu zmiennoprzecinkowego
x=data[['Force','Voltage','Acceleration1','Acceleration2','Acceleration3']].astype(float)

# Usunięcie zerowych danych, które występują na końcu zbiorów
df = x.loc[(x != 0).any(axis=1)]

column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

BATCH_SIZE = 128
OUT_STEPS = 100
IN_STEPS = 1500
MAX_EPOCHS = 1000
SEQ_STRIDE = 4
LEARNING_RATE_ADAM = 0.001
LSTM_UNITS = 64


checkpoint_path = "F16/model_E{}_S{}-{}_BS{}_SS{}/model".format(MAX_EPOCHS, IN_STEPS, OUT_STEPS, BATCH_SIZE, SEQ_STRIDE)
plot_pred_path = "F16/model_E{}_S{}-{}_BS{}_SS{}/predict".format(MAX_EPOCHS, IN_STEPS, OUT_STEPS, BATCH_SIZE, SEQ_STRIDE)
plot_loss_path = "F16/model_E{}_S{}-{}_BS{}_SS{}/history".format(MAX_EPOCHS, IN_STEPS, OUT_STEPS, BATCH_SIZE, SEQ_STRIDE)
history_path = "F16/model_E{}_S{}-{}_BS{}_SS{}/train_history.pkl".format(MAX_EPOCHS, IN_STEPS, OUT_STEPS, BATCH_SIZE, SEQ_STRIDE)



class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
  
  def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
      labels = tf.stack(
          [labels[:, :, self.column_indices[name]] for name in self.label_columns],
          axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels

# WindowGenerator.split_window = split_window



  def plot(self, model=None, plot_col='Acceleration1', max_subplots=3):
    
    inputs, labels = self.example_test

    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    # print(len(inputs))
    # max_n = min(max_subplots, len(inputs))
    # num_dim = inputs.ndim
    if inputs.ndim < 3:
      inputs = tf.expand_dims(inputs, axis=0)
      labels = tf.expand_dims(labels, axis=0)
      max_n = 1
    else:
      max_n = min(max_subplots, len(inputs))


    if model is not None:
      predictions = model(inputs)
    
    
    for n in range(max_n):
      plt.subplot(max_n, 1, n+1,)
      plt.title("F16_E{}_S{}-{}_BS{}_SS{}".format(MAX_EPOCHS, IN_STEPS, OUT_STEPS, BATCH_SIZE, SEQ_STRIDE))
      plt.ylabel(f'{plot_col} [normed]')
      plt.plot(self.input_indices, inputs[n, :, plot_col_index],
              label='Inputs', marker='.', zorder=-10,)
      
      if self.label_columns:
        label_col_index = self.label_columns_indices.get(plot_col, None)
      else:
        label_col_index = plot_col_index

      if label_col_index is None:
        continue

      plt.scatter(self.label_indices, labels[n, :, label_col_index],
                  edgecolors='k', label='Labels', c='#2ca02c', s=64)
      if model is not None:
        odleglosc = np.linalg.norm(labels[n, :, label_col_index]-predictions[n, :, label_col_index])
        print("odl: ", odleglosc)
        print("odl. śr: ", odleglosc/OUT_STEPS)
        # predictions = model(inputs)
        plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                    marker='X', edgecolors='k', label='Predictions',
                    c='#ff7f0e', s=64)
        # plt.annotate("Odl. euklidesowa: {:.3f}, Śr.: {:.3f}".format(odleglosc, odleglosc/OUT_STEPS), xy = (0,0))
        plt.text(0.05, 0.95, "Odl. euklidesowa: {:.3f}, Śr.: {:.3f}".format(odleglosc, odleglosc/OUT_STEPS), transform=plt.gca().transAxes,
                  verticalalignment='top', horizontalalignment='left')
      if n == 0:
        plt.legend(loc = 'lower left')
      plt.grid()
    plt.xlabel('')
    

    if model is not None:
        plt.savefig(plot_pred_path)
        plt.show()
# WindowGenerator.plot = plot

  def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride = SEQ_STRIDE,#int(self.total_window_size/2),
        shuffle=False,
        batch_size=BATCH_SIZE)

    ds = ds.map(self.split_window)

    return ds

# WindowGenerator.make_dataset = make_dataset

  @property
  def train(self):
    return self.make_dataset(self.train_df)

  @property
  def val(self):
    return self.make_dataset(self.val_df)

  @property
  def test(self):
    return self.make_dataset(self.test_df)

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
      # No example batch was found, so get one from the `.train` dataset
      # print(next(iter(self.train)))
      result = next(iter(self.train))
      # # TO MOJE
      # for batch in self.train:
      #   last_batch = batch
      # result = last_batch
      # # ^^^
      # # And cache it for next time
      self._example = result
    return result

  @property
  def last_train(self):
    for batch in self.train:
        last_batch = batch
    inputs, labels = last_batch
    inputs_last, labels_last = inputs[-1], labels[-1]
    return inputs_last, labels_last

  @property
  def example_test(self):
    result = next(iter(self.test))
    return result


# print(len(train_df))
# print(int(len(train_df)/BATCH_SIZE))
multi_window = WindowGenerator(input_width=IN_STEPS,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS, label_columns = ['Acceleration1'])


# print(multi_window)
# print(multi_window.example)
# print(multi_window.train)
# print(multi_window.val)
# print(multi_window.test)


multi_window.plot()


multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units].
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(units = LSTM_UNITS, return_sequences=False, activation = 'tanh'),
    # tf.keras.layers.Dropout(0.2),
    # Shape => [batch, out_steps*features].
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features].
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])



def compile_and_fit(model, window, patience=MAX_EPOCHS+1):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  # checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path,
  #                                                           save_weights_only = False,
  #                                                           save_best_only = False
  #                                                           )



  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE_ADAM),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])
  # print(window.train)
  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history



# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TO JAK TRZEBA BĘDZIE DOUCZAĆ, to wczytujemy model ze ścieżki
# !!!!! PAMIĘTAĆ O ZMIANIE ŚCIEŻKI W .save()
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
multi_lstm_model = tf.keras.models.load_model(r'D:\Users\User\Desktop\Studia\MGR\SEM 1\IMS\F16\model_E1000_S1500-100_BS128_SS4\model')
# checkpoint_path = ""
plot_pred_path = r'D:\Users\User\Desktop\Studia\MGR\SEM 1\IMS\F16\model_E1000_S1500-100_BS128_SS4\pred2'
# plot_loss_path =



# history = compile_and_fit(multi_lstm_model, multi_window)

# Zapisanie modelu
# multi_lstm_model.save(checkpoint_path)

# Zapisanie pliku historii
# with open(history_path, 'wb') as file:
#   pickle.dump(history.history, file)

# Ewentualne wczytanie historii
# with open(history_path, 'rb') as file:
#     history = pickle.load(file)

multi_window.plot(model = multi_lstm_model, max_subplots=1)



# plt.figure(120)

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(['Train', 'Validation'], loc='upper right')
# plt.grid()
# plt.savefig(plot_loss_path)
# plt.show()

print("Done")

#### DO ZROBIENIA
# OGARNAC JAK WYCIAGNAC PREDYKCJE SPOD FUNKCJI PLOTA
# I JAK ZROBIC ZEBY DZIALALO TEZ NA PREDYKCJE BEZ LABELI, ZEBY SOBIE NIEWIADOME PREDYKOWAC