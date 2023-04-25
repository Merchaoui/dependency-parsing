# Imports all the necesary spacenames
import tensorflow as tf
import keras
from keras import layers
from keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Embedding
from keras.callbacks import EarlyStopping

# Class in charge of handling the model
class MyModelHandler:
  def __init__(self, max_len, max_len_pre, char_len_pre, word_count, upos_count, action_count, relation_count, tag_count_pre):
    self.max_len = max_len
    self.max_len_pre = max_len_pre
    self.char_len_pre = char_len_pre
    self.word_count = word_count
    self.upos_count = upos_count
    self.action_count = action_count
    self.relation_count = relation_count
    self.tag_count_pre = tag_count_pre

  # Receives all the necessary information to create the model
  def get_model(self, model_type):
    if model_type=="Basic":
      # Input Layer stack
      inputs_stack = keras.Input(shape=(self.max_len,))
      # Input layer buffer
      inputs_buffer = keras.Input(shape=(self.max_len,))

      #embedding layer for stack
      embedding_layer_stack = Embedding(self.word_count,self.max_len,mask_zero=True)(inputs_stack)
      #embedding layer for buffer
      embedding_layer_buffer = Embedding(self.word_count,self.max_len,mask_zero=True)(inputs_buffer)

      conct = layers.Concatenate()([embedding_layer_stack, embedding_layer_buffer])

      x = layers.Conv1D(filters=128, kernel_size=3)(conct)
      x = layers.GlobalMaxPooling1D()(x)

      outputs_1 = layers.Dense(self.action_count, activation=('softmax'))(x)
      outputs_2 = layers.Dense(self.relation_count, activation=('softmax'))(x)

      return keras.Model(inputs=[inputs_stack, inputs_buffer], outputs=[outputs_1, outputs_2])

    elif model_type=="Advanced":
      # Input Layer stack words
      inputs_stack = keras.Input(shape=(self.max_len,))
      # Input layer buffer words
      inputs_buffer = keras.Input(shape=(self.max_len,))
      # Input Layer stack upos
      inputs_stack_pos = keras.Input(shape=(self.max_len,))
      # Input layer buffer upos
      inputs_buffer_pos = keras.Input(shape=(self.max_len,))

      #embedding layer for stack
      embedding_layer_stack = Embedding(self.word_count,self.max_len,mask_zero=True)
      #embedding layer for buffer
      embedding_layer_buffer = Embedding(self.word_count,self.max_len,mask_zero=True)
      #embedding layer for upos stack
      embedding_layer_stack_pos = Embedding(self.upos_count,self.max_len,mask_zero=True)
      #embedding layer for upos buffer
      embedding_layer_buffer_pos = Embedding(self.upos_count,self.max_len,mask_zero=True)

      conv1D = layers.Conv1D(128,3)
      maxpool= layers.GlobalMaxPooling1D()

      dense1=Dense(self.action_count, activation=('softmax'))
      dense2=Dense(self.relation_count, activation=('softmax'))

      x=embedding_layer_stack(inputs_stack)
      y=embedding_layer_buffer(inputs_buffer)
      r=embedding_layer_stack_pos(inputs_stack_pos)
      s=embedding_layer_buffer_pos(inputs_buffer_pos)
      z=layers.Concatenate(axis=1)([x, y,r,s])
      z=conv1D(z)
      z=maxpool(z)
      output1=dense1(z)
      output2=dense2(z)

      return keras.Model(inputs=[inputs_stack,inputs_buffer,inputs_stack_pos,inputs_buffer_pos], outputs=[output1, output2])
    
    # Implement the model using the functional API with a bidirectional layer.
    elif model_type=="Advanced_Pre":
      # Create the input of the functional api model with the length of the dictionary
      inputs = keras.Input(shape=(self.max_len_pre,))
      # Input is the size of the vocabulary, output is the length of the vector of each represented word, Mark zero to skip the zeros (padded values)
      embedding_layer = layers.Embedding(self.word_count,self.max_len_pre,mask_zero=True)

      lstm_layer =layers.Bidirectional(layers.LSTM(self.max_len_pre,return_sequences=True))

      # Time distributed layer with a dense layer inside with a softmax function for multiclassification
      dense = layers.TimeDistributed(layers.Dense(self.tag_count_pre, activation=('softmax')))

      # Create the input of the functional api model with the length of the dictionary and the length of the characters
      char_inputs = keras.Input(shape=(self.max_len_pre,self.char_len_pre))
      # Embedding layer for the characters
      char_embedding_layer = layers.Embedding(self.char_len_pre,self.max_len_pre,mask_zero=True)

      char_level_LSTM = layers.TimeDistributed(layers.Bidirectional(layers.LSTM(self.max_len_pre,return_sequences=False)))

      # Create the final model
      y = char_embedding_layer(char_inputs)
      y = char_level_LSTM(y)
      x = embedding_layer(inputs)

      # Concatenate word and char model into one
      z = layers.Concatenate()([x, y])

      z = lstm_layer(z)
      outputs = dense(z)

      # Create the model
      return keras.Model(inputs=[inputs,char_inputs], outputs=outputs)    
      
  # Trains and compile the model with the specified parameters
  def train(self, loss, optimizer, metrics, epochs, batch_size, train_ds, val_inputs, val_outputs, test_inputs, test_outputs):
    # Compile the model
    self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # Monitor val_loss and if the minimun after three epochs keep incresing, stop the training
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1 , patience=3)

    # Train the model. Save the history for later use.
    self.history = self.model.fit(train_ds,validation_data=(val_inputs, val_outputs), epochs=epochs, batch_size=batch_size, callbacks=[es])

    #Evaluate accuracy against the test data
    loss, accuracy = self.model.evaluate(test_inputs, test_outputs)

   