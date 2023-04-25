# Imports all the necesary spacenames
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pathlib import Path
import tensorflow as tf
import conllu
import numpy as np

# Class in charge of manipulating the data
class MyDataHandler_P1:
  # Receives the max length of the sentences and the max char (optional) of word as parameters
  def __init__(self, max_len, char_len=0):
    # Creates a different tokenizer for words, char and tags
    # Use NA as the oov token, clear filters (because we want puntuaction) and lower the words
    self.word_tokenizer = Tokenizer(oov_token='<NA>', filters='', lower=True)
    # Use the tokenizer at char level and filter the puntuation symbols
    self.char_tokenizer = Tokenizer(char_level=True, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    # Use the tokinizer "as is", because it online encode the UPOS
    self.tag_tokenizer = Tokenizer()

    # Save the maximum lengths for later use
    self.max_len = max_len
    self.char_len = char_len

  # Return Tokenized values (and creates the dictionary if not exists)
  def get_Tokenized_Values(self, values, tokenizer):
    # Create the dictionary only if it does not exist
    if len(tokenizer.word_index) <= 0:
      tokenizer.fit_on_texts(values)
    
    # Transform values texts to numbers
    inputs_sequence = tokenizer.texts_to_sequences(values)
    
    return (inputs_sequence, tokenizer.word_index)
    
  # Return character Tokenized values (and creates the dictionary if not exists)
  def get_char_Tokenized_Values(self, values, tokenizer):
    # Separate each word in sentence
    list_sentences=[]
    # A manual manipulation of the dataset is made to generate a object that can be use with characters in the model
    # For each sentence
    for i in range (0,len(values)):
        len_word=len(values[i])
        s=''
        sc=[]
        list_sentences.append(sc)
        # For each word
        for j in range(0,len_word):
            # The first character cant be a space
            if j==0:
                s=s + values[i][j]
            else:
                # If the amount of word is the padding limit, break
                # This truncate sentences with more than 128 words
                if j == self.max_len:
                    break
                else:
                    s=s +' '+ values[i][j]
        sc.append(s)
  
    # Create the dictionary only if it does not exist
    if len(tokenizer.word_index) <= 0:
      # For each sentence
      for i in range(0,len(list_sentences)):
        self.char_tokenizer.fit_on_texts(list_sentences[i])
    
    # Encode the sentences with padding to ids
    ids2=[]
    # Loop in the sentences to convert each one into ids
    for m in range(0,len(list_sentences)):
        ids2_caracters=self.char_tokenizer.texts_to_sequences(list_sentences[m])
        ids2.append(ids2_caracters)
    
    # Generate the final list of sentences converted to characters
    list2_cara=[]
    for i in range (0,len(ids2)):
        len_sen=len(ids2[i][0])
        # List of ids in character of each word
        car2_id=[]
        # List of ids in character of each sentence
        cara_sen=[]
        # Count the number of word
        count_w=1
        # For the length of the sentence
        for j in range(0,len_sen):
            if j == len_sen-1:
              car2_id.append((ids2[i][0][j]))
              cara_sen.append(car2_id)
              # Limit to the max_len (128)
              for m in range(0,self.max_len-count_w):
                  car2_id=[]
                  cara_sen.append(car2_id)
            else:
            # if is not a space we add it to the list
              if ids2[i][0][j] != 1: 
                  car2_id.append(ids2[i][0][j])
              else:
                  count_w+=1
                  cara_sen.append(car2_id)
                  car2_id=[]

        list2_cara.append(cara_sen)
        
    # Return the new list of characters and the dictionary word count
    return (list2_cara, self.char_tokenizer.word_index)

  # Detokenize a value. Recieves the type of value as a parameter.
  def get_Detokenized_Values(self, values, value_type):
    # Depending on the value to detokenize use the correct tokenizer
    if value_type=="Input":
    # Transform number into values texts
      decoded_sequence = self.word_tokenizer.sequences_to_texts(values)
    elif value_type=="Output":
      decoded_sequence = self.tag_tokenizer.sequences_to_texts(values)
    return decoded_sequence

  # Get the values padded and truncated
  def get_Padded_Values(self, values):  
    # Pad the sequence with zeros after the last value and cut it if excedes the max len of the class
    return pad_sequences(values, padding='post', maxlen=self.max_len, value=0, truncating='post')
    
  # Get the values padded and truncated
  def get_char_Padded_Values(self, values):  
    # Pad each words in characters level
    final_pad=[]
    for i in range(0,len(values)):
      # Pad the amount of character (and truncate) for each word
      padded_car = pad_sequences(values[i], padding='post', maxlen=self.char_len, value=0, truncating='post')
      final_pad.append(padded_car)
        
    return final_pad
   
  # Get the values in a one hot encode
  def get_OneHotEncoding_Values(self, values, num_classes):
    hot_values = tf.keras.utils.to_categorical(values, num_classes=num_classes)
    return hot_values
  
  # Get the values in a tensor (needed to second model)
  def get_char_Tensor(self, values):
    train_tensor = tf.convert_to_tensor(values)
    return train_tensor

  # Transform training information into batches
  def get_Batches(self,inputs, outputs, train_batch_size):
    train_ds = tf.data.Dataset.from_tensor_slices((inputs,outputs))
    return train_ds.batch(batch_size=train_batch_size)

  # Transform training information into batches
  def get_char_Batches(self,inputs, inputs_char, outputs, train_batch_size):
    train_ds = tf.data.Dataset.from_tensor_slices(([inputs, inputs_char],outputs))
    return train_ds.batch(batch_size=train_batch_size)
              
  # Transform the data in a format ready to be used by the char model
  def get_char_Procesed_Data(self, train_x, train_y, replaceOOV, showSamples):
    # Return Tokenized values (and creates the dictionary if not exists)
    (inputs, words_index_inputs) = self.get_Tokenized_Values(train_x, self.word_tokenizer)
    (char_inputs, char_index_inputs) = self.get_char_Tokenized_Values(train_x, self.char_tokenizer)
    (outputs, words_index_outputs) = self.get_Tokenized_Values(train_y, self.tag_tokenizer)

    # If true, replaces the OOV values in the output by the other classification
    if replaceOOV:
      # Get the tag correspondig to other from the dictionary
      tag_other = self.tag_tokenizer.word_index['x']

      # Replace all values in train_y that match the index of OOV in train_x
      for x in range(len(train_x)):
        matches = [i for i,x in enumerate(train_x[x]) if x==1]

        for indice in matches:
          train_y[x][indice]= tag_other

    # Get the values padded and truncated
    padded_inputs = self.get_Padded_Values(inputs) 
    padded_char_inputs = self.get_char_Padded_Values(char_inputs) 
    padded_outputs = self.get_Padded_Values(outputs)

    # Get the output values in a one hot encode
    binary_outputs = self.get_OneHotEncoding_Values(padded_outputs, num_classes=len(words_index_outputs)+1)

    # If true, shows a sample of the values to manual validation
    if showSamples:
      print("Values sample: ")
      print("Inputs: " , train_x[0])
      print("Inputs coded: " , inputs[0])
      print("Inputs padding coded: " , padded_inputs[0])
      print("Inputs Sentence count: " , len(train_x))
      print("Dictionary Inputs count: " , len(words_index_inputs))
      print("Char Inputs: " , train_x[0])
      print("Char coded: " , char_inputs[0])
      print("Char inputs padding coded: " , padded_char_inputs[0])
      print("Char dictionary Inputs count: " , len(char_index_inputs))
      print("Outputs: ", train_y[0])
      print("Outputs coded: ", outputs[0])
      print("Outputs padding coded: ", padded_outputs[0])
      print("Outputs padding bynary: ", binary_outputs[0])
      print("Outputs Sentence count: " , len(train_x))
      print("Dictionary Outputs count: " , len(words_index_outputs))
    
    # Get the characters in a tensor
    tensor_char_input = self.get_char_Tensor(padded_char_inputs)

    # Returns the formated information and the lenght of the dictionaries
    return (padded_inputs, len(words_index_inputs) + 1, 
              tensor_char_input, len(char_index_inputs) + 1,
              binary_outputs, len(words_index_outputs) + 1)

  # Format a text to an acceptable value for advanced prediction
  def get_char_Test_Data(self, text_to_predict):
    # Tokenize the sentences and words
    (inputs, words_index_inputs) = self.get_Tokenized_Values(text_to_predict, self.word_tokenizer)
    (char_inputs, char_index_inputs) = self.get_char_Tokenized_Values(text_to_predict, self.char_tokenizer)
    
    # Pad the sentences and words
    padded_inputs = self.get_Padded_Values(inputs) 
    padded_char_inputs = self.get_char_Padded_Values(char_inputs)

    # Transform to tensor the characters
    tensor_char_inputs = self.get_char_Tensor(padded_char_inputs)

    # returns a tuple with the texto to predict and the padded numbers
    return (text_to_predict, padded_inputs, tensor_char_inputs)

  # Detokenize and print an advanced value.
  def print_Detokenized_char_Values(self, values):
    # For each coded value
    for i in range(len(values)):
      print("/////////////////New Sentence//////////////////////")
      for ii in range(len(values[i])):
        # Use argmax to pass to binary the float result and then sequence to text to transform it into a word
        res_val = self.tag_tokenizer.sequences_to_texts([[values[i][ii][1]]])
        
        if (res_val):
          # Print the original and the predicted value
          print(values[i][ii][0], res_val)