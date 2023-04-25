# Imports all the necesary spacenames
# import module sys to get the type of exception
import sys
import numpy as np

from conllu import SentenceList

# Class in charge of manipulating the oracle data
class My_Oracle_Test_Handler:
  def __init__(self, sentence, model_p1=None, dh_p1=None):
    # Creates a new class with each values
    self.units = []
    self.sentence = sentence

    # Predict upos with P1 model
    if model_p1 is not None:
      forms=[]
      upos=[]
      for i in range(len(sentence)):
        forms.append(sentence[i]['form'])
        upos.append(sentence[i]['upos'])

      (text_to_predict, padded_inputs, tensor_char_inputs) = dh_p1.get_char_Test_Data(forms)
      predicted_outputs = model_p1.predict([padded_inputs, tensor_char_inputs])

      res = []
      for x in predicted_outputs:
        # Use argmax to pass to binary the float result
        res.append(np.argmax(x))

      new_upos = dh_p1.get_Detokenized_Values([res], "Output")[0].split(' ')

      for i in range(len(sentence)):
        sentence[i]['upos']= new_upos[i].upper()
    
    for i in range(len(sentence)):
      self.units.append(Unit(self.sentence[i]))
    self.stack=[]
    self.cbuffer=self.units.copy()
    
    # Add a special ROOT item as ID 0
    self.stack.insert(0,Unit(None))
    
  def get_Stack(self):
    return self.stack

  # σ|i - rightmost element from the stack
  def get_i(self):
    return self.stack[len(self.stack) - 1]

  def get_Buffer(self):
    return self.cbuffer

  # j|β - leftmost element from the buffer
  def get_j(self):
    if len(self.cbuffer) > 0:
      return self.cbuffer[0]

  def get_State(self):
    stack_names=[]
    stack_upos=[]
    buffer_names=[]
    buffer_upos=[]
    for i in range(len(self.stack)):
      stack_names.append(self.stack[i].get_form())
      stack_upos.append(self.stack[i].get_upos())
    for i in range(len(self.cbuffer)):
      buffer_names.append(self.cbuffer[i].get_form())
      buffer_upos.append(self.cbuffer[i].get_upos())

    return (stack_names, buffer_names, stack_upos, buffer_upos)

  # Perform the required action on the buffer and stack
  def make_action(self, action):
    if action=="SHIFT" or action=="RA":
      self.stack.append(self.cbuffer[0])
      self.cbuffer.pop(0)
    elif action=="REDUCE" or action=="LA":
      self.stack.remove(self.stack[len(self.stack) - 1])

  def get_sentence(self):
    return self.sentence

# Class that represent an unit of the oracle process
class Unit:
  def __init__(self, token):
    self.has_head = False
    self.head_id = -1
    if token is not None:
      self.token = token
      # Items to use
      self.word_id = token['id']
      self.form = token['form']
      self.upos = token['upos']
      # Items to complete
      self.token['head'] = -1
      self.token['deprel'] = ''
    else:
      self.word_id = 0
      self.form = "root"
      self.upos = "X"

  # Get word id
  def get_word_id(self):
    return self.word_id

  # Get word
  def get_form(self):
    return self.form

  def get_upos(self):
    return self.upos

  def get_has_head(self):
    return self.has_head

  def get_deprel(self):
    return self.token['deprel']

  def set_deprel(self, deprel):
    self.token['deprel'] = deprel.lower()

  # Set if the unit have head or not
  def set_has_head(self, value, head_id):
    self.has_head = value
    self.head_id = head_id
    self.token['head'] = head_id

  def get_Token(self):
    return self.token