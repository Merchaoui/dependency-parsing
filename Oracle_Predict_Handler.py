# Imports all the necesary spacenames
# import module sys to get the type of exception
import sys
import numpy as np

# Class in charge of manipulating the oracle data
class My_Oracle_Predict_Handler:
  def __init__(self, sentence, model_p1, dh_p1):
    # Creates a new class with each values
    self.units = []

    text_to_predict = sentence.replace('.',' .').replace(',',' ,').replace('?',' ?').replace(':',' :')
    text_to_predict = text_to_predict.split()

    self.sentence = text_to_predict

    (_, padded_inputs, tensor_char_inputs) = dh_p1.get_char_Test_Data([text_to_predict])

    predicted_outputs = model_p1.predict([padded_inputs, tensor_char_inputs])

    res = []
    for x in range(len(text_to_predict)):
      # Use argmax to pass to binary the float result
      res.append(np.argmax(predicted_outputs[0][x]))

    new_upos = dh_p1.get_Detokenized_Values([res], "Output")[0].split(' ')

    for i in range(len(text_to_predict)):
      self.units.append(Unit(i+1,text_to_predict[i], new_upos[i]))
    self.stack=[]
    self.cbuffer=self.units.copy()
    
    # Add a special ROOT item as ID 0
    self.stack.insert(0,Unit(0,'Root','X'))
    
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

  def get_units(self):
    return self.units

# Class that represent an unit of the oracle process
class Unit:
  def __init__(self, word_id, form, upos):
    self.has_head = False
    
    # Items to use
    self.word_id = word_id
    self.form = form
    self.upos = upos
    # Items to complete
    self.head_id = -1
    self.deprel = ''
    
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

  def get_head(self):
    return self.head_id

  def get_deprel(self):
    return self.deprel

  def set_deprel(self, deprel):
    self.deprel = deprel.lower()

  def set_head(self, head_id):
    self.has_head = True
    self.head_id = head_id