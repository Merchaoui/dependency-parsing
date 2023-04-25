# Imports all the necesary spacenames
# import module sys to get the type of exception
import sys

# Class in charge of manipulating the oracle data
class My_Oracle_Simulator_Handler:
  def __init__(self, ids, forms, heads, deprels, upos):
    # Creates a new class with each values
    self.units = []
    for i in range(len(ids)):
      self.units.append(Unit(ids[i],forms[i],heads[i],deprels[i],upos[i]))
    self.stack=[]
    self.cbuffer=self.units.copy()
    # Add a special ROOT item as ID 0
    self.stack.insert(0,Unit(0,'root',0,'root','X'))
    
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

  def get_State(self, action, relation):
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

    return (stack_names, buffer_names, action, relation, stack_upos, buffer_upos)

  # Get the arcs for the sentence
  def get_arcs(self,ids, heads):
    arcs = []
  
    # For each head, generate a tuple with its corresponding id
    for i in range(len(heads)):
      arcs.append((heads[i], ids[i]))

    return arcs

  # Validate if the id has pending arcs
  def validate_pending_arcs(self,uid, arcs):
    for arc in arcs:
      if arc[0]==uid:
        return True
    
    return False

  #Determines if a dependency tree has crossing arcs or not.
  #Parameters:
  #arcs (list): A list of tuples of the form (headid, dependentid, coding the arcs of the sentence, e.g, [(0,3), (1,4), ...]
  #Returns:
  #A boolean: True if the tree is projective, False otherwise
  def is_projective(self, arcs: list):
    for (i,j) in arcs:
      for (k,l) in arcs:
        if (i,j) != (k,l) and min(i,j) < min(k,l) < max(i,j) < max(k,l):
          return False

    return True

  # Perform the required action on the buffer and stack
  def make_action(self, action):
    if action=="SHIFT" or action=="RA":
      self.stack.append(self.cbuffer[0])
      self.cbuffer.pop(0)
    elif action=="REDUCE" or action=="LA":
      self.stack.remove(self.stack[len(self.stack) - 1])

# Class that represent an unit of the oracle process
class Unit:
  def __init__(self, uid, form, head, deprel, upos):
    self.uid = uid
    self.form = form
    self.head = head
    self.deprel = deprel
    self.upos = upos
    self.has_head = False

  # Get id
  def get_uid(self):
    return self.uid

  # Get word
  def get_form(self):
    return self.form

  # Get parent
  def get_head(self):
    return self.head

  # Get relation
  def get_deprel(self):
    return self.deprel

  def get_upos(self):
    return self.upos

  def get_has_head(self):
    return self.has_head

  # Set if the unit have head or not
  def set_has_head(self, value):
    self.has_head = value