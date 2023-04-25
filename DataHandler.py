# Imports all the necesary spacenames
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pathlib import Path
import tensorflow as tf
import conllu
import numpy as np
from Oracle_Simulator_Handler import My_Oracle_Simulator_Handler
from Oracle_Test_Handler import My_Oracle_Test_Handler
from Oracle_Predict_Handler import My_Oracle_Predict_Handler

# Class in charge of manipulating the data
class MyDataHandler:
  def __init__(self, max_len):
    # Prepare dictionaries for columns FORM, UPOS and DEPREL 
    # To get ready to convert from text/label to a numeric value and viceversa
    # Use NA as the oov token, clear filters (because we want puntuaction) and lower the words
    self.form_tokenizer = Tokenizer(oov_token='<NA>', filters='', lower=True)
    self.upos_tokenizer = Tokenizer()
    self.deprel_tokenizer = Tokenizer()
    self.action_tokenizer = Tokenizer()

    self.max_len = max_len

  # Reads the file in universal dependecies format and returns a list with inputs
  def get_UD_Values(self, path, rewrite):
    ids=[]
    forms=[]
    heads=[]
    #Universal dependency relation to the HEAD or a defined language-specific subtype of one.
    deprels=[]
    upos=[]
    
    # Open en readmode and read the data
    with open(path, mode="r", encoding="utf-8") as data:
      text = data.read()

    # Parse the text to get each sentence in a new row
    sentences = conllu.parse(text)
    # Create a empty list of sentences to use later to save without multi words
    new_sentences = conllu.models.SentenceList()
    
    # Loop for every sentence to create the dataset
    for sentence in sentences:
      new_sentence_ids=[]
      new_sentence_forms=[]
      new_sentence_heads=[]
      new_sentence_deprels=[]
      new_sentence_upos=[]
      new_sentence = conllu.models.TokenList()
   
      for token in sentence:
        # if the UD word is marked with a _, its not a valid word
        if (token['form'] != "_" and token['upos'] != "_"
        and token['id'] != "" and token['id'] > 0):
          new_sentence_ids.append(token['id'])
          new_sentence_forms.append(token['form'])
          new_sentence_heads.append(token['head'])
          new_sentence_deprels.append(token['deprel'])
          new_sentence_upos.append(token['upos']) 
          new_sentence.append(token)
        elif rewrite:
          print("Replacing in :", sentence)
       
      # Add the new sentence to the list
      new_sentences.append(new_sentence)
 
      ids.append(new_sentence_ids)
      forms.append(new_sentence_forms)
      heads.append(new_sentence_heads)
      deprels.append(new_sentence_deprels)
      upos.append(new_sentence_upos)
         
    return (ids, forms, heads, deprels, upos, new_sentences)

  def load_dictionaries(self, forms, deprels, upos):
    if len(self.form_tokenizer.word_index) <= 0:
      self.form_tokenizer.fit_on_texts(forms)
    if len(self.deprel_tokenizer.word_index) <= 0:
      self.deprel_tokenizer.fit_on_texts(deprels)
      self.deprel_tokenizer.fit_on_texts(["None"])
    if len(self.upos_tokenizer.word_index) <= 0:
      self.upos_tokenizer.fit_on_texts(upos)
    if len(self.action_tokenizer.word_index) <= 0:
      self.action_tokenizer.fit_on_texts([["LA","RA","REDUCE","SHIFT"]])

  # Gets the data from a path in a format ready to be used
  def get_Procesed_Data(self, path, replaceOOV, showSamples, rewrite):
    # Reads the file in universal dependecies format and returns a tuple with inputs and outputs
    (ids, forms, heads, deprels, upos, valid_sentences) = self.get_UD_Values(path, rewrite)

    # If true, replaces the OOV values in the output by the other classification
    if replaceOOV:
      # Get the tag correspondig to other from the dictionary
      tag_other = self.upos_tokenizer.word_index['x']

      # Replace all values in train_y that match the index of OOV in train_x
      for x in range(len(forms)):
        matches = [i for i,x in enumerate(forms[x]) if x==1]

        for indice in matches:
          upos[x][indice]= tag_other

    self.load_dictionaries(forms, deprels, upos)
   
    # If true, shows a sample of the values to manual validation
    if showSamples:
      print("Values sample 1: ")
      print("Ids: " , ids[0])
      print("Forms: " , forms[0])
      print("Heads: " , heads[0])
      print("Deprels: " , deprels[0])
      print("Upos: " , upos[0])
      print("Values sample 2: ")
      print("Ids: " , ids[1])
      print("Forms: " , forms[1])
      print("Heads: " , heads[1])
      print("Deprels: " , deprels[1])
      print("Upos: " , upos[1])
      print("Inputs Sentence count: " , len(ids))
      print("Dictionary forms count: " , len(self.form_tokenizer.word_index) + 1)
      print("Dictionary deprels count: " , len(self.deprel_tokenizer.word_index) + 1)
      print("Dictionary deprels list: " , self.deprel_tokenizer.word_index)
      print("Dictionary upos count: " , len(self.upos_tokenizer.word_index) + 1)
      print("Dictionary upos list: " , self.upos_tokenizer.word_index)
      print("Dictionary action count: " , len(self.action_tokenizer.word_index) + 1)
      print("Dictionary action list: " , self.action_tokenizer.word_index)
    
    return (ids, forms, heads, deprels, upos, valid_sentences, len(self.form_tokenizer.word_index) + 1, len(self.upos_tokenizer.word_index) + 1 , len(self.action_tokenizer.word_index) + 1, len(self.deprel_tokenizer.word_index) + 1)

  # Tokenize a value. Recieves the type of value as a parameter.
  def get_Tokenized_Values(self, values, value_type):
    # Depending on the value to detokenize use the correct tokenizer
    if value_type=="Form":
    # Transform text into values numbers
      coded_sequence = self.form_tokenizer.texts_to_sequences(values)
    elif value_type=="Upos":
      coded_sequence = self.upos_tokenizer.texts_to_sequences(values)
    elif value_type=="Deprel":
      coded_sequence = self.deprel_tokenizer.texts_to_sequences(values)
    elif value_type=="Action":
      coded_sequence = self.action_tokenizer.texts_to_sequences(values)
    return coded_sequence

  # Get the values padded and truncated
  def get_Padded_Values(self, values, padding_type):  
    # Pad the sequence with zeros after the last value and cut it if excedes the max len of the class
    return pad_sequences(values, padding=padding_type, maxlen=self.max_len, value=0, truncating='post')

  # Get the values in a one hot encode
  def get_OneHotEncoding_Values(self, values):
    hot_values = tf.keras.utils.to_categorical(values)
    return hot_values

  # Detokenize a value. Recieves the type of value as a parameter.
  def get_Detokenized_Values(self, values, value_type):
    # Depending on the value to detokenize use the correct tokenizer
    if value_type=="Form":
    # Transform number into values texts
      decoded_sequence = self.form_tokenizer.sequences_to_texts(values)
    elif value_type=="Upos":
      decoded_sequence = self.upos_tokenizer.sequences_to_texts(values)
    elif value_type=="Deprel":
      decoded_sequence = self.deprel_tokenizer.sequences_to_texts(values)
    elif value_type=="Action":
      decoded_sequence = self.action_tokenizer.sequences_to_texts(values)
    
    return decoded_sequence

  def get_Oracle(self, ids, forms, heads, deprels, upos, sentences, filter_non_projective, show_samples, show_non_projective, rewrite=False, path=""):
    # The dependency tree for each sentence can be represented using the column 7 (head)
    oracletraining = []
    dependenciestree = []
    # Create a empty list of sentences to use later to save without multi words
    new_sentences = conllu.models.SentenceList()
    count = 0

    for x in range(len(ids)):
      oh = My_Oracle_Simulator_Handler(ids[x], forms[x], heads[x], deprels[x], upos[x]) 

      # List of tuples with all the arcs of the sentence
      arcs = oh.get_arcs(ids[x], heads[x])
      # Just to show the dependencies for manual validation
      dependencies = []

      # Remove non-projective sentences (i.e., with crossing arcs) from your loaded corpus
      if oh.is_projective(arcs) or filter_non_projective==False:
        new_sentences.append(sentences[x])
        count = count + 1
        # Save the state to use later in predictions. Validate if this is what we want.
        states=[]
        with_heads=[]

        # While we have words to process (buffer not empty, stack not only with root)
        while len(oh.get_Buffer()) > 0:
          # σ|i - rightmost element from the stack
          i = oh.get_i()
          # j|β - leftmost element from the buffer
          j = oh.get_j()

          # 1. Check if we can apply Left arc
          # ¬[i = 0] - token i is not the artificial root node 0
          # ¬∃k∃l [(k, l , i) ∈ A] - token i does not already have a head
          # Head of i should be equal to id of j (namely, i is connected to j in the tree)
          if i is not None and i.get_uid() != 0 and i.get_has_head()==False and i.get_head()==j.get_uid():
            action = "LA"
            relation = i.get_deprel()
            # Mark that the unit has head
            i.set_has_head(True)
            arcs.remove((i.get_head(), i.get_uid()))
            dependencies.append((j.get_form(), relation, i.get_form()))

          # 2. Check if we can apply Right arc
          # ¬∃k∃l [(k, l , j) ∈ A] - j does not already have a head
          # Head of j should be equal to id of i (namely, j is connected to i in the tree)
          elif i is not None and j.get_has_head()==False and j.get_head()==i.get_uid():
            action= "RA"
            relation = j.get_deprel()
            # Mark that the unit has head
            j.set_has_head(True)
            arcs.remove((j.get_head(), j.get_uid()))
            dependencies.append((i.get_form(), relation, j.get_form()))

          # 3. Check if we can Reduce
          # The top token i (to be popped from the stack) has a head
          # The top token i should be already assigned to its "parents"
          elif i is not None and i.get_has_head() == True and oh.validate_pending_arcs(i.get_uid(), arcs) == False:
            action= "REDUCE"
            relation= "None"

          # 4. If none worked Shift
          else:
            action= "SHIFT"
            relation="None"
          
          # For now save like this, but the correct form is going to be defined when the model is created
          state = oh.get_State(action, relation)
          states.append(state)

          # Implement the action (REDUCE, SHIFT)
          oh.make_action(action)
          
        if len(states) > 0:
          oracletraining.append(states)
        elif show_samples:
          print("Sentence has no arcs: ", forms[x])

        dependenciestree.append(dependencies)
      elif show_non_projective:
        print("Sentence is not proyective: ", forms[x])

    if (show_samples):
      print("Total projective sentences: ", count)
      print("Dependencies: ", dependenciestree[0])
      print("Oracle training: ")
      for i in range(len(oracletraining[0])):
        print(oracletraining[0][i][0], oracletraining[0][i][1], oracletraining[0][i][2], oracletraining[0][i][3], oracletraining[0][i][4], oracletraining[0][i][5])

    # Save the cleaned dataset to the new path
    if rewrite:
      with open(path, 'w') as f:
        f.writelines([sentence.serialize() for sentence in new_sentences])

    return (oracletraining, dependenciestree, new_sentences)

  def get_Training(self, oracletraining, show_samples):
    list_stacks=[]
    list_stacks_upos=[]
    list_buffers=[]
    list_buffers_upos=[]
    list_actions=[]
    list_relations=[]

    for i in range(len(oracletraining)):
      for ii in range(len(oracletraining[i])): 
        list_stacks.append(oracletraining[i][ii][0])
        list_stacks_upos.append(oracletraining[i][ii][4])
        list_buffers.append(oracletraining[i][ii][1])
        list_buffers_upos.append(oracletraining[i][ii][5])
        list_actions.append(oracletraining[i][ii][2])
        list_relations.append(oracletraining[i][ii][3])
        
    list_coded_stacks = self.get_Tokenized_Values(list_stacks,"Form")
    list_coded_stacks_upos = self.get_Tokenized_Values(list_stacks_upos,"Upos")
    list_coded_buffers = self.get_Tokenized_Values(list_buffers,"Form")
    list_coded_buffers_upos = self.get_Tokenized_Values(list_buffers_upos,"Upos")
    list_coded_actions = self.get_Tokenized_Values(list_actions,"Action")
    list_coded_relations = self.get_Tokenized_Values(list_relations,"Deprel")

    list_padded_stacks = self.get_Padded_Values(list_coded_stacks,'pre')
    list_padded_stacks_upos = self.get_Padded_Values(list_coded_stacks_upos,'pre')
    list_padded_buffers = self.get_Padded_Values(list_coded_buffers,'post')
    list_padded_buffers_upos = self.get_Padded_Values(list_coded_buffers_upos,'post')
    list_binary_actions = self.get_OneHotEncoding_Values(list_coded_actions).tolist()
    list_binary_relations = self.get_OneHotEncoding_Values(list_coded_relations).tolist()

    # Not sure this is necessary
    list_train_words = []
    list_train_upos = []
    for x in range(len(list_padded_stacks)):
      words = np.concatenate((list_padded_stacks[x],list_padded_buffers[x]), axis=0).tolist()
      list_train_words.append(words)
      upos = np.concatenate((list_padded_stacks_upos[x],list_padded_buffers_upos[x]), axis=0).tolist()
      list_train_upos.append(upos)
      
    if show_samples:
      print("Training samples 1: ")
      print("Stack: ", list_stacks[0])
      print("Coded Stack: ", list_coded_stacks[0])
      print("Padded Stack: ", list_padded_stacks[0])
      print("Buffer: ", list_buffers[0])
      print("Coded Buffer: ", list_coded_buffers[0])
      print("Padded Buffer: ", list_padded_buffers[0])
      print("Train words: ", list_train_words[0])
      print("Stack upos: ", list_stacks_upos[0])
      print("Coded Stack upos: ", list_coded_stacks_upos[0])
      print("Padded Stack upos: ", list_padded_stacks_upos[0])
      print("Buffer upos: ", list_buffers_upos[0])
      print("Coded Buffer upos: ", list_coded_buffers_upos[0])
      print("Padded Buffer upos: ", list_padded_buffers_upos[0])
      print("Train upos: ", list_train_upos[0])
      print("Action: ", list_actions[0])
      print("Coded Action: ", list_coded_actions[0])
      print("Binary Action: ", list_binary_actions[0])
      print("Relation: ", list_relations[0])
      print("Coded Relation: ", list_coded_relations[0])
      print("Binary Relation: ", list_binary_relations[0])
      print("Training samples 2: ")
      print("Stack: ", list_stacks[1])
      print("Coded Stack: ", list_coded_stacks[1])
      print("Padded Stack: ", list_padded_stacks[1])
      print("Buffer: ", list_buffers[1])
      print("Coded Buffer: ", list_coded_buffers[1])
      print("Padded Buffer: ", list_padded_buffers[1])
      print("Train words: ", list_train_words[1])
      print("Stack upos: ", list_stacks_upos[1])
      print("Coded Stack upos: ", list_coded_stacks_upos[1])
      print("Padded Stack upos: ", list_padded_stacks_upos[1])
      print("Buffer upos: ", list_buffers_upos[1])
      print("Coded Buffer upos: ", list_coded_buffers_upos[1])
      print("Padded Buffer upos: ", list_padded_buffers_upos[1])
      print("Train upos: ", list_train_upos[1])
      print("Action: ", list_actions[1])
      print("Coded Action: ", list_coded_actions[1])
      print("Binary Action: ", list_binary_actions[1])
      print("Relation: ", list_relations[1])
      print("Coded Relation: ", list_coded_relations[1])
      print("Binary Relation: ", list_binary_relations[1])

    return (list_padded_stacks, list_padded_buffers, list_padded_stacks_upos, list_padded_buffers_upos, list_binary_actions, list_binary_relations)

  def get_Real_Training(self, stack, buff, stack_upos, buff_upos, show_samples):
    coded_stacks = self.get_Tokenized_Values(stack,"Form")
    coded_stacks_upos = self.get_Tokenized_Values(stack_upos,"Upos")
    coded_buffers = self.get_Tokenized_Values(buff,"Form")
    coded_buffers_upos = self.get_Tokenized_Values(buff_upos,"Upos")

    padded_stacks = self.get_Padded_Values(coded_stacks,'pre')
    padded_stacks_upos = self.get_Padded_Values(coded_stacks_upos,'pre')
    padded_buffers = self.get_Padded_Values(coded_buffers,'post')
    padded_buffers_upos = self.get_Padded_Values(coded_buffers_upos,'post')

    words = np.concatenate((padded_stacks[0],padded_buffers[0]), axis=0).tolist()
    upos = np.concatenate((padded_stacks_upos[0],padded_buffers_upos[0]), axis=0).tolist()

    if show_samples:
      print("Stack: ", stack[0])
      print("Coded Stack: ", coded_stacks[0])
      print("Padded Stack: ", padded_stacks[0])
      print("Buffer: ", buff[0])
      print("Coded Buffer: ", coded_buffers[0])
      print("Padded Buffer: ", padded_buffers[0])
      print("Words: ", words)
      print("Stack upos: ", stack_upos[0])
      print("Coded Stack upos: ", coded_stacks_upos[0])
      print("Padded Stack upos: ", padded_stacks_upos[0])
      print("Buffer upos: ", buff_upos[0])
      print("Coded Buffer upos: ", coded_buffers_upos[0])
      print("Padded Buffer upos: ", padded_buffers_upos[0])
      print("Upos: ", upos)

    return (padded_stacks, padded_buffers, padded_stacks_upos, padded_buffers_upos)

  def get_Real_Oracle(self, sentences, show_samples, model, path, use_upos=False, model_p1=None, dh_p1 = None):
    dependenciestree = []
    # Create a empty list of sentences to use later to save the predictions
    new_sentences = conllu.models.SentenceList()

    for x in range(len(sentences)):
    #for x in range(1):
      oh = My_Oracle_Test_Handler(sentences[x], model_p1=model_p1, dh_p1=dh_p1) 

      dependencies = []
      hasRootHead = False

      # While we have words to process (buffer not empty)
      while len(oh.get_Buffer()) > 0:
         # σ|i - rightmost element from the stack
        i = oh.get_i()
        # j|β - leftmost element from the buffer
        j = oh.get_j()
 
        # Get state and predict
        (stack_names, buffer_names, stack_upos, buffer_upos) = oh.get_State()
        (pred_stack, pred_buffer, pred_stack_upos, pred_buffer_upos) = self.get_Real_Training([stack_names], [buffer_names], [stack_upos], [buffer_upos], show_samples)
        if (use_upos):
          (pre_action,pre_relation) = model.predict([pred_stack, pred_buffer, pred_stack_upos, pred_buffer_upos])
        else:
          (pre_action,pre_relation) = model.predict([pred_stack, pred_buffer])

        arg=-1
        num_relation = np.argmax(pre_relation)
        relation = self.get_Detokenized_Values([[num_relation]], "Deprel")[0]
        
        # In case is a bad predicted value, save the deprel
        if j.get_deprel()=='':
          j.set_deprel(relation)

        try_again = True

        # If I cant apply the action, try again with a new one
        while (try_again == True):
          # Get the next value
          num_action = np.argsort(pre_action, axis=1)[:,arg][0]
          action = self.get_Detokenized_Values([[num_action]], "Action")[0].upper()     
 
          # If is a left arc and does not have head 
          if action == "LA" and i.get_word_id() != 0 and i.get_has_head()==False:
            # I cant have two heads root
            if (j.get_word_id()==0 and hasRootHead == True):
              arg=arg-1  
            else:
              i.set_has_head(True, j.get_word_id())
              i.set_deprel(relation)
              dependencies.append((j.get_form(), relation, i.get_form()))
              try_again = False
              # If root, then the sentence have a predicted head
              if (j.get_word_id()==0):
                hasRootHead = True
  
          # If is a right arc and does not have head
          elif action == "RA" and j.get_has_head()==False:
            # I cant have two heads root
            if (i.get_word_id()==0 and hasRootHead == True):
              arg=arg-1  
            else:
              j.set_has_head(True, i.get_word_id())
              j.set_deprel(relation)
              dependencies.append((i.get_form(), relation, j.get_form()))
              try_again = False
              # If root, then the sentence have a predicted head
              if (i.get_word_id()==0):
                hasRootHead = True

          # If is a reduce and the item of the stack does not have head
          elif action == "REDUCE" and i is not None and i.get_has_head() == True:
            try_again = False

          # Just shift
          elif action == "SHIFT":
            try_again = False
          
          # If any action cant be applied, get a new one
          else:
            arg=arg-1     
              
        # Implement the action
        oh.make_action(action)

      new_sentence = oh.get_sentence()

      # All lost head will be assigned to the element wich head is the root, unless there is no root.
      # In that case, use the most left element WITHOUT A HEAD and asign it to the root.
      lost_heads_id = 0   
      lost_heads_form = ''    
      for token in new_sentence:
        if (hasRootHead == False and token['head']==-1):
          token['head'] = 0
          token['deprel'] = 'root'
          dependencies.append(('ROOT', 'root', token['form']))
          hasRootHead = True
          lost_heads = token['id']
          lost_heads_form = token['form']
          break
        elif (token['head']==0):
          lost_heads = token['id']
          lost_heads_form = token['form']

      for token in new_sentence:
        if token['head']==-1:
          token['head'] = lost_heads
          dependencies.append((lost_heads_form, token['deprel'], token['form']))

      new_sentences.append(new_sentence)
      dependenciestree.append(dependencies)
      
    # Save the cleaned dataset to the new path
    with open(path, 'w') as f:
      f.writelines([sentence.serialize() for sentence in new_sentences])

    return dependenciestree

  def get_Predictions(self, sentences, show_samples, model, model_p1=None, dh_p1 = None):
    dependenciestree = []
    # Create a empty list of sentences to use later to save the predictions
    new_sentences = conllu.models.SentenceList()

    for x in range(len(sentences)):
      oh = My_Oracle_Predict_Handler(sentences[x], model_p1=model_p1, dh_p1=dh_p1) 

      dependencies = []
      hasRootHead = False

      # While we have words to process (buffer not empty)
      while len(oh.get_Buffer()) > 0:
         # σ|i - rightmost element from the stack
        i = oh.get_i()
        # j|β - leftmost element from the buffer
        j = oh.get_j()
 
        # Get state and predict
        (stack_names, buffer_names, stack_upos, buffer_upos) = oh.get_State()
        (pred_stack, pred_buffer, pred_stack_upos, pred_buffer_upos) = self.get_Real_Training([stack_names], [buffer_names], [stack_upos], [buffer_upos], show_samples)
        (pre_action,pre_relation) = model.predict([pred_stack, pred_buffer, pred_stack_upos, pred_buffer_upos])

        arg=-1
        num_relation = np.argmax(pre_relation)
        relation = self.get_Detokenized_Values([[num_relation]], "Deprel")[0]
        
        # In case is a bad predicted value, save the deprel
        if j.get_deprel()=='':
          j.set_deprel(relation)

        try_again = True

        # If I cant apply the action, try again with a new one
        while (try_again == True):
          # Get the next value
          num_action = np.argsort(pre_action, axis=1)[:,arg][0]
          action = self.get_Detokenized_Values([[num_action]], "Action")[0].upper()     

          # If is a left arc and does not have head 
          if action == "LA" and i.get_word_id() != 0 and i.get_has_head()==False:
            # I cant have two heads root
            if (j.get_word_id()==0 and hasRootHead == True):
              arg=arg-1  
            else:
              i.set_head(j.get_word_id())
              i.set_deprel(relation)
              dependencies.append((j.get_form(), relation, i.get_form()))
              try_again = False
              # If root, then the sentence have a predicted head
              if (j.get_word_id()==0):
                hasRootHead = True
  
          # If is a right arc and does not have head
          elif action == "RA" and j.get_has_head()==False:
            # I cant have two heads root
            if (i.get_word_id()==0 and hasRootHead == True):
              arg=arg-1  
            else:
              j.set_head(i.get_word_id())
              j.set_deprel(relation)
              dependencies.append((i.get_form(), relation, j.get_form()))
              try_again = False
              # If root, then the sentence have a predicted head
              if (i.get_word_id()==0):
                hasRootHead = True

          # If is a reduce and the item of the stack does not have head
          elif action == "REDUCE" and i is not None and i.get_has_head() == True:
            try_again = False

          # Just shift
          elif action == "SHIFT":
            try_again = False
          
          # If any action cant be applied, get a new one
          else:
            arg=arg-1     
              
        # Implement the action
        oh.make_action(action)

      units = oh.get_units()

      # All lost head will be assigned to the element wich head is the root, unless there is no root.
      # In that case, use the most left element WITHOUT A HEAD and asign it to the root.
      lost_heads_id = 0  
      lost_heads_form = ''  

      for token in units:
        if (hasRootHead == False and token.get_head()==-1):
          token.set_head(0)
          token.set_deprel('root')
          dependencies.append(('ROOT', 'root', token.get_form()))
          hasRootHead = True
          lost_heads = token.get_word_id()
          lost_heads_form = token.get_form()
          break
        elif (token.get_head()==0):
          lost_heads = token.get_word_id()
          lost_heads_form = token.get_form()

      for token in units:
        if token.get_head()==-1:
          token.set_head(lost_heads)
          dependencies.append((lost_heads_form, token.get_deprel(), token.get_form()))

      dependenciestree.append(dependencies)     
   
    return dependenciestree
