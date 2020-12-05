# import flask.Flask
# from flask import Flask, url_for

import tensorflow as tf

from flask import Flask, render_template, request
app = Flask(__name__)




import tensorflow as tf
try:
  import tensorflow.compat.v2 as tf
except Exception:
  pass

tf.enable_v2_behavior()


from tensorflow import keras
import tensorflow_datasets as tfds
import os
import re
import numpy as np
import pandas as pd
import json




def x_preprocess_sentence(sentence):
  sentence = sentence.lower()
  sentence = sentence.replace("’", "'").replace("`", "'")
  sentence = re.sub(r'["."]+', ".", sentence)
  sentence = re.sub(r'[","]+', ",", sentence)
  sentence = re.sub(r'["!"]+', "!", sentence)
  sentence = re.sub(r'["?"]+', "?", sentence)
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)

  sentence = sentence.replace("i'm", "i am")
  sentence = sentence.replace("you're", "you are")
  sentence = sentence.replace("he's", "he is")
  sentence = sentence.replace("she's", "she is")
  sentence = sentence.replace("they're", "they are")
  sentence = sentence.replace("we're", "we are")
  sentence = sentence.replace("it's", "it is")
  sentence = sentence.replace("that's", "that is")
  sentence = sentence.replace("here's", "here is")
  sentence = sentence.replace("i'll", "i will")
  sentence = sentence.replace("you’ll", "you will")
  sentence = sentence.replace("he'll", "he will")
  sentence = sentence.replace("she'll", "she will")
  sentence = sentence.replace("it'll", "it will")
  sentence = sentence.replace("we'll", "we will")
  sentence = sentence.replace("haven't", "have not")
  sentence = sentence.replace("i've", "i have")
  sentence = sentence.replace("you've", "you have")
  sentence = sentence.replace("he's", "he has")
  sentence = sentence.replace("she's", "she has")
  sentence = sentence.replace("we've", "we have")
  sentence = sentence.replace("they've", "they have")
  sentence = sentence.replace("should've", "should have")
  sentence = sentence.replace("could've", "could have")
  sentence = sentence.replace("i'd", "i would")
  sentence = sentence.replace("you'd", "you would")
  sentence = sentence.replace("he'd", "he would")
  sentence = sentence.replace("she'd", "she would")
  sentence = sentence.replace("we'd", "we would")
  sentence = sentence.replace("they'd", "they would")
  sentence = sentence.replace("don't", "do not")
  sentence = sentence.replace("can't", "can not")
  sentence = sentence.replace("aren't", "are not")
  sentence = sentence.replace("couldn't", "could not")
  sentence = sentence.replace("wouldn't", "would not")
  sentence = sentence.replace("shouldn't", "should not")
  sentence = sentence.replace("isn't", "is not")
  sentence = sentence.replace("doesn't", "does not")
  sentence = sentence.replace("didn't", "did not")
  sentence = sentence.replace("hasn't", "has not")
  sentence = sentence.replace("wasn't", "was not")
  sentence = sentence.replace("won't", "will not")
  sentence = sentence.replace("weren't", "were not")
  sentence = sentence.replace("let's", "let us")
  sentence = sentence.replace("y'all", "you all")

  sentence = re.sub(r"[^a-z?.!,]+", " ", sentence)
  sentence = re.sub(r'[" "]+', " ", sentence)
  sentence = sentence.strip()
  return sentence


# MAX_SAMPLES = 9999999999
# path_to_movie_lines = './movie_lines.txt'
# path_to_movie_conversations = './movie_conversations.txt'

# def load_conversations():
#   # dictionary of line id to text
#   id2line = {}
#   with open(path_to_movie_lines, errors='ignore') as file:
#     lines = file.readlines()
#   for line in lines:
#     parts = line.replace('\n', '').split(' +++$+++ ')
#     id2line[parts[0]] = parts[4]

#   inputs, outputs = [], []
#   with open(path_to_movie_conversations, 'r') as file:
#     lines = file.readlines()
#   for line in lines:
#     parts = line.replace('\n', '').split(' +++$+++ ')
#     # get conversation in a list of line ID
#     conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]
#     for i in range(len(conversation) - 1):
#       inputs.append(x_preprocess_sentence(id2line[conversation[i]]))
#       outputs.append(x_preprocess_sentence(id2line[conversation[i + 1]]))
#       if len(inputs) >= MAX_SAMPLES:
#         return inputs, outputs
#   return inputs, outputs

# questions_films, answers_films = load_conversations()


# questions = []
# answers = []
# for ar_n in range(2):
#   # print(ar_n)
#   with open('./questions_answers_12/'+str(ar_n)+'/questions.json', 'r') as f:
#       questions = questions + json.load(f)
#   with open('./questions_answers_12/'+str(ar_n)+'/answers.json', 'r') as f:
#       answers = answers + json.load(f)

# questions = questions + questions_films
# answers = answers + answers_films


# save
# tokenizer.save_to_file('./questions_answers_12/tokenizer')
# Load
tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file('./questions_answers_12/tokenizer')

# Define start and end token to indicate the start and end of a sentence
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# Vocabulary size plus start and end token
VOCAB_SIZE = tokenizer.vocab_size + 2





def scaled_dot_product_attention(query, key, value, mask):
  """Calculate the attention weights. """
  matmul_qk = tf.matmul(query, key, transpose_b=True)

  # scale matmul_qk
  depth = tf.cast(tf.shape(key)[-1], tf.float32)
  logits = matmul_qk / tf.math.sqrt(depth)

  # add the mask to zero out padding tokens
  if mask is not None:
    logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k)
  attention_weights = tf.nn.softmax(logits, axis=-1)

  output = tf.matmul(attention_weights, value)

  return output


class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, name="multi_head_attention"):
    super(MultiHeadAttention, self).__init__(name=name)
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.query_dense = tf.keras.layers.Dense(units=d_model)
    self.key_dense = tf.keras.layers.Dense(units=d_model)
    self.value_dense = tf.keras.layers.Dense(units=d_model)

    self.dense = tf.keras.layers.Dense(units=d_model)

  def split_heads(self, inputs, batch_size):
    inputs = tf.reshape(
        inputs, shape=(batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(inputs, perm=[0, 2, 1, 3])

  def call(self, inputs):
    query, key, value, mask = inputs['query'], inputs['key'], inputs[
        'value'], inputs['mask']
    batch_size = tf.shape(query)[0]

    # linear layers
    query = self.query_dense(query)
    key = self.key_dense(key)
    value = self.value_dense(value)

    # split heads
    query = self.split_heads(query, batch_size)
    key = self.split_heads(key, batch_size)
    value = self.split_heads(value, batch_size)

    # scaled dot-product attention
    scaled_attention = scaled_dot_product_attention(query, key, value, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    # concatenation of heads
    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))

    # final linear layer
    outputs = self.dense(concat_attention)

    return outputs


def create_padding_mask(x):
  mask = tf.cast(tf.math.equal(x, 0), tf.float32)
  # (batch_size, 1, 1, sequence length)
  return mask[:, tf.newaxis, tf.newaxis, :]


create_padding_mask(tf.constant([[1, 2, 0, 3, 0], [0, 0, 0, 4, 5]]))

def create_look_ahead_mask(x):
  seq_len = tf.shape(x)[1]
  look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
  padding_mask = create_padding_mask(x)
  return tf.maximum(look_ahead_mask, padding_mask)


create_look_ahead_mask(tf.constant([[1, 2, 0, 4, 5]]))


class PositionalEncoding(tf.keras.layers.Layer):

  def __init__(self, position, d_model):
    super(PositionalEncoding, self).__init__()
    self.pos_encoding = self.positional_encoding(position, d_model)

  def get_angles(self, position, i, d_model):
    angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    return position * angles

  def positional_encoding(self, position, d_model):
    angle_rads = self.get_angles(
        position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
        i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
        d_model=d_model)
    # apply sin to even index in the array
    sines = tf.math.sin(angle_rads[:, 0::2])
    # apply cos to odd index in the array
    cosines = tf.math.cos(angle_rads[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return tf.cast(pos_encoding, tf.float32)

  def call(self, inputs):
    return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


sample_pos_encoding = PositionalEncoding(50, 512)


def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  attention = MultiHeadAttention(
      d_model, num_heads, name="attention")({
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': padding_mask
      })
  attention = tf.keras.layers.Dropout(rate=dropout)(attention)
  attention = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(inputs + attention)

  outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention + outputs)

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)


sample_encoder_layer = encoder_layer(
    units=512,
    d_model=128,
    num_heads=4,
    dropout=0.3,
    name="sample_encoder_layer")


def encoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name="encoder"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

  for i in range(num_layers):
    outputs = encoder_layer(
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name="encoder_layer_{}".format(i),
    )([outputs, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)


sample_encoder = encoder(
    vocab_size=8192,
    num_layers=2,
    units=512,
    d_model=128,
    num_heads=4,
    dropout=0.3,
    name="sample_encoder")


def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
  enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name="look_ahead_mask")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

  attention1 = MultiHeadAttention(
      d_model, num_heads, name="attention_1")(inputs={
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': look_ahead_mask
      })
  attention1 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention1 + inputs)

  attention2 = MultiHeadAttention(
      d_model, num_heads, name="attention_2")(inputs={
          'query': attention1,
          'key': enc_outputs,
          'value': enc_outputs,
          'mask': padding_mask
      })
  attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
  attention2 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention2 + attention1)

  outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(outputs + attention2)

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)


sample_decoder_layer = decoder_layer(
    units=512,
    d_model=128,
    num_heads=4,
    dropout=0.3,
    name="sample_decoder_layer")


def decoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name='decoder'):
  inputs = tf.keras.Input(shape=(None,), name='inputs')
  enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name='look_ahead_mask')
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
  
  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

  for i in range(num_layers):
    outputs = decoder_layer(
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name='decoder_layer_{}'.format(i),
    )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)



sample_decoder = decoder(
    vocab_size=8192,
    num_layers=2,
    units=512,
    d_model=128,
    num_heads=4,
    dropout=0.3,
    name="sample_decoder")



def transformer(vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name="transformer"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")
  dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

  enc_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='enc_padding_mask')(inputs)
  # mask the future tokens for decoder inputs at the 1st attention block
  look_ahead_mask = tf.keras.layers.Lambda(
      create_look_ahead_mask,
      output_shape=(1, None, None),
      name='look_ahead_mask')(dec_inputs)
  # mask the encoder outputs for the 2nd attention block
  dec_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='dec_padding_mask')(inputs)

  enc_outputs = encoder(
      vocab_size=vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
  )(inputs=[inputs, enc_padding_mask])

  dec_outputs = decoder(
      vocab_size=vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
  )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

  outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

  return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)



sample_transformer = transformer(
    vocab_size=8192,
    num_layers=4,
    units=512,
    d_model=128,
    num_heads=4,
    dropout=0.3,
    name="sample_transformer")



def create_model():

  # Hyper-parameters
  NUM_LAYERS = 2
  D_MODEL = 256
  NUM_HEADS = 8
  UNITS = 512
  DROPOUT = 0.1

  return transformer(
      vocab_size=VOCAB_SIZE,
      num_layers=NUM_LAYERS,
      units=UNITS,
      d_model=D_MODEL,
      num_heads=NUM_HEADS,
      dropout=DROPOUT)

model = create_model()

MAX_LENGTH = 14



model.load_weights('./questions_answers_12/weights/max_DNN_TPU_1024.h5')



def evaluate(sentence):

  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  output = tf.expand_dims(START_TOKEN, 0)

  for i in range(MAX_LENGTH):
    predictions = model(inputs=[sentence, output], training=False)
    # print('predictions: {}'.format(sentence))

    # predictions = cpu_model(inputs=[sentence, output], training=False)

    # select the last word from the seq_len dimension
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # return the result if the predicted_id is equal to the end token
    if tf.equal(predicted_id, END_TOKEN[0]):
      break

    # concatenated the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)


def predict(sentence):
  sentence = x_preprocess_sentence(sentence)
  if len(sentence.split(' ')) > 12:
    return 'length should be not more than 12'

  my_answer = ''
  for h_t in hand_translate:
    if h_t in sentence:
      if ('hello' in sentence) or ('hi' in sentence):
        my_answer = my_answer + 'Hi. '
      my_answer = my_answer + hand_translate[h_t]
      break
  if my_answer != '':
    return my_answer

  

  prediction = evaluate(sentence)

  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])

  return predicted_sentence



hand_translate = {
    'how are you':"I'm fine. Thank you",
    "are you robot":"Yes. I'm robot. My name is Roborobot",
    "what is your name":"my name is Roborobot",
    "what is your profession":"i work as bot",
    "where you from":"i from Mars"
}





@app.route("/chatbot", methods=['GET'])
def get_form(name=None):
    return render_template('index.html', name=name)

@app.route("/chatbot", methods=['POST'])
def get_answer(name=None):
    data = request.json
    return predict(data)
    # return "answer"+data

if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=4567)
# app.run(host='0.0.0.0', port=4567)