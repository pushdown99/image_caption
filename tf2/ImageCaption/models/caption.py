import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.applications import efficientnet
from ..config import *

class TransformerEncoderBlock(layers.Layer):
  def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
    super().__init__(**kwargs)
    self.embed_dim = embed_dim
    self.dense_dim = dense_dim
    self.num_heads = num_heads
    self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
    self.dense_proj = layers.Dense(embed_dim, activation="relu")
    self.layernorm_1 = layers.LayerNormalization()

  def call(self, inputs, training, mask=None):
    inputs = self.dense_proj(inputs)
    attention_output = self.attention(query=inputs, value=inputs, key=inputs, attention_mask=None)
    proj_input = self.layernorm_1(inputs + attention_output)
    return proj_input

class PositionalEmbedding(layers.Layer):
  def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
    super().__init__(**kwargs)
    self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
    self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)
    self.sequence_length = sequence_length
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim

  def call(self, inputs):
    length = tf.shape(inputs)[-1]
    positions = tf.range(start=0, limit=length, delta=1)
    embedded_tokens = self.token_embeddings(inputs)
    embedded_positions = self.position_embeddings(positions)
    return embedded_tokens + embedded_positions

  def compute_mask(self, inputs, mask=None):
    return tf.math.not_equal(inputs, 0)

class TransformerDecoderBlock(layers.Layer):
  def __init__(self, embed_dim, ff_dim, num_heads, vocab_size, **kwargs):
    super().__init__(**kwargs)
    self.embed_dim      = embed_dim
    self.ff_dim         = ff_dim
    self.num_heads      = num_heads
    self.vocab_size     = vocab_size
    self.attention_1    = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
    self.attention_2    = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
    self.dense_proj     = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
    self.layernorm_1    = layers.LayerNormalization()
    self.layernorm_2    = layers.LayerNormalization()
    self.layernorm_3    = layers.LayerNormalization()

    self.embedding = PositionalEmbedding(embed_dim=EMBED_DIM, sequence_length=SEQ_LENGTH, vocab_size=self.vocab_size)
    self.out = layers.Dense(self.vocab_size)
    self.dropout_1 = layers.Dropout(0.1)
    self.dropout_2 = layers.Dropout(0.5)
    self.supports_masking = True


  def call(self, inputs, encoder_outputs, training, mask=None):
    inputs = self.embedding(inputs)
    causal_mask = self.get_causal_attention_mask(inputs)
    inputs = self.dropout_1(inputs, training=training)

    if mask is not None:
      padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
      combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
      combined_mask = tf.minimum(combined_mask, causal_mask)
    else :
      combined_mask = None
      padding_mask  = None

    attention_output_1 = self.attention_1(
      query=inputs, value=inputs, key=inputs, attention_mask=combined_mask#None
    )
    out_1 = self.layernorm_1(inputs + attention_output_1)

    attention_output_2 = self.attention_2(
      query=out_1, value=encoder_outputs, key=encoder_outputs, attention_mask=padding_mask#None
    )
    out_2 = self.layernorm_2(out_1 + attention_output_2)

    proj_output = self.dense_proj(out_2)
    proj_out = self.layernorm_3(out_2 + proj_output)
    proj_out = self.dropout_2(proj_out, training=training)

    preds = self.out(proj_out)
    return preds

  def get_causal_attention_mask(self, inputs):
    input_shape = tf.shape(inputs)
    batch_size, sequence_length = input_shape[0], input_shape[1]
    i = tf.range(sequence_length)[:, tf.newaxis]
    j = tf.range(sequence_length)
    mask = tf.cast(i >= j, dtype="int32")
    mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
    mult = tf.concat([tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], axis=0,)
    return tf.tile(mask, mult)

class Model(keras.Model):
  def __init__(self, tokennizer, vocab_size, captions_image=5, verbose=True):
    super().__init__()
    self.cnn_model      = self.get_cnn_model()
    self.batch_size     = BATCH_SIZE
    self.epochs         = EPOCHS
    self.tokennizer     = tokennizer
    self.vocab_size     = vocab_size
    self.encoder        = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=NUM_HEADS)
    self.decoder        = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=NUM_HEADS, vocab_size=self.vocab_size)
    self.loss_tracker   = keras.metrics.Mean(name="loss")
    self.acc_tracker 	= keras.metrics.Mean(name="accuracy")
    self.cross_entropy  = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    self.early_stop     = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    self.lr_scheduler   = custom_schedule(EMBED_DIM)
    self.optimizer      = keras.optimizers.Adam(learning_rate=self.lr_scheduler, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    self.verbose        = verbose
    self.captions_image = captions_image

  def Dataset(self, train_dataset, valid_dataset):
    self.train_dataset  = train_dataset
    self.valid_dataset  = valid_dataset

  def Compile(self):
    super().compile (optimizer=self.optimizer, loss=self.cross_entropy)

  def Fit(self):
    self._hist = super().fit(self.train_dataset, epochs=self.epochs, validation_data=self.valid_dataset, callbacks=[self.early_stop])
    return self._hist

  def Evaluate(self):
    self.train_metrics = super().evaluate(self.train_dataset, batch_size=self.batch_size)
    self.valid_metrics = super().evaluate(self.valid_dataset, batch_size=self.batch_size)
    if TEST_DATASET:
        self.test_metrics = super().evaluate(self.test_dataset, batch_size=self.batch_size)

    if self.verbose:
      print("Train metrics  : loss {}, accuracy {}".format(self.train_metrics[0], self.train_metrics[1])) 
      print("Valid metrics  : loss {}, accuracy {}".format(self.valid_metrics[0], self.valid_metrics[1])) 
      if TEST_DATASET:
        print("Test metrics   : loss {}, accuracy {}".format(self.test_metrics[0], self.test_metrics[1])) 

  def Save(self):
    if not os.path.exists(SAVE_DIR):
      os.mkdir(SAVE_DIR)

    # Save training history under the form of a json file
    self._histdict = self._hist.history
    json.dump(history_dict, open(os.path.join(SAVE_DIR, 'history.json'), 'w'))

    # Save weights model
    super().save_weights(os.path.join(SAVE_DIR, 'model.h5'))

    # Save config model train
    config_train = {"IMAGE_SIZE"	: IMAGE_SIZE,
                    "MAX_VOCAB_SIZE": MAX_VOCAB_SIZE,
                    "SEQ_LENGTH" 	: SEQ_LENGTH,
                    "EMBED_DIM" 	: EMBED_DIM,
                    "NUM_HEADS" 	: NUM_HEADS,
                    "FF_DIM" 		: FF_DIM,
                    "BATCH_SIZE" 	: self.batch_size,
                    "EPOCHS" 		: EPOCHS,
                    "VOCAB_SIZE" 	: self.vocab_size}
    json.dump(config_train, open(os.path.join(SAVE_DIR, 'config.json'), 'w'))

    # Save tokenizer model
    self.save_tokenizer(self.tokenizer, SAVE_DIR)

  def save_tokenizer(tokenizer, path_save):
    input 	= tf.keras.layers.Input(shape=(1,), dtype=tf.string)
    output 	= self.tokenizer(input)
    model 	= tf.keras.Model(input, output)
    #model.save(path_save + "tokenizer", save_format='tf')
    super().save(os.path.join(SAVE_DIR, 'tokenizer'))

  def get_cnn_model(self):
    base_model = efficientnet.EfficientNetB0(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet",)
    base_model.trainable = False
    base_model_out = base_model.output
    base_model_out = layers.Reshape((-1, 1280))(base_model_out)
    cnn_model = keras.models.Model(base_model.input, base_model_out)
    return cnn_model

  def call(self, inputs):
    x = self.cnn_model(inputs[0])
    x = self.encoder(x, False)
    x = self.decoder(inputs[2],x,training=inputs[1],mask=None)
    return x

  def calculate_loss(self, y_true, y_pred, mask):
    loss = self.loss(y_true, y_pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

  def calculate_accuracy(self, y_true, y_pred, mask):
    accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
    accuracy = tf.math.logical_and(mask, accuracy)
    accuracy = tf.cast(accuracy, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

  def train_step(self, batch_data):
    batch_img, batch_seq = batch_data
    batch_loss 	= 0
    batch_acc 	= 0

    # 1. Get image embeddings
    img_embed = self.cnn_model(batch_img)

    # 2. Pass each of the five captions one by one to the decoder
    # along with the encoder outputs and compute the loss as well as accuracy
    # for each caption.
    for i in range(self.captions_image):
      with tf.GradientTape() as tape:
        # 3. Pass image embeddings to encoder
        encoder_out = self.encoder(img_embed, training=True)

        batch_seq_inp = batch_seq[:, i, :-1]
        batch_seq_true = batch_seq[:, i, 1:]

        # 4. Compute the mask for the input sequence
        mask = tf.math.not_equal(batch_seq_inp, 0)

        # 5. Pass the encoder outputs, sequence inputs along with
        # mask to the decoder
        batch_seq_pred = self.decoder(batch_seq_inp, encoder_out, training=True, mask=mask)

        # 6. Calculate loss and accuracy
        caption_loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
        caption_acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)

        # 7. Update the batch loss and batch accuracy
        batch_loss += caption_loss
        batch_acc += caption_acc

      # 8. Get the list of all the trainable weights
      train_vars = (self.encoder.trainable_variables + self.decoder.trainable_variables)

      # 9. Get the gradients
      grads = tape.gradient(caption_loss, train_vars)

      # 10. Update the trainable weights
      self.optimizer.apply_gradients(zip(grads, train_vars))

    loss = batch_loss
    acc = batch_acc / float(self.captions_image)

    self.loss_tracker.update_state(loss)
    self.acc_tracker.update_state(acc)
    return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

  def test_step(self, batch_data):
    batch_img, batch_seq = batch_data
    batch_loss  = 0
    batch_acc   = 0

    # 1. Get image embeddings
    img_embed = self.cnn_model(batch_img)

    # 2. Pass each of the five captions one by one to the decoder
    # along with the encoder outputs and compute the loss as well as accuracy
    # for each caption.
    for i in range(self.captions_image):
      # 3. Pass image embeddings to encoder
      encoder_out = self.encoder(img_embed, training=False)

      batch_seq_inp = batch_seq[:, i, :-1]
      batch_seq_true = batch_seq[:, i, 1:]

      # 4. Compute the mask for the input sequence
      mask = tf.math.not_equal(batch_seq_inp, 0)

      # 5. Pass the encoder outputs, sequence inputs along with
      # mask to the decoder
      batch_seq_pred = self.decoder(batch_seq_inp, encoder_out, training=False, mask=mask)

      # 6. Calculate loss and accuracy
      caption_loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
      caption_acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)

      # 7. Update the batch loss and batch accuracy
      batch_loss += caption_loss
      batch_acc += caption_acc

    loss = batch_loss
    acc = batch_acc / float(self.captions_image)

    self.loss_tracker.update_state(loss)
    self.acc_tracker.update_state(acc)
    return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

  @property
  def metrics(self):
    # We need to list our metrics here so the `reset_states()` can be
    # called automatically.
    return [self.loss_tracker, self.acc_tracker]

@tf.keras.utils.register_keras_serializable()
class custom_schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
   def __init__(self, d_model, warmup_steps=4000):
      super(custom_schedule, self).__init__()
      self.d_model = d_model
      self.d_model = tf.cast(self.d_model, tf.float32)
      self.warmup_steps = warmup_steps

   def __call__(self, step):
      arg1 = tf.math.rsqrt(step)
      arg2 = step * (self.warmup_steps ** -1.5)
      return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

   def get_config(self):
      config = {
        'd_model': self.d_model,
        'warmup_steps': self.warmup_steps
        }
      return config

