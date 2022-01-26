import tensorflow as tf
from tensorflow.keras.layers import Dense, AdditiveAttention, LSTM

class Encoder(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Encoder, self).__init__()
        self.units = units
        self.lstm = LSTM(
            self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
    
    def call(self, inputs, state=None):
        output, state_h, state_c = self.lstm(inputs, initial_state=state)
        return output, [state_h, state_c]
class Decoder(tf.keras.layers.Layer):
    def __init__(self, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.lstm = LSTM(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.attention = PntrAttention(self.dec_units)
    
    def call(self, input, enc_output, initial_state=None):
        state_h, state_c = initial_state
        output, state_h, state_c = self.lstm(input, initial_state=[state_h, state_c])
        attention_weights = self.attention(query=output, value=enc_output)
        return attention_weights

class PntrAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = Dense(units, use_bias=False)
        self.W2 = Dense(units, use_bias=False)
        self.attention = AdditiveAttention()
    
    def call(self, query, value):
        w1_query = self.W1(query)
        w2_key = self.W2(value)

        _, attention_weights = self.attention(
            inputs=[w1_query, value, w2_key],
            return_attention_scores=True
        )

        return attention_weights
class PointerNetwork(tf.keras.Model):
    def __init__(self, location_count, units, batch_size=64):
        super().__init__()
        self.encoder = Encoder(units)
        self.decoder = Decoder(units)
        self.batch_size = batch_size
        self.location_count = location_count
    
    def call(self, inputs, training=False):
        enc_input, target_locations = inputs
        enc_output, enc_states = self.encoder(enc_input)
        enc_state_h, enc_state_c = enc_states
        scores = self.decoder(target_locations, enc_output, [enc_state_h, enc_state_c])
        return scores