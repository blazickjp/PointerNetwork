import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, AdditiveAttention

from ShapeChecker import ShapeChecker

class Encoder(tf.keras.layers.Layer):
    def __init__(self, input_size, units):
        super(Encoder, self).__init__()
        self.units = units
        self.input_size = input_size

        self.lstm = keras.layers.LSTM(
            self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
    
    def call(self, coordinates, state=None):
        shape_checker = ShapeChecker()
        shape_checker(coordinates, ('batch', 'coordinates', 'features'))

        output, state_h, state_c = self.lstm(coordinates, initial_state=state)
        shape_checker(output, ('batch', 'coordinates', 'enc_units'))
        shape_checker(state_h, ('batch', 'enc_units'))
        shape_checker(state_c, ('batch', 'enc_units'))

        return output, [state_h, state_c]

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = Dense(units, use_bias=False)
        self.W2 = Dense(units, use_bias=False)
        self.attention = AdditiveAttention()
    
    def call(self, query, value):
        shape_checker = ShapeChecker()
        shape_checker(query, ('batch', 'output_locations', 'query_units'))
        shape_checker(value, ('batch', 'coordinates', 'value_units'))

        w1_query = self.W1(query)
        shape_checker(w1_query, ('batch', 'output_locations', 'attn_units'))

        w2_key = self.W2(value)
        shape_checker(w2_key, ('batch', 'coordiantes', 'attn_units'))

        _, attention_weights = self.attention(
            inputs=[w1_query, value, w2_key],
            return_attention_scores=True
        )
        shape_checker(attention_weights, ('batch', 'output_locations', 'coordinates'))

        return attention_weights

class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_size, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.output_size = output_size
        self.lstm = tf.keras.layers.LSTM(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.attention = SelfAttention(self.dec_units)
    
    def call(self, new_locations, enc_output, state=None):
        shape_checker = ShapeChecker()
        shape_checker(new_locations, ('batch', 'output_locations', 'features'))
        shape_checker(enc_output, ('batch', 'coordinates', 'enc_units'))
        
        if state is not None:
            state_h, state_c = state
            shape_checker(state_h, ('batch', 'dec_units'))
            shape_checker(state_c, ('batch', 'dec_units'))
        
        rnn_output, state_h, state_c = self.lstm(new_locations, initial_state=[state_h, state_c])

        shape_checker(rnn_output, ('batch', 'output_locations', 'dec_units'))
        shape_checker(state_h, ('batch', 'dec_units'))
        shape_checker(state_c, ('batch', 'dec_units'))

        attention_weights = self.attention(query=rnn_output, value=enc_output)
        shape_checker(attention_weights, ('batch', 'output_locations', 'coordinates'))

        return attention_weights, [state_h, state_c]

class PointerNetwork(tf.keras.Model):
    def __init__(self, location_count, units, batch_size=64):
        super().__init__()
        self.encoder = Encoder(location_count * 2, units)
        self.decoder = Decoder(location_count, units)
        self.batch_size = batch_size
        self.location_count = location_count
        self.shape_checker = ShapeChecker()
    
    def train_step(self, inputs):
        self.shape_checker = ShapeChecker()
        input_coordinates, target_locations = inputs

        with tf.GradientTape() as tape:
            enc_output, enc_state = self.encoder(input_coordinates)
            enc_state_h, enc_state_c = enc_state
            self.shape_checker(enc_output, ('batch', 'coordinates', 'enc_units'))
            self.shape_checker(enc_state_h, ('batch', 'enc_units'))
            self.shape_checker(enc_state_c, ('batch', 'enc_units'))

            dec_state = [enc_state_h, enc_state_c]
            loss = tf.constant(0.0)

            for t in tf.range(self.location_count):
                new_locations = target_locations[:, t:t+2]
                step_loss, dec_state = self._loop_step(new_locations, enc_output, dec_state)
                loss += step_loss
            
            average_loss = loss / (self.location_count * self.batch_size)

        variables = self.trainable_variables
        gradients = tape.gradient(average_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return {'batch_loss': average_loss}
    
    def _loop_step(self, new_locations, enc_output, dec_state):
        input_location, target_location = new_locations[:, 0:1], new_locations[:, 1:2]

        dec_result, dec_state = self.decoder(input_location, enc_output, dec_state)
        dec_state_h, dec_state_c = dec_state
        self.shape_checker(dec_result, ('batch', 'output_locations', 'probabilities'))
        self.shape_checker(dec_state_h, ('batch', 'dec_units'))
        self.shape_checker(dec_state_c, ('batch', 'dec_units'))

        y = tf.squeeze(target_location, [2])
        y_pred = dec_result
        step_loss = self.loss(y, y_pred)

        return step_loss, dec_state