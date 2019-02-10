from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import tensorflow as tf
import numpy as np


class DNN(object):

    def __init__(self, state_size):
        self.learning_rate = 0.0001
        self.activation = 'tanh'
        self.state_size = state_size
        self.model = self._make_model()

    def _huber_loss(self, y_true, y_pred, clip_delta=2.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta
        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _make_model(self):
        model = Sequential()
        model.add(Dense(200, input_dim=2*self.state_size, activation=self.activation))
        model.add(Dense(200, activation=self.activation))
        model.add(Dropout(0.3))
        model.add(Dense(200, activation=self.activation))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='linear'))
        model.compile(loss=self._huber_loss, optimizer=Adam(lr=self.learning_rate))
        return model

    def train(self, X, y):
        hist = self.model.fit(X, y, epochs=10,  verbose=0, shuffle=True)
        return hist

    def predict_Q(self, state, state_action):
        state = state.reshape(self.state_size)
        state_action = state_action.reshape(self.state_size)
        #print(np.array([np.hstack((state, state_action))]))
        return self.model.predict(np.array([np.hstack((state, state_action))]))[0][0]
