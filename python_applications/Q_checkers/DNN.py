from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import numpy as np


class DNN(object):

    def __init__(self, state_size):
        self.learning_rate = 0.001
        self.activation = 'relu'
        self.model = self._make_model()
        self.state_size = state_size

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta
        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _make_model(self):
        model = Sequential()
        model.add(Dense(100, input_dim=self.state_size+4, activation=self.activation))
        model.add(Dense(100, activation=self.activation))
        model.add(Dense(100, activation=self.activation))
        model.add(Dense(1, activation='linear'))
        model.compile(loss=self._huber_loss, optimizer=Adam(lr=self.learning_rate))
        return model

    def train(self, X, y):
        loss = self.model.fit(X, y, nb_epoch=1,  verbose=0, shuffle=True)
        return loss

    def predict_Q(self, state, action):
        state = state.reshape((1, self.state_size))
        return self.model.predict(np.hstack((state,action)))[0]

    def best_action(self, state, actions):
        q_preds = []
        for action in actions:
            q_preds.append(self.predict_Q(state, action))

        return np.argmax(q_preds)