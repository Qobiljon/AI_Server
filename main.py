import numpy as np
import random
import math
import sys


def load_file(df):
    # load a comma-delimited text file into an np matrix
    result_list = []
    f = open(df, 'r')
    for line in f:
        line = line.rstrip('\n')  # "1.0,2.0,3.0"
        s_vals = line.split(',')  # ["1.0", "2.0, "3.0"]
        f_vals = list(map(np.float32, s_vals))  # [1.0, 2.0, 3.0]
        result_list.append(f_vals)  # [[1.0, 2.0, 3.0] , [4.0, 5.0, 6.0]]
    f.close()
    return np.asarray(result_list, dtype=np.float32)  # not necessary


def show_vector(v, dec):
    fmt = "%." + str(dec) + "f"  # like %.4f
    for i in range(len(v)):
        x = v[i]
        if x >= 0.0:
            print(' ', end='')
        print(fmt % x + '  ', end='')
    print('')


def show_matrix(m, dec):
    fmt = "%." + str(dec) + "f"  # like %.4f
    for i in range(len(m)):
        for j in range(len(m[i])):
            x = m[i, j]
            if x >= 0.0:
                print(' ', end='')
            print(fmt % x + '  ', end='')
        print('')


def show_matrix_partial(m, num_rows, dec, indices):
    fmt = "%." + str(dec) + "f"  # like %.4f
    last_row = len(m) - 1
    width = len(str(last_row))
    for i in range(num_rows):
        if indices:
            print("[", end='')
            print(str(i).rjust(width), end='')
            print("] ", end='')

        for j in range(len(m[i])):
            x = m[i, j]
            if x >= 0.0:
                print(' ', end='')
            print(fmt % x + '  ', end='')
        print('')
    print(" . . . ")

    if indices:
        print("[", end='')
        print(str(last_row).rjust(width), end='')
        print("] ", end='')
    for j in range(len(m[last_row])):
        x = m[last_row, j]
        if x >= 0.0:
            print(' ', end='')
        print(fmt % x + '  ', end='')
    print('')


class NeuralNetwork:
    def __init__(self, num_input, num_hidden, num_output, seed):
        self.ni = num_input
        self.nh = num_hidden
        self.no = num_output

        self.iNodes = np.zeros(shape=[self.ni], dtype=np.float32)
        self.hNodes = np.zeros(shape=[self.nh], dtype=np.float32)
        self.oNodes = np.zeros(shape=[self.no], dtype=np.float32)

        self.ihWeights = np.zeros(shape=[self.ni, self.nh], dtype=np.float32)
        self.hoWeights = np.zeros(shape=[self.nh, self.no], dtype=np.float32)

        self.hBiases = np.zeros(shape=[self.nh], dtype=np.float32)
        self.oBiases = np.zeros(shape=[self.no], dtype=np.float32)

        self.rnd = random.Random(seed)  # allows multiple instances
        self.initialize_weights()

    def set_weights(self, weights):
        if len(weights) != self.total_weights(self.ni, self.nh, self.no):
            print("Warning: len(weights) error in setWeights()")

        idx = 0
        for i in range(self.ni):
            for j in range(self.nh):
                self.ihWeights[i, j] = weights[idx]
                idx += 1

        for j in range(self.nh):
            self.hBiases[j] = weights[idx]
            idx += 1

        for j in range(self.nh):
            for k in range(self.no):
                self.hoWeights[j, k] = weights[idx]
                idx += 1

        for k in range(self.no):
            self.oBiases[k] = weights[idx]
            idx += 1

    def get_weights(self):
        tw = self.total_weights(self.ni, self.nh, self.no)
        result = np.zeros(shape=[tw], dtype=np.float32)
        idx = 0  # points into result

        for i in range(self.ni):
            for j in range(self.nh):
                result[idx] = self.ihWeights[i, j]
                idx += 1

        for j in range(self.nh):
            result[idx] = self.hBiases[j]
            idx += 1

        for j in range(self.nh):
            for k in range(self.no):
                result[idx] = self.hoWeights[j, k]
                idx += 1

        for k in range(self.no):
            result[idx] = self.oBiases[k]
            idx += 1

        return result

    def initialize_weights(self):
        num_wts = self.total_weights(self.ni, self.nh, self.no)
        wts = np.zeros(shape=[num_wts], dtype=np.float32)
        lo = -0.01
        hi = 0.01
        for idx in range(len(wts)):
            wts[idx] = (hi - lo) * self.rnd.random() + lo
        self.set_weights(wts)

    def compute_outputs(self, x_values):
        h_sums = np.zeros(shape=[self.nh], dtype=np.float32)
        o_sums = np.zeros(shape=[self.no], dtype=np.float32)

        for i in range(self.ni):
            self.iNodes[i] = x_values[i]

        for j in range(self.nh):
            for i in range(self.ni):
                h_sums[j] += self.iNodes[i] * self.ihWeights[i, j]

        for j in range(self.nh):
            h_sums[j] += self.hBiases[j]

        for j in range(self.nh):
            self.hNodes[j] = self.hypertan(h_sums[j])

        for k in range(self.no):
            for j in range(self.nh):
                o_sums[k] += self.hNodes[j] * self.hoWeights[j, k]

        for k in range(self.no):
            o_sums[k] += self.oBiases[k]

        soft_out = self.softmax(o_sums)
        for k in range(self.no):
            self.oNodes[k] = soft_out[k]

        result = np.zeros(shape=self.no, dtype=np.float32)
        for k in range(self.no):
            result[k] = self.oNodes[k]

        return result

    def train(self, train_data, max_epochs, learn_rate):
        ho_grads = np.zeros(shape=[self.nh, self.no], dtype=np.float32)  # hidden-to-output weights gradients
        ob_grads = np.zeros(shape=[self.no], dtype=np.float32)  # output node biases gradients
        ih_grads = np.zeros(shape=[self.ni, self.nh], dtype=np.float32)  # input-to-hidden weights gradients
        hb_grads = np.zeros(shape=[self.nh], dtype=np.float32)  # hidden biases gradients

        o_signals = np.zeros(shape=[self.no], dtype=np.float32)  # output signals: gradients w/o assoc. input terms
        h_signals = np.zeros(shape=[self.nh], dtype=np.float32)  # hidden signals: gradients w/o assoc. input terms

        epoch = 0
        x_values = np.zeros(shape=[self.ni], dtype=np.float32)
        t_values = np.zeros(shape=[self.no], dtype=np.float32)
        num_train_items = len(train_data)
        indices = np.arange(num_train_items)  # [0, 1, 2, . . n-1]  # rnd.shuffle(v)

        while epoch < max_epochs:
            self.rnd.shuffle(indices)  # scramble order of training items
            for ii in range(num_train_items):
                idx = indices[ii]

                for j in range(self.ni):
                    x_values[j] = train_data[idx, j]  # get the input values
                for j in range(self.no):
                    t_values[j] = train_data[idx, j + self.ni]  # get the target values
                self.compute_outputs(x_values)  # results stored internally

                # 1. compute output node signals
                for k in range(self.no):
                    derivative = (1 - self.oNodes[k]) * self.oNodes[k]  # softmax
                    o_signals[k] = derivative * (self.oNodes[k] - t_values[k])  # E=(t-o)^2 do E'=(o-t)

                # 2. compute hidden-to-output weight gradients using output signals
                for j in range(self.nh):
                    for k in range(self.no):
                        ho_grads[j, k] = o_signals[k] * self.hNodes[j]

                # 3. compute output node bias gradients using output signals
                for k in range(self.no):
                    ob_grads[k] = o_signals[k] * 1.0  # 1.0 dummy input can be dropped

                # 4. compute hidden node signals
                for j in range(self.nh):
                    _sum = 0.0
                    for k in range(self.no):
                        _sum += o_signals[k] * self.hoWeights[j, k]
                    derivative = (1 - self.hNodes[j]) * (1 + self.hNodes[j])  # tanh activation
                    h_signals[j] = derivative * _sum

                # 5 compute input-to-hidden weight gradients using hidden signals
                for i in range(self.ni):
                    for j in range(self.nh):
                        ih_grads[i, j] = h_signals[j] * self.iNodes[i]

                # 6. compute hidden node bias gradients using hidden signals
                for j in range(self.nh):
                    hb_grads[j] = h_signals[j] * 1.0  # 1.0 dummy input can be dropped

                # update weights and biases using the gradients

                # 1. update input-to-hidden weights
                for i in range(self.ni):
                    for j in range(self.nh):
                        delta = -1.0 * learn_rate * ih_grads[i, j]
                        self.ihWeights[i, j] += delta

                # 2. update hidden node biases
                for j in range(self.nh):
                    delta = -1.0 * learn_rate * hb_grads[j]
                    self.hBiases[j] += delta

                    # 3. update hidden-to-output weights
                for j in range(self.nh):
                    for k in range(self.no):
                        delta = -1.0 * learn_rate * ho_grads[j, k]
                        self.hoWeights[j, k] += delta

                # 4. update output node biases
                for k in range(self.no):
                    delta = -1.0 * learn_rate * ob_grads[k]
                    self.oBiases[k] += delta

            epoch += 1

            if epoch % 10 == 0:
                mse = self.mean_squared_error(train_data)
                print("epoch = " + str(epoch) + " ms error = %0.4f " % mse)
        # end while

        result = self.get_weights()
        return result

    # end train

    def accuracy(self, tdata):  # train or test data matrix
        num_correct = 0
        num_wrong = 0
        x_values = np.zeros(shape=[self.ni], dtype=np.float32)
        t_values = np.zeros(shape=[self.no], dtype=np.float32)

        for i in range(len(tdata)):  # walk thru each data item
            for j in range(self.ni):  # peel off input values from curr data row
                x_values[j] = tdata[i, j]
            for j in range(self.no):  # peel off tareget values from curr data row
                t_values[j] = tdata[i, j + self.ni]

            y_values = self.compute_outputs(x_values)  # computed output values)
            max_index = np.argmax(y_values)  # index of largest output value

            if abs(t_values[max_index] - 1.0) < 1.0e-5:
                num_correct += 1
            else:
                num_wrong += 1

        return (num_correct * 1.0) / (num_correct + num_wrong)

    def mean_squared_error(self, tdata):  # on train or test data matrix
        sum_squared_error = 0.0
        x_values = np.zeros(shape=[self.ni], dtype=np.float32)
        t_values = np.zeros(shape=[self.no], dtype=np.float32)

        for ii in range(len(tdata)):  # walk thru each data item
            for jj in range(self.ni):  # peel off input values from curr data row
                x_values[jj] = tdata[ii, jj]
            for jj in range(self.no):  # peel off tareget values from curr data row
                t_values[jj] = tdata[ii, jj + self.ni]

            y_values = self.compute_outputs(x_values)  # computed output values

            for j in range(self.no):
                err = t_values[j] - y_values[j]
                sum_squared_error += err * err  # (t-o)^2

        return sum_squared_error / len(tdata)

    @staticmethod
    def hypertan(x):
        if x < -20.0:
            return -1.0
        elif x > 20.0:
            return 1.0
        else:
            return math.tanh(x)

    @staticmethod
    def softmax(o_sums):
        result = np.zeros(shape=[len(o_sums)], dtype=np.float32)
        m = max(o_sums)
        divisor = 0.0
        for k in range(len(o_sums)):
            divisor += math.exp(o_sums[k] - m)
        for k in range(len(result)):
            result[k] = math.exp(o_sums[k] - m) / divisor
        return result

    @staticmethod
    def total_weights(n_input, n_hidden, n_output):
        tw = (n_input * n_hidden) + (n_hidden * n_output) + n_hidden + n_output
        return tw


def main():
    print("Back-Propagation ( Python =", str(sys.version)[0:5], "NumPy =", str(np.version.version), ')')

    num_input = 4
    num_hidden = 5
    num_output = 3
    print("Neural Network = %d-%d-%d" % (num_input, num_hidden, num_output))
    nn = NeuralNetwork(num_input=4, num_hidden=5, num_output=3, seed=3)

    train_data_matrix = load_file('trainData.txt')
    # show_matrix_partial(train_data_matrix, 4, 1, True)
    test_data_matrix = load_file('testData.txt')

    print('Training starts')
    nn.train(train_data_matrix, max_epochs=50, learn_rate=0.05)
    print('Training completed')

    acc_train = nn.accuracy(train_data_matrix)
    acc_test = nn.accuracy(test_data_matrix)

    print("Accuracy: 120 train items = %0.4f " % acc_train)
    print("Accuracy:  30 test  items = %0.4f " % acc_test)


if __name__ == "__main__":
    main()
