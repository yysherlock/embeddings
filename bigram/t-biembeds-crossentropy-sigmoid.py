from __future__ import division
import numpy as np
import configparser
import pickle
import threading
import queue
import time
from processor import Processor
from six.moves import xrange
from copy import deepcopy

"""
1. check_thread: check correctness of gradients calculation
2. despatch_thread: despatch batches, main thread
3. workers: work thread, creating by Worker()
"""

processor = Processor('bi-config.ini','COPA')

########### Util functions #############
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def softmax(x): # x is np array, N x v
    return (np.exp(x.T) / np.sum(np.exp(x.T), axis = 0)).T # due to broadcast rule, we use tranverse .T

def feedforward(a0, weight, bias):
    """ a0 -- z1 -- a1, return a1 """
    a1 = sigmoid(np.dot(weight, a0) + bias) # h x v, v x 1
    return a1 # h x 1

def feedforward_all(a0, weight, bias):
    """ a0 -- z1 -- a1, return a1, for N examples """
    a1 = sigmoid(np.dot(weight, a0.T) + bias) # h x N
    return a1.T # N x h

def add_list_org(l1, l2, sz):
    return [l1[i] + l2[i]/sz for i in range(len(l1))]

def add_list(l1,l2):
    return [l1[i]+l2[i] for i in xrange(len(l1))]

def idxmapping(idx, rowsize, colsize):
    assert idx < rowsize*colsize, "idx error, out of index"
    row = idx / colsize
    col = idx % colsize
    return row,col

class Opt(object):
    learning_rate = 0.1
    L = 2
    weight_decay = 1e-3
    tolerance = 0.2
    batch_size = 50000
    maxecho = 2000
    lambda_ = 1e-3
    thread_count = 1

class Worker(threading.Thread):

    weights, biases = None, None
    wgradients, bgradients = None, None
    gradients_lock = threading.Lock()

    def __init__(self, worker_queue):
        super().__init__() # remember this!!
        self.worker_queue = worker_queue

    def run(self):
        while True:
            idx_list = self.worker_queue.get()
            patch_data, patch_label = processor.transform_data(idx_list,'nec')
            sz = patch_label.shape[0]

            inc_wgradients = [np.zeros(weight.shape) for weight in self.weights]
            inc_bgradients = [np.zeros(bias.shape) for bias in self.biases]
            for i in xrange(sz):
                data = patch_data[i][np.newaxis].T # ith row, v x 1
                label = patch_label[i][np.newaxis].T
                ws, bs = self.applygradient(data, label)
                for idx in xrange(Opt.L): inc_wgradients[idx] += ws[idx]
                for idx in xrange(Opt.L): inc_bgradients[idx] += bs[idx]

            with Worker.gradients_lock:
                # partial update avg_wgradients, avg_bgradients in this batch
                for idx in xrange(Opt.L): Worker.wgradients[idx] += inc_wgradients[idx]
                for idx in xrange(Opt.L): Worker.bgradients[idx] += inc_bgradients[idx]

            self.worker_queue.task_done() # remember this!

    def applygradient(self, train_data, train_label):
        """ calculate wgradients, bgradients for all layers (L-1 ~ 0)
        d: represents dJ/dz at lth layer
        At current layer:
            a0 -- z1 -- a1
        """
        wgradients = []
        bgradients = []
        # generate outputs for each layer, i.e. ai for each layer i
        a0 = train_data # (v,1)
        outputs = [a0] # output[i] represents ai, i range from 0 to L
        for idx in xrange(Opt.L):
            weight = Worker.weights[idx] # weight: h x v
            bias = Worker.biases[idx] # bias: h x 1
            a0 = feedforward(a0, weight, bias) # h x 1
            outputs.append(a0)

        # d: h x 1
        # d for quadratic
        # d = outputs[Opt.L] * (1-outputs[Opt.L]) * (outputs[Opt.L] - train_label) # at layer L, initial, 10 x 1
        # d for cross entropy
        d = outputs[Opt.L] - train_label

        for l in xrange(Opt.L-1,-1,-1): # L-1 ~ 0
            # calculate gradient at layer l, l range from L-1 to 0
            # at layer l, current d is at l+1
            a0 = outputs[l]
            a1 = outputs[l+1]
            wgradient = np.dot(d,a0.T) \
                + Opt.lambda_*Worker.weights[l]# h x 1,1 x v -> h x v
            bgradient = d # h x 1
            wgradients = [wgradient] + wgradients
            bgradients = [bgradient] + bgradients

            # update d for next layer, i.e. layer l-1
            # quadratic update and cross entropy update are same
            d = np.dot(self.weights[l].T,d) * (a0*(1-a0)) # d at layer l, here * is elementwise product

        return (wgradients, bgradients)

class BigramEmbed(object):
    def __init__(self, architecture, processor, worker_queue): # with one hidden layer

        self.processor = processor
        self.L = len(architecture) - 1
        self.architecture = architecture
        # initial weights and biases
        self.weights = []
        self.biases = []
        for idx in xrange(len(architecture)-1):
            # follows N(0,1) distribution: np.random.randn(shape)
            # follows N(miu, sigma^2) distribution: sigma * np.random.randn(shape) + miu
            self.weights.append( 0.1 * np.random.randn( architecture[idx+1], architecture[idx] ) ) # L x H x V (append L times: V x H)
            self.biases.append( 0.1 * np.random.randn( architecture[idx+1], 1 ) ) # L x H (append L times: H x 1)

        self.batch_size = Opt.batch_size
        self.tolerance = Opt.tolerance
        self.learning_rate = Opt.learning_rate
        self.maxecho = Opt.maxecho
        self.lambda_ = Opt.lambda_
        self.worker_queue = worker_queue

        for i in xrange(Opt.thread_count):
            worker = Worker(self.worker_queue)
            worker.daemon = True
            worker.start()

    def train(self):
        """ SVD for training """
        converge = False
        iteration = 0
        start = 0
        train_idx_list = self.processor.dataidxs
        data_size = len(train_idx_list)

        while not converge and iteration < self.maxecho:
            expectend = start + self.batch_size
            if expectend <= data_size: batch_idx_list = train_idx_list[start : expectend]
            else: batch_idx_list = train_idx_list[start : data_size] + train_idx_list[0 : expectend - data_size]

            wgradients, bgradients = self.mini_batch(batch_idx_list)
            wgradients_magnitude = np.array([ np.linalg.norm(wgradient) for wgradient in wgradients ])
            bgradients_magnitude = np.array([ np.linalg.norm(bgradient) for bgradient in bgradients ])

            self.update(wgradients, bgradients)

            start = expectend % data_size
            # batch_data: context vectors, batch_label: target vectors


            if np.sum(wgradients_magnitude) < self.tolerance and np.sum(bgradients_magnitude) < self.tolerance:
                converge = True

            iteration += 1
            print('magnitude:',np.sum(wgradients_magnitude))
            print('iteration: ',iteration)

        #print('weights:',self.weights,'biases:',self.biases)

    def mini_batch(self, batch_idx_list):
        # set weights and biases for Worker class
        Worker.weights = deepcopy(self.weights)
        Worker.biases = deepcopy(self.biases)
        Worker.wgradients = [np.zeros(weight.shape) for weight in self.weights]
        Worker.bgradients = [np.zeros(bias.shape) for bias in self.biases]
        sz = len(batch_idx_list)
        start,num = 0,int(len(batch_idx_list) / Opt.thread_count)
        while start < len(batch_idx_list):
            self.worker_queue.put(batch_idx_list[start : start + num])
            start += num
        self.worker_queue.join()

        """
        # gradient_check
        layer = 1
        sampled_data, sampled_label = self.processor.transform_data(batch_idx_list[:10],'nec')
        for i in range(10):
            wgradients, bgradients = self.applygradient(sampled_data[i][np.newaxis].T, sampled_label[i][np.newaxis].T)
            self.gradient_check(sampled_data[i][np.newaxis].T, sampled_label[i][np.newaxis].T, wgradients[layer], bgradients[layer], layer)
        """
        # update self.weights and self.biases
        avg_wgradients = [wgradient/sz for wgradient in Worker.wgradients]
        avg_bgradients = [bgradient/sz for bgradient in Worker.bgradients]
        #self.update(avg_wgradients, avg_bgradients)

        return (avg_wgradients, avg_bgradients)


    def applygradient(self, train_data, train_label):
        """ calculate wgradients, bgradients for all layers (L-1 ~ 0)
        d: represents dJ/dz at lth layer
        At current layer:
            a0 -- z1 -- a1
        """
        wgradients = []
        bgradients = []
        # generate outputs for each layer, i.e. ai for each layer i
        a0 = train_data # (784,1)
        outputs = [a0] # output[i] represents ai, i range from 0 to L
        for idx in xrange(self.L):
            weight = self.weights[idx] # weight: h x v
            bias = self.biases[idx] # bias: h x 1
            a0 = feedforward(a0, weight, bias) # h x 1
            outputs.append(a0)

        # d: h x 1
        # d = - outputs[self.L] * (1-outputs[self.L]) * (train_label-outputs[self.L]) # at layer L, initial, 10 x 1
        d = outputs[self.L] - train_label

        for l in xrange(self.L-1,-1,-1): # L-1 ~ 0
            # calculate gradient at layer l, l range from L-1 to 0
            # at layer l, current d is at l+1
            a0 = outputs[l]
            a1 = outputs[l+1]
            wgradient = np.dot(d,a0.T) + self.lambda_*self.weights[l]# h x 1,1 x v -> h x v
            bgradient = d # h x 1
            wgradients = [wgradient] + wgradients
            bgradients = [bgradient] + bgradients

            # update d for next layer, i.e. layer l-1
            d = np.dot(self.weights[l].T,d) * (a0*(1-a0)) # d at layer l
            # v x h, h x 1 -> v x 1 * v x 1 -> v x 1

        return (wgradients, bgradients)

    def update(self, wgradients, bgradients, iteration = None):
        if iteration:
            pass # decay learning rate
        for idx in xrange(self.L):
            self.weights[idx] -= self.learning_rate * wgradients[idx]
            self.biases[idx] -= self.learning_rate * bgradients[idx]

    def gradient_check(self, sampled_data, sampled_label, wgradient, bgradient, layer = 1):
        # gradient check
        numeric_wgradient, wsample = self.computeNumericGradient(sampled_data.T, sampled_label.T, self.weights[layer])
        numeric_bgradient, bsample = self.computeNumericGradient(sampled_data.T, sampled_label.T, self.biases[layer])
        for x,ws in enumerate(wsample):
            index = idxmapping(ws, *self.weights[layer].shape)
            close = np.isclose(wgradient[index], numeric_wgradient[x])
            diffw = np.linalg.norm(wgradient[index] - numeric_wgradient[x])
            print('diffw:', diffw, close)

        for x,bs in enumerate(bsample):
            index = idxmapping(bs, *self.biases[layer].shape)
            close = np.isclose(bgradient[index], numeric_bgradient[x])
            diffb = np.linalg.norm(bgradient[index] - numeric_bgradient[x])
            print('diffb:', diffb, close)

    def computeNumericGradient(self, input, label, theta, epsilon=1e-4, sampleNum = 10):
        """ theta: w or b
        """
        h,v = theta.shape
        sample = np.random.randint(0, h*v, sampleNum)
        grad = np.zeros(sampleNum)

        for i,idx in enumerate(sample):
            # change theta
            theta[idxmapping(idx,h,v)] += epsilon
            c1 = self.getCost(input, label)
            theta[idxmapping(idx,h,v)] -= 2*epsilon
            c2 = self.getCost(input, label)
            grad[i] = (c1 - c2) / (2*epsilon)
            theta[idxmapping(idx,h,v)] += epsilon

        return grad, sample

    def cost(self, output, label):
        # When use quadratic cost function
        # return 0.5 * np.sum((output - label)**2)

        # When top layer is softmax layer:
        # return np.sum(np.nan_to_num(-np.log(output) * label))

        # When top layer is sigmoid layer
        return np.sum(-label*np.log(output) - (1-label)*np.log(1-output)) \
        + 0.5 * self.lambda_ * sum([np.sum(weight**2) for weight in self.weights])
        # influence: 1) dJ/dW, J has an additional term lambda/2 * (||W1||+||W2||+...+||WL||)
                    # we should add a term on its gradient dJ/dW, for Wl, its:
                    # lambda * Wl
                    #2)
    def predict_output(self, input_data):
        """ Todo: need to add softmax layer later"""
        for idx in xrange(len(self.weights)):
            weight = self.weights[idx] # weight: h x v
            bias = self.biases[idx] # bias: h x 1
            input_data = feedforward_all(input_data, weight, bias) # N x h
        output = input_data
        return output

    def getCost(self, input_data, label): # for a mini batch
        output = self.predict_output(input_data) # output: N x v
        return self.cost(output, label)

    def predict_target(self,input_data): # N x v, N examples and V input/visual units
        output = self.predict_output(input_data) # N x v
        return np.argmax(output, axis = 1).T #  1 x N

    def get_embeddings(self,fn):
        with open(fn,'wb') as f:
            obj = []
            for i in range(self.processor.volcab_size):
                # wordid: i
                input_vec = np.zeros([self.processor.volcab_size,1])
                input_vec[i,0] = 1.
                embedding = feedforward(input_vec, self.weights[0], self.biases[0]) # embeddings
                obj.append(embedding)

            pickle.dump(obj,f)

DEBUG = True

if DEBUG:

    v = processor.volcab_size
    worker_queue = queue.Queue()

    bimodel = BigramEmbed([v, 50, v], processor, worker_queue)

    bimodel.train()

    bimodel.get_embeddings(processor.biembeddingsfn)
