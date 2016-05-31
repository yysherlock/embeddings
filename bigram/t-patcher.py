import threading
import queue

opt = {'learning_rate':0.1, \
    'weight_decay': 1e-3, \
    'tolerance': 1.0, \
    'batch_size': 100, \
    'maxecho': 2000, \
    'lambda': 1e-3 }
processor = Processor('bi-config.ini','DEBUG')

class Worker(threading.Thread):
    tot = 0
    tot_lock = threading.Lock()

    def __init__(self, worker_queue):
        super().__init__()
        self.worker_queue = worker_queue

    def run(self):
        inc = 0
        while True:
            item = self.worker_queue.get()
            inc = sum(item)
            with Worker.tot_lock:
                Worker.tot += inc
            #print(Worker.tot)
            self.worker_queue.task_done()

class Model(object):
    def __init__(self, worker_queue, batch_size, thread_count, state):
        self.state = state
        self.thread_count = thread_count
        self.batch_size = batch_size
        self.worker_queue = worker_queue
        for i in range(self.thread_count):
            worker = Worker(self.worker_queue)
            worker.daemon = True
            worker.start()

    def train(self):
        data_list = [i+1 for i in range(1000)]
        data_size = len(data_list)
        start = 0
        iteration = 0
        while iteration < 10:
            expectend = start + self.batch_size
            if expectend <= data_size: batch_list = data_list[start : expectend]
            else: batch_list = data_list[start : data_size] + data_list[0 : expectend - data_size]

            self.mini_batch(batch_list)

            start = expectend % data_size
            iteration += 1

    def mini_batch(self, batch_list):
        Worker.tot = self.state
        start,num = 0, int(len(batch_list) / self.thread_count)

        while start < len(batch_list):
            self.worker_queue.put(batch_list[start:start+num])
            start += num
        #print('size:',self.worker_queue.qsize())
        self.worker_queue.join()
        self.state = Worker.tot

worker_queue = queue.Queue()
model = Model(worker_queue, 100, 2, 0)
model.train()
print(model.state)
