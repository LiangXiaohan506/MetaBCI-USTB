# -*- coding: utf-8 -*-
# License: MIT License
"""
Start another process, define a framework for offline modeling and online processing with three functions:
    pre(): for offline modeling;

    consume(): for online prediction;

    post(): for subsequent custom operations.

In the actual usage process, you only need to customize the operations of the above functions.
"""
from typing import Optional, Any
from abc import abstractmethod
import os
import multiprocessing
import queue
from .logger import get_logger
### ==============================添加内容=============================== ###
import socket
### ==============================添加内容=============================== ###
logger = get_logger("worker")


class ProcessWorker(multiprocessing.Process):
    """Online processing.

    author: Lichao Xu

    Created on: 2021-04-01

    update log:
        2022-08-10 by Wei Zhao

    Parameters
    ----------
    timeout: float
        Timer setting.
    name: str
        Custom name for the online processing process.

    Attributes
    ----------
    daemon: bool
    _exit:
        Multiprocess event handling.
    _in_queue: queue
        Data sharing between the online processing process and the main process.

    Tip
    ----
    ..  code-block:: python
        :linenos:
        :emphasize-lines: 2
        :caption: A example using brainflow. worker

        from brainflow. worker import ProcessWorker
        class FeedbackWorker(ProcessWorker):
            def __init__():
                #Initialization

            def pre(self):
                #Off-line modeling

                #Online processing of data flow between stimulus interfaces
                info = StreamInfo(
                    name='meta_feedback',
                    type='Markers',
                    channel_count=1,
                    nominal_srate=0,
                    channel_format='int32',
                    source_id=self.lsl_source_id)
                self.outlet = StreamOutlet(info)
                print('waiting connection...')
                while not self._exit:
                    if self.outlet.wait_for_consumers(1e-3):
                        break
                print('Connected')

            def consume(self, data) :
                #Online processing
                if self.outlet.have_consumers ():
                    self.outlet.push_sample(“online results，list")

            def post(self):
                pass

    """

    def __init__(self, timeout: float = 1e-3, name: Optional[str] = None):
        multiprocessing.Process.__init__(self)
        self.daemon = False
        self._exit = multiprocessing.Event()
        self._in_queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
        self.timeout = timeout
        self.worker_name = name

    def put(self, data):
        """Put the data in the queue

        author: Lichao Xu

        Created on: 2021-04-01

        update log:
            2022-08-10 by Wei Zhao

        Parameters
        ----------
        data: ndarray, shape(n_samples, n_channels+1)
            Single trial of online data.

        """

        logger.info(
            "put samples in worker-{}".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )
        self._in_queue.put(data)

    def run(self):
        """
        author: Lichao Xu

        Created on: 2021-04-01

        update log:
            2022-08-10 by Wei Zhao

        Online processing process:
            ① Customize the `pre()` function to build a model using offline data.

            ② Clear the queue and wait for data retrieval thread in the main process to get data within a fixed time.

            ③ Customize the `consume()` function to process online data and provide feedback.

            ④ Customize the `post()` function to perform subsequent operations.

            ⑤ Wait for the next online label to start the next online processing.

            ⑥ Close the online processing process, clear the queue, and stop online experiments.

        """
        logger.info(
            "start worker-{}".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )
        self._exit.clear()
        logger.info(
            "pre hook executed in worker-{}".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )
        self.pre()
        self.clear_queue()
        while not self._exit.is_set():
            try:
                data = self._in_queue.get(timeout=self.timeout)
                logger.info(
                    "consume samples in worker-{}".format(
                        self.worker_name if self.worker_name else os.getpid()
                    )
                )
                self.consume(data)
            except queue.Empty:
                # if queue is empty, loop to wait for next data until exiting
                pass
        logger.info(
            "post hook executed in worker-{}".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )
        self.post()
        self.clear_queue()
        logger.info(
            "worker{} exit".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )

    @abstractmethod
    def pre(self):
        """Custom function to build a model using offline data.

        author: Lichao Xu

        Created on: 2021-04-01

        update log:
            2022-08-10 by Wei Zhao

        """
        pass

    @abstractmethod
    def consume(self, data):
        """Custom function to process online data.

        author: Lichao Xu

        Created on: 2021-04-01

        update log:
            2022-08-10 by Wei Zhao

        Parameters
        ----------
        data: ndarray, shape(n_samples, n_channels+1)
            Single trial of online data.

        """
        pass

    @abstractmethod
    def post(self):
        pass

    def stop(self):
        """Stop the online processing process.

        author: Lichao Xu

        Created on: 2021-04-01

        update log:
            2022-08-10 by Wei Zhao

        """
        logger.info(
            "stop worker-{}".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )
        self._exit.set()

    def settimeout(self, timeout=0.01):
        """Set the timer.

        author: Lichao Xu

        Created on: 2021-04-01

        update log:
            2022-08-10 by Wei Zhao

        """
        self.timeout = timeout

    def clear_queue(self):
        """Clear the queue.

        author: Lichao Xu

        Created on: 2021-04-01

        update log:
            2022-08-10 by Wei Zhao

        """
        logger.info(
            "clearing queue items in worker-{}".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )
        while True:
            try:
                self._in_queue.get(timeout=self.timeout)
            except queue.Empty:
                break
        logger.info(
            "all queue items in worker-{} are cleared".format(
                self.worker_name if self.worker_name else os.getpid()
            )
        )

### ==============================添加内容=============================== ###
def command_output(SOCKET_HOST, SOCKET_PORT):
    """Socket 接收命令输出

    author: Guangjin Liang <3330635482@qq.com>

    Created on: 2025-06-11

    update log:
        2025-07-1 by Guangjin Liang <3330635482@qq.com>: Initial implementation.
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server_socket.bind((SOCKET_HOST, SOCKET_PORT))
        server_socket.listen(1)
        print(f"Server listening on {SOCKET_HOST}:{SOCKET_PORT}")
        client_socket, address = server_socket.accept()
        print(f"Connected by {address}")
        return server_socket, client_socket
    except Exception as e:
        raise RuntimeError(f"Socket 连接失败: {e}")
### ==============================添加内容=============================== ###
