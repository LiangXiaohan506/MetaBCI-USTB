# -*- coding: utf-8 -*-
"""
Amplifiers.

"""
import socket
import struct
import threading
import time
from abc import abstractmethod
from collections import deque
from typing import List, Optional, Tuple, Dict, Any

import numpy
import numpy as np
import pylsl
import queue

### ==============================添加内容=============================== ###
import os
import enum
import json
import ctypes
import platform
import pkg_resources
from numpy.ctypeslib import ndpointer
### ==============================添加内容=============================== ###

from .logger import get_logger
from .workers import ProcessWorker

logger_amp = get_logger("amplifier")
logger_marker = get_logger("marker")


class RingBuffer(deque):
    """Online data RingBuffer.
    -author: Lichao Xu
    -Created on: 2021-04-01
    -update log:
        None
    Parameters
    ----------
        size: int,
            Size of the RingBuffer.
    """

    def __init__(self, size=1024, segment=None):
        """Ring buffer object based on python deque data
        structure to store data.

        Parameters
        ----------
        size : int, optional
            maximum buffer size, by default 1024
        """
        super(RingBuffer, self).__init__(maxlen=size)
        self.max_size = size
        self.segment = segment

    def isfull(self):
        """Whether current buffer is full or not.

        Returns
        ----------
        boolean
        """
        return len(self) == self.max_size

    def get_all(self):
        """Access all current buffer value.

        Returns
        ----------
        list
            the list of current buffer
        """
        return list(self)


class Marker(RingBuffer):
    """Intercept online data.
    -author: Lichao Xu
    -Created on: 2021-04-01
    -update log:
        2022-08-10 by Wei Zhao
    -update log:
        2024-09-01 by Duan Shunguo<dsg@tju.edu.cn>
    Parameters
    ----------
        interval: list,
            Time Window.
        srate: int,
            Amplifier setting sample rate.
        events: list,
            Event label.
        patch_size: int,
            Online data patch delivered everytime
    """

    def __init__(
        self, interval: list, srate: float, events: Optional[List[int]] = None,
        patch_size: Optional[int] = None
    ):
        self.events = events
        if events is not None:
            self.interval = [int(i * srate) for i in interval]
            self.latency = 0 if self.interval[1] <= 0 else self.interval[1]
            max_tlim = max(0, self.interval[0], self.interval[1])
            min_tlim = min(0, self.interval[0], self.interval[1])
            size = max_tlim - min_tlim
            if min_tlim >= 0:
                self.epoch_ind = [self.interval[0], self.interval[1]]
            else:
                self.epoch_ind = [
                    self.interval[0] - min_tlim,
                    self.interval[1] - min_tlim,
                ]
        else:
            # continous mode
            self.interval = [int(i * srate) for i in interval]
            self.latency = self.interval[1] - self.interval[0]
            size = self.latency
            self.epoch_ind = [0, size]

        self.patch_size = patch_size
        self.threshold = (
            self.epoch_ind[1] - self.epoch_ind[0]
            if patch_size is not None
            else 0
        )
        self.threshold_ind = (
            self.epoch_ind[1] - patch_size
            if patch_size is not None
            else 0
        )

        self.countdowns: Dict[str, int] = {}
        self.is_rising = True
        super().__init__(size=size)

    def __call__(self, event: int):
        """Record label position.
        Parameters
        ----------
            event: int,
                Online real-time data tagging.
        """
        # add new countdown items
        if self.events is not None:
            event = int(event)
            if event != 0 and self.is_rising:
                if event in self.events:
                    # new_key = hashlib.md5(''.join(
                    # [str(event), str(datetime.datetime.now())])
                    # .encode()).hexdigest()
                    new_key = "".join(
                        [
                            str(event),
                            # datetime.datetime.now().strftime("%Y-%m-%d \
                            #     -%H-%M-%S"),
                        ]
                    )
                    self.countdowns[new_key] = self.latency + 1
                    logger_marker.info("find new event {}".format(new_key))
                self.is_rising = False
            elif event == 0:
                self.is_rising = True
        else:
            if "fixed" not in self.countdowns:
                self.countdowns["fixed"] = self.latency

        drop_items = []
        # update countdowns
        for key, value in self.countdowns.items():
            value = value - 1
            if isinstance(self.patch_size, int) and 0 < value < self.threshold:
                if value % self.patch_size == 0:
                    self.countdowns[key] = value
                    return True
            if value == 0:
                drop_items.append(key)
                logger_marker.info("trigger epoch for event {}".format(key))
            self.countdowns[key] = value

        for key in drop_items:
            del self.countdowns[key]
        if drop_items and self.isfull():
            return True
        return False

    def get_epoch(self):
        """
        Fetch data from buffer.
        If the self.patch_size is not None, the data will be instantly sent even though buffer is not full.
        """
        data = super().get_all()
        if isinstance(self.patch_size, int) and self.threshold_ind > 0:
            return data[self.threshold_ind: self.epoch_ind[1]]
        return data[self.epoch_ind[0]: self.epoch_ind[1]]


class BaseAmplifier:
    """Base Ampifier class.
    -author: Lichao Xu
    -Created on: 2021-04-01
    -update log:
        2022-08-10 by Wei Zhao
    """

    def __init__(self):
        self._markers = {}
        self._workers = {}
        self._exit = threading.Event()

    @abstractmethod
    def recv(self):
        """the minimal recv data function, usually a package."""
        pass

    def start(self):
        """start the loop."""
        for work_name in self._workers:
            logger_amp.info("clear marker buffer")
            self._markers[work_name].clear()
        logger_amp.info("start the loop")
        self._t_loop = threading.Thread(target=self._inner_loop,
                                        name="main_loop")
        self._t_loop.start()

    def _inner_loop(self):
        """Inner loop in the threading."""
        self._exit.clear()
        logger_amp.info("enter the inner loop")
        while not self._exit.is_set():
            try:
                samples = self.recv()
                if samples:
                    self._detect_event(samples)
            except Exception:
                pass
        logger_amp.info("exit the inner loop")

    def stop(self):
        """stop the loop."""
        logger_amp.info("stop the loop")
        self._exit.set()
        logger_amp.info("waiting the child thread exit")
        self._t_loop.join()
        self.clear()

    def _detect_event(self, samples):
        """detect event label"""
        for work_name in self._workers:
            logger_amp.info("process worker-{}".format(work_name))
            marker = self._markers[work_name]
            worker = self._workers[work_name]
            for sample in samples:
                marker.append(sample)
                if marker(sample[-1]) and worker.is_alive():
                    worker.put(marker.get_epoch())

    def up_worker(self, name):
        logger_amp.info("up worker-{}".format(name))
        self._workers['feedback_worker'].start()

    def down_worker(self, name):
        logger_amp.info("down worker-{}".format(name))
        self._workers[name].stop()
        self._workers[name].clear_queue()

    def register_worker(self, name: str,
                        worker: ProcessWorker,
                        marker: Marker):
        logger_amp.info("register worker-{}".format(name))
        self._workers[name] = worker
        self._markers[name] = marker

    def unregister_worker(self, name: str):
        logger_amp.info("unregister worker-{}".format(name))
        del self._markers[name]
        del self._workers[name]

    def clear(self):
        logger_amp.info("clear all workers")
        worker_names = list(self._workers.keys())
        for name in worker_names:
            self._markers[name].clear()
            self.down_worker(name)
            self.unregister_worker(name)


class NeuroScan(BaseAmplifier):
    """An amplifier implementation for NeuroScan device.
    Intercept online data.
    -author: Lichao Xu
    -Created on: 2021-04-01
    -update log:
        2022-08-10 by Wei Zhao
    """

    _COMMANDS = {
        "stop_connect": b"CTRL\x00\x01\x00\x02\x00\x00\x00\x00",
        "start_acq": b"CTRL\x00\x02\x00\x01\x00\x00\x00\x00",
        "stop_acq": b"CTRL\x00\x02\x00\x02\x00\x00\x00\x00",
        "start_trans": b"CTRL\x00\x03\x00\x03\x00\x00\x00\x00",
        "stop_trans": b"CTRL\x00\x03\x00\x04\x00\x00\x00\x00",
        "show_ver": b"CTRL\x00\x01\x00\x01\x00\x00\x00\x00",
        "show_edf": b"CTRL\x00\x03\x00\x01\x00\x00\x00\x00",
        "start_imp": b"CTRL\x00\x02\x00\x03\x00\x00\x00\x00",
        "req_version": b"CTRL\x00\x01\x00\x01\x00\x00\x00\x00",
        "correct_dc": b"CTRL\x00\x02\x00\x05\x00\x00\x00\x00",
        "change_setup": b"CTRL\x00\x02\x00\x04\x00\x00\x00\x00",
    }

    def __init__(
        self,
        device_address: Tuple[str, int] = ("127.0.0.1", 4000),
        srate: float = 1000,
        num_chans: int = 68,
    ):
        super().__init__()
        self.device_address = device_address
        self.srate = srate
        self.num_chans = num_chans
        self.neuro_link = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # the size of a package in neuroscan data is
        # srate/25*(num_chans+1)*4 bytes
        self.pkg_size = srate / 25 * (num_chans + 1) * 4
        self.timeout = 2 * 25 / self.srate

    def _unpack_header(self, b_header):
        ch_id = struct.unpack(">4s", b_header[:4])
        w_code = struct.unpack(">H", b_header[4:6])
        w_request = struct.unpack(">H", b_header[6:8])
        pkg_size = struct.unpack(">I", b_header[8:])
        return (ch_id[0].decode("utf-8"), w_code[0], w_request[0], pkg_size[0])

    def _unpack_data(self, num_chans, b_data):
        fmt = ">" + str((num_chans + 1) * 4) + "B"
        samples = (
            np.array(list(struct.iter_unpack(fmt, b_data)), dtype=np.uint8)
            .view(np.int32)
            .astype(np.float64)
        )
        samples[:, -1] = samples[:, -1] - 65280
        samples[:, :-1] = samples[:, :-1] * 0.0298 * 1e-6
        return samples.tolist()

    def _recv(self, num_bytes):
        fragments = []
        b_count = 0
        while b_count < num_bytes:
            try:
                chunk = self.neuro_link.recv(num_bytes - b_count)
            except socket.timeout as e:
                raise e
            b_count += len(chunk)
            fragments.append(chunk)

        b_data = b"".join(fragments)
        return b_data

    def recv(self):
        b_header = self._recv(12)
        header = self._unpack_header(b_header)
        samples = None
        if header[-1] != 0:
            b_data = self._recv(header[-1])
            samples = self._unpack_data(self.num_chans, b_data)
        return samples

    def send(self, message):
        self.neuro_link.sendall(message)

    def set_timeout(self, timeout):
        if self.neuro_link:
            self.neuro_link.settimeout(timeout)

    def command(self, method):
        if method == "connect":
            self.neuro_link.connect(self.device_address)
        elif method == "start_acquire":
            self.send(self._COMMANDS["start_acq"])
            self.set_timeout(None)
            self.recv()
            self.recv()
            self.set_timeout(self.timeout)
        elif method == "stop_acquire":
            self.set_timeout(None)
            self.send(self._COMMANDS["stop_acq"])
            self.recv()
            self.recv()
            self.set_timeout(self.timeout)
        elif method == "start_transport":
            self.send(self._COMMANDS["start_trans"])
            time.sleep(1e-2)
            self.start()
        elif method == "stop_transport":
            self.send(self._COMMANDS["stop_trans"])
            self.stop()
        elif method == "disconnect":
            self.send(self._COMMANDS["stop_connect"])
            if self.neuro_link:
                self.neuro_link.close()
                self.neuro_link = None

    def connect_tcp(self):
        self.neuro_link.connect(self.device_address)

    def start_acq(self):
        self.send(self._COMMANDS["start_acq"])
        self.set_timeout(None)
        self.recv()
        self.recv()
        self.set_timeout(self.timeout)

    def stop_acq(self):
        self.set_timeout(None)
        self.send(self._COMMANDS["stop_acq"])
        self.recv()
        self.recv()
        self.set_timeout(self.timeout)

    def start_trans(self):
        self.send(self._COMMANDS["start_trans"])
        time.sleep(1e-2)
        self.start()

    def stop_trans(self):
        self.send(self._COMMANDS["stop_trans"])
        self.stop()

    def close_connection(self):
        self.send(self._COMMANDS["stop_connect"])
        if self.neuro_link:
            self.neuro_link.close()
            self.neuro_link = None


class Curry8(BaseAmplifier):
    """An amplifier implementation for Curry8.
    Intercept online data.
    -author: Ziyu Zhou
    -Created on: 2023-07-07
    """

    def __init__(
            self,
            device_address: Tuple[str, int] = ("127.0.0.1", 4000),
            srate: float = 1000,
            num_chans: int = 68,
    ):
        super().__init__()
        self.device_address = device_address
        self.srate = srate
        self.num_chans = num_chans
        self.neuro_link = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # the size of a package in neuroscan data is
        # srate/25*(num_chans+1)*4 bytes
        self.pkg_size = srate / 25 * (num_chans + 1) * 4
        self.timeout = 2 * 25 / self.srate

    def _unpack_header(self, b_header):
        ch_id = b_header[:4].decode()
        w_code = struct.unpack(">H", b_header[4:6])
        w_request = struct.unpack(">H", b_header[6:8])
        startSample = struct.unpack(">I", b_header[8:12])
        pkg_size = struct.unpack(">I", b_header[12:16])
        return (ch_id, w_code[0], w_request[0], startSample[0], pkg_size[0])

    def _unpack_data(self, num_chans, b_data):
        samples = np.frombuffer(b_data,
                                dtype=np.float32).reshape(-1,
                                                          num_chans).astype(np.float64)
        samples[:, -1] = samples[:, -1] - 65280
        return samples

    def _recv(self, num_bytes):
        fragments = b""
        b_count = 0
        while b_count < num_bytes:
            try:
                chunk = self.neuro_link.recv(num_bytes - b_count)
            except socket.timeout as e:
                raise e
            b_count += len(chunk)
            fragments += chunk

        b_data = fragments
        return b_data

    def recv(self):
        b_header = self._recv(20)
        header = self._unpack_header(b_header)
        if header[-1] != 0:
            b_data = self._recv(header[-1])
            if header[0] == "DATA":
                if header[1] == self.dataType(
                        "Data_Eeg") and header[2] == self.blockType("DataTypeFloat32bit"):
                    samples = self._unpack_data(self.num_chans, b_data)
                    return samples.tolist()
        return []

    def send(self, message):
        self.neuro_link.sendall(message)

    def set_timeout(self, timeout):
        if self.neuro_link:
            self.neuro_link.settimeout(timeout)

    def connect_tcp(self):
        self.neuro_link.connect(self.device_address)

    def start_acq(self):
        self.send(self.command_code("RequestAmpConnect"))
        self.set_timeout(None)

        b_header = self._recv(20)
        header = self._unpack_header(b_header)
        print("start_acq", header)

        self.set_timeout(self.timeout)

    def stop_acq(self):
        self.set_timeout(None)
        self.send(self.command_code("RequestAmpDisonnect"))

        b_header = self._recv(20)
        header = self._unpack_header(b_header)
        print("stop_acq", header)

        self.set_timeout(self.timeout)

    def start_trans(self):  # send data
        self.send(self.command_code("RequestStreamingStart"))
        time.sleep(1e-2)

        b_header = self._recv(20)
        header = self._unpack_header(b_header)
        print("start_trans", header)

        self.start()

    def stop_trans(self):
        self.send(self.command_code("RequestStreamingStop"))
        self.stop()

    def close_connection(self):
        if self.neuro_link:
            self.neuro_link.close()
            self.neuro_link = None

    def update_basic_info(self):
        status, basicInfo, header = self.getBasicInfo()
        if status:
            self.srate = basicInfo["srate"]
            self.num_chans = basicInfo["num_chans"]
            self.basicInfo = basicInfo
            return True
        else:
            return False

    def update_channel_info(self):
        status, channelInfo, header = self.getChannelInfoList()
        if status:
            self.chanelNameList = [x["chanLabel"] for x in channelInfo]
            self.channelInfo = channelInfo
            return True
        else:
            return False

    def getBasicInfo(self):
        maxChans = 300

        # sendCommand
        self.send(self.command_code('RequestBasicInfoAcq'))

        b_header = self._recv(20)
        header = self._unpack_header(b_header)

        if header[0] != 'DATA' \
                or header[1] != self.dataType("Data_Info") \
                or header[2] != self.infoType("InfoType_BasicInfo"):
            return 0, None, header

        # read basicInfo
        b_data = self._recv(header[-1])
        basicInfo = {
            'size': struct.unpack('<I', b_data[0:4])[0],
            'num_chans': struct.unpack('<I', b_data[4:8])[0],
            'srate': struct.unpack('<I', b_data[8:12])[0],
            'dataSize': struct.unpack('<I', b_data[12:16])[0],
            'allowClientToControlAmp': struct.unpack('<I', b_data[16:20])[0],
            'allowClientToControlRec': struct.unpack('<I', b_data[20:24])[0]
        }

        if basicInfo['num_chans'] > 0 and basicInfo['num_chans'] < maxChans and basicInfo['srate'] > 0 and (
                basicInfo['dataSize'] == 2 or basicInfo['dataSize'] == 4):
            status = 1
        else:
            status = 0

        return status, basicInfo, header

    def getChannelInfoList(self):
        numChannels = self.num_chans

        self.send(self.command_code("RequestChannelInfo"))

        b_header = self._recv(20)
        header = self._unpack_header(b_header)

        if header[0] != 'DATA' \
                or header[1] != self.dataType("Data_Info") \
                or header[2] != self.infoType("InfoType_ChannelInfo"):
            status = 0
            infoList = None
            return status, infoList, header
        infoListRaw = self._recv(header[-1])

        offset_channelId = 0
        offset_chanLabel = offset_channelId + 4
        offset_chanType = offset_chanLabel + 80
        offset_deviceType = offset_chanType + 4
        offset_eegGroup = offset_deviceType + 4
        offset_posX = offset_eegGroup + 4
        offset_posY = offset_posX + 8
        offset_posZ = offset_posY + 8
        offset_posStatus = offset_posZ + 8
        offset_bipolarRef = offset_posStatus + 4
        offset_addScale = offset_bipolarRef + 4
        offset_isDropDown = offset_addScale + 4
        offset_isNoFilter = offset_isDropDown + 4

        chanInfoLen = offset_isNoFilter + 4
        chanInfoLen = round(chanInfoLen / 8) * 8

        infoList = []

        for i in range(numChannels):
            j = chanInfoLen * i
            chanInfo = {
                'id': struct.unpack('<I', infoListRaw[j + offset_channelId: j + offset_chanLabel])[0],
                'chanLabel': infoListRaw[j + offset_chanLabel: j + offset_chanType].replace(b'\x00', b'').decode(
                    'utf-8'),
                'chanType': struct.unpack('<I', infoListRaw[j + offset_chanType: j + offset_deviceType])[0],
                'deviceType': struct.unpack('<I', infoListRaw[j + offset_deviceType: j + offset_eegGroup])[0],
                'eegGroup': struct.unpack('<I', infoListRaw[j + offset_eegGroup: j + offset_posX])[0],
                'posX': struct.unpack('<d', infoListRaw[j + offset_posX: j + offset_posY])[0],
                'posY': struct.unpack('<d', infoListRaw[j + offset_posY: j + offset_posZ])[0],
                'posZ': struct.unpack('<d', infoListRaw[j + offset_posZ: j + offset_posStatus])[0],
                'posStatus': struct.unpack('<I', infoListRaw[j + offset_posStatus: j + offset_bipolarRef])[0],
                'bipolarRef': struct.unpack('<I', infoListRaw[j + offset_bipolarRef: j + offset_addScale])[0],
                'addScale': struct.unpack('<f', infoListRaw[j + offset_addScale: j + offset_isDropDown])[0],
                'isDropDown': struct.unpack('<I', infoListRaw[j + offset_isDropDown: j + offset_isNoFilter])[0],
                'isNoFilter': struct.unpack('<II', infoListRaw[j + offset_isNoFilter: j + chanInfoLen])
            }
            infoList.append(chanInfo)
        status = 1

        return status, infoList, header

    def get_server_version(self):
        self.send(self.command_code('RequestVersion'))

        b_header = self._recv(20)
        header = self._unpack_header(b_header)
        print("get_server_version", header)

        b_data = self._recv(header[-1])
        version = struct.unpack("<I", b_data)[0]
        return version

    def controlCode(self, type):
        if type == 'CTRL_FromServer':
            return 1
        elif type == 'CTRL_FromClient':
            return 2
        else:
            return -1

    def receiveType(self, code):
        if code == 1:
            return "StartAmplifier"
        elif code == 2:
            return "StopAmplifier"

    def requestType(self, type):
        if type == 'RequestVersion':
            return 1
        elif type == 'RequestChannelInfo':
            return 3
        elif type == 'RequestBasicInfoAcq':
            return 6
        elif type == 'RequestStreamingStart':
            return 8
        elif type == 'RequestStreamingStop':
            return 9
        elif type == 'RequestAmpConnect':
            return 10
        elif type == 'RequestAmpDisconnect':
            return 11
        elif type == 'RequestDelay':
            return 16
        else:
            return -1

    def dataType(self, type):
        if type == 'Data_Info':
            return 1
        elif type == 'Data_Eeg':
            return 2
        elif type == 'Data_Events':
            return 3
        elif type == 'Data_Impedance':
            return 4
        else:
            return -1

    def infoType(self, type):
        if type == 'InfoType_Version':
            return 1
        elif type == 'InfoType_BasicInfo':
            return 2
        elif type == 'InfoType_ChannelInfo':
            return 4
        elif type == 'InfoType_StatusAmp':
            return 7
        elif type == 'InfoType_Time':
            return 9
        else:
            return -1

    def blockType(self, t):
        d = -1
        if t == 'DataTypeFloat32bit':
            d = 1
        elif t == 'DataTypeFloat32bitZIP':
            d = 2
        elif t == 'DataTypeEventList':
            d = 3
        return d

    def command_code(self, method):
        c_chID = b"CTRL"
        w_Code = struct.pack('>H', self.controlCode('CTRL_FromClient'))
        w_Request = struct.pack('>H', self.requestType(method))
        un_Sample = struct.pack('>I', 0)
        un_Size = struct.pack('>I', 0)
        un_SizeUn = struct.pack('>I', 0)

        header = c_chID + w_Code + w_Request + un_Sample + un_Size + un_SizeUn
        return header

    def __del__(self):
        print("The session has been disconnected!")
        self.close_connection()


class Neuracle(BaseAmplifier):
    """ An amplifier implementation for neuracle devices.
    -author: Jie Mei
    -Created on: 2022-12-04

    Brief introduction:
    This class is a class for get package data from Neuracle device. To use
    this class, you must start the Neusen W software first, and then click
    the DataService icon on the right part and set parameter. The default
    port is 8712, and you do not need to modifiy it.
    (warning, this class was developed under Newsen W 2.0.1 version, we are
    not sure if it supports the newer version. You could ask for support
    from the Neuracle company.)

    Args:
        device_address: (ip, port)
        srate: sample rate of device, the default value of Neuracle is 1000
        num_chans: channel of data, for Neuracle, including data
                    channel and trigger channel
    """

    def __init__(self,
                 device_address: Tuple[str, int] = ('127.0.0.1', 8712),
                 srate=1000,
                 num_chans=9):
        super().__init__()
        self.device_address = device_address
        self.srate = srate
        self.num_chans = num_chans
        self.tcp_link = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._update_time = 0.04
        self.pkg_size = int(
            self._update_time *
            4 *
            self.num_chans *
            self.srate)

    def set_timeout(self, timeout):
        if self.tcp_link:
            self.tcp_link.settimeout(timeout)

    def recv(self):
        # wait for the socket available
        data = None
        # rs, _, _ = select.select([self.tcp_link], [], [], 9)
        try:
            raw_data = self.tcp_link.recv(self.pkg_size)
        except Exception:
            self.tcp_link.close()
            print("Can not receive data from socket")
        else:
            data, evt = self._unpack_data(raw_data)
            data = data.reshape(len(data) // self.num_chans, self.num_chans)
        return data.tolist()

    def _unpack_data(self, raw):
        len_raw = len(raw)
        event, hex_data = [], []
        # unpack hex_data in row
        hex_data = raw[:len_raw - np.mod(len_raw, 4 * self.num_chans)]
        n_item = int(len(hex_data) / 4 / self.num_chans)
        format_str = '<' + (str(self.num_chans) + 'f') * n_item
        unpack_data = struct.unpack(format_str, hex_data)

        return np.asarray(unpack_data), event

    def connect_tcp(self):
        self.tcp_link.connect(self.device_address)

    def start_trans(self):
        time.sleep(1e-2)
        self.start()

    def stop_trans(self):
        self.stop()

    def close_connection(self):
        if self.tcp_link:
            self.tcp_link.close()
            self.tcp_link = None


class LSLInlet:
    """Base class for a intlet"""

    def __init__(self, info: pylsl.StreamInfo) -> None:
        self.inlet = pylsl.StreamInlet(
            info, max_buflen=3,
            processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter)

        self.name = info.name()
        self.channel_count = info.channel_count()

    def stream_action(self):
        pass


class DataInlet(LSLInlet):
    dtypes = [[], np.float32, np.float64, None,
              np.int32, np.int16, np.int8, np.int64]

    def __init__(self, info: pylsl.StreamInfo) -> None:
        super().__init__(info)
        # Define two queue for storage the data retrieved from device
        # and their timestamp range.
        self.data_queue: queue.Queue[Any] = queue.Queue(3)

    def stream_action(self):
        samples, ts = self.inlet.pull_chunk(
            timeout=0.0, max_samples=40)
        if ts:
            samples = np.asarray(samples)
            ts = np.asarray(ts)
            pack_data = np.hstack((samples, ts.reshape((-1, 1))))
            self.data_queue.put(pack_data)

    def get_data(self):
        if self.data_queue.full():
            data = self.data_queue.get()
            return data
        else:
            return np.asarray([0])


class MarkerInlet(LSLInlet):
    def __init__(self, info: pylsl.StreamInfo) -> None:
        super().__init__(info)

    def stream_action(self):
        marker_value, marker_ts = self.inlet.pull_sample(0.0)
        if marker_ts:
            # cache = []
            # for content, ts in zip(marker_value, marker_ts):
            try:
                int_label = int(marker_value[0])
            except Exception:
                raise ValueError(
                    "The marker value: {} can not be \
                        typed into int".format(marker_value))
                # cache.append([int_label, ts])
            return [int_label, marker_ts]
        else:
            return []


class LSLapps():
    """An amplifier implementation for Lab streaming layer (LSL) apps.
    LSL ref as: https://github.com/sccn/labstreaminglayer
    The LSL provides many builded apps for communiacting with varities
    of devices, and some of the are EEG acquiring device, like EGI, g.tec,
    DSI and so on. For metabci, here we just provide a pathway for reading
    lsl data streams, which means as long as the the LSL providing the app,
    the metabci could support its online application. Considering the
    differences among different devices for transfering the event trigger.
    YOU MUST BE VERY CAREFUL to determine wethher the data stream reading
    from the LSL apps contains a event channel. For example, the neuroscan
    synamp II will append a extra event channel to the raw data channel.
    Because we do not have chance to test each device that LSL supported, so
    please modify this class before using with your own condition.
    """

    def __init__(self, ):
        super().__init__()
        self.marker_inlet = None
        self.data_inlet = None
        self.device_data = None
        self.marker_data = None
        self.marker_cache = list()
        self.marker_count = 0
        self.streams_count = 0
        self.pending_stream = []
        self.data_response = np.zeros(1)
        self.bg_stream_checker = pylsl.ContinuousResolver()
        time.sleep(1.5)
        self.stream_checker_threading = threading.Thread(
            target=self.stream_checker, name="stream_checker")
        self.stream_checker_threading.start()

    def stream_checker(self):
        while True:
            streams = self.bg_stream_checker.results()
            if len(streams) != self.streams_count:
                self.streams_count = len(streams)
                for info in streams:
                    if info.type() == 'Markers':
                        if info.nominal_srate() != pylsl.IRREGULAR_RATE \
                                or info.channel_format() != pylsl.cf_string:
                            print('Invalid marker stream ' + info.name())
                        print('Adding marker inlet: ' + info.name())
                        self.marker_inlet = MarkerInlet(info)
                    elif info.nominal_srate() != pylsl.IRREGULAR_RATE \
                            and info.channel_format() != pylsl.cf_string:
                        print('Adding data inlet: ' + info.name())
                        self.data_inlet = DataInlet(info)
                    else:
                        print('Don\'t know what to do \
                                with stream ' + info.name())
            time.sleep(0.5)

    def recv(self):
        if self.marker_inlet is not None:
            self.marker_data = self.marker_inlet.stream_action()
        # Check if there are markers retriving from the stream.
        if self.marker_data:
            self.marker_cache.append(self.marker_data)
            # print("Catch a trigger, content is: {}".format(self.marker_data))
        if self.data_inlet is not None:
            self.data_inlet.stream_action()
            self.data_response = self.data_inlet.get_data()
        # Check if there are devices data from the stream. Because we kept
        # a buffer, so the data will be delay about 80points. In case we
        # miss the labels
        if self.data_response.any():
            device_data = self.data_response
            epoch_length = device_data.shape[0]
            # Create a zero vector as label line
            label_line = np.zeros(epoch_length)
            # Find the label position
            for label in self.marker_cache:
                position = device_data[:, -1].searchsorted(label[-1])
                # The smaller index in the marker cache means a earlier label
                if position >= epoch_length:
                    # IF larger than the epoch max index, we say the timestamp
                    # is out of the range of current device epoch.
                    break
                else:
                    label_line[position] = int(label[0])
                    # print("The trigger position has been \
                    #       assigned to {}".format(position))
                    # print("LSL clock delta: \
                    #         {}".format(label[-1]-device_data[position, -1]))
                    # POP out the current index
                    self.marker_cache.remove(label)
            # Replaced the last column of device_data as the trigger column
            device_data[:, -1] = label_line
            return device_data.tolist()
        else:
            return []

    def start_trans(self):
        time.sleep(1e-2)
        self.start()

    def stop_trans(self):
        self.stop()


class HTOnlineSystem(BaseAmplifier):
    """
    An amplifier implementation for digital electroencephalograph device. It will analog amplify the collected
    EEG signals, analog filter them and convert them into digital signals. Then it is transmitted to the host
    computer EEG acquisition software through Ethernet for data display and storage.

    author: Wei Zhao <vivian@tju.edu.cn>

    Created on: 2023-12-4

    update log:


    Parameters
    ----------
    device_address : Tuple[ip : str, port : int]
        ip : IP address of the collection host computer.
        port : The port number.
    srate : float
        Sampling Rate, default is 1000.
    packet_samples: float
        The number of sampling points contained in each data packet, default is 100.
    num_chans: int
        Number of channels, default is 32.

    Attributes
    ----------
    tcp_link : socket object
        Socket object used for TCP connections.
    packet_points : int
        The number of sample points for all channels contained in the data packet.
    pkg_size : int
        The number of bytes occupied by the data packet.
    timeout: float
        Overtime time.

    Raises
    ----------
    ValueError
        Srate mismatch.
        Samples for each package mismatch.
        Num of chans mismatch.

    """

    _COMMANDS = {
        "start_acq": bytes([165, 16, 1, 90]),
        "stop_acq": bytes([165, 16, 2, 90]),
        "get_srate": bytes([165, 1, 1, 90]),
        "get_samples": bytes([165, 1, 2, 90]),
        "get_num_chs": bytes([165, 1, 3, 90]),
        "get_name_chs": bytes([165, 1, 4, 90])
    }

    def __init__(
        self,
        device_address: Tuple[str, int] = ("127.0.0.1", 4000),
        srate: float = 1000,
        packet_samples: float = 100,
        num_chans: int = 32
    ):
        super().__init__()
        self.device_address = device_address
        self.srate = srate
        self.packet_samples = packet_samples
        self.tcp_link = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.num_chans = num_chans
        self.packet_points = (num_chans + 1) * packet_samples
        self.pkg_size = self.packet_points * 4
        self.timeout = 2 * 25 / self.srate

    def _unpack_header(self, b_header):
        """
        Unpack header.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:

        Parameters
        ----------
        b_header: bytes
            Frame header to be unpacked.

        Returns
        ----------
        upack_header :  cell(header : int, attribute_id : int, attribute_num : int, pkg_size : int)
        header : int
            Frame header.
        attribute_id : int
            The attribute id.
        attribute_num : int
            Number of attribute values.
        pkg_size : int
            Number of bytes in all attribute values

        """

        header = struct.unpack("<B", b_header[:1])
        attribute_id = struct.unpack("<B", b_header[1:2])
        attribute_num = struct.unpack("<H", b_header[2:4])
        pkg_size = struct.unpack("<I", b_header[4:])
        return (header[0], attribute_id[0], attribute_num[0], pkg_size[0])

    def _unpack_data(self, b_data):
        """
        Unpack data.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:

        Parameters
        ----------
        b_data: bytes
            Data to be unpacked.

        Returns
        ----------
        samples :  list
            Unpacked data.

        """

        fmt = "<" + str(self.packet_points) + "f"
        samples = np.array(struct.unpack(fmt, b_data))  # 解开包
        samples = samples.reshape(-1, self.num_chans + 1)
        return samples.tolist()

    def _recv(self, num_bytes):
        """
        Receive the specified bytes of data.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:

        Parameters
        ----------
        num_bytes: int
            Number of bytes to accept.

        Returns
        ----------
        b_data:  bytes
            Received data of specified byte size.

        """

        fragments = []
        b_count = 0
        while b_count < num_bytes:
            try:
                chunk = self.tcp_link.recv(num_bytes - b_count)
            except socket.timeout as e:
                raise e
            b_count += len(chunk)
            fragments.append(chunk)

        b_data = b"".join(fragments)
        return b_data

    def recv(self):
        """
        The minimal recv data function, usually a package.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:

        Parameters
        ----------

        Returns
        ----------
        samples:  bytes
            An unpacked data packet.

        """

        samples = None
        try:
            b_header = self._recv(8)
            header = self._unpack_header(b_header)
            if header[-1] == self.pkg_size:
                raw_data = self._recv(self.pkg_size)
                self._recv(1)

        except Exception:
            self.tcp_link.close()
            print("Can not receive data from socket")
        else:
            samples = self._unpack_data(raw_data)
        return samples

    def send(self, message):
        """
        Send command.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:

        Parameters
        ----------
        samples: bytes
            The command bytes needed to be sent.

        Returns
        ----------

        """

        self.tcp_link.sendall(message)

    def get_srate(self):
        """
        Get the sampling rate of the device.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:

        Parameters
        ----------

        Returns
        ----------
        srate : int
            The sampling rate of the device.

        """

        self.tcp_link.sendall(self._COMMANDS["get_srate"])
        b_data = self._recv(13)
        srate = int.from_bytes(b_data[8:10], "little")
        return srate

    def get_samples(self):
        """
        Get the number of sample points contained in each data packet of the device.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:


        Parameters
        ----------

        Returns
        ----------
        num_samples : int
            The number of sample points contained in each data packet.

        """

        self.tcp_link.sendall(self._COMMANDS["get_samples"])
        b_data = self._recv(13)
        num_samples = int.from_bytes(b_data[8:10], "little")
        return num_samples

    def get_num_chs(self):
        """
        Get the number of channels of the device.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:

        Parameters
        ----------

        Returns
        ----------
        num_chs : int
            The number of channels of the device.

        """

        self.tcp_link.sendall(self._COMMANDS["get_num_chs"])
        b_data = self._recv(13)
        num_chs = int.from_bytes(b_data[8:10], "little")
        return num_chs

    def get_name_chans(self):
        """
        Get the channels used by the devices.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:

        Parameters
        ----------

        Returns
        ----------
        chs_list : str
            The channels used by the devices.

        """

        self.tcp_link.sendall(self._COMMANDS["get_name_chs"])
        b_header = self._recv(8)
        header = self._unpack_header(b_header)
        samples = None
        attr_nums = (self.num_chans + 1) * 8
        if header[-1] == attr_nums:
            b_data = self._recv(attr_nums)
            samples = struct.unpack("<" + str(attr_nums) + "B", b_data)
            self._recv(1)  # 帧尾
        chs_list = []
        ch = ""
        for sample in samples:
            if chr(sample) == "\t":
                chs_list.append(ch)
                ch = ""
            elif chr(sample) != " ":
                ch += chr(sample)
        return chs_list

    def set_timeout(self, timeout):
        """
        Set timeout.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:

        Parameters
        ----------
        timeout: float
            Overtime time.

        Returns
        ----------

        """

        if self.tcp_link:
            self.tcp_link.settimeout(timeout)

    def connect_tcp(self):
        """
        Establish tcp connection.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:

        Parameters
        ----------

        Returns
        ----------

        """

        self.tcp_link.connect(self.device_address)
        if self.get_srate() != self.srate:
            raise ValueError("Srate mismatch.")
        if self.get_samples() != self.packet_samples:
            raise ValueError("Samples for each package mismatch.")
        if self.get_num_chs() != self.num_chans + 1:
            raise ValueError("Num of chans mismatch.")

    def close_connection(self):
        """
        Close tcp connection.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:

        Parameters
        ----------

        Returns
        ----------

        """
        if self.tcp_link:
            self.tcp_link.close()
            self.tcp_link = None

    def start_acq(self):
        """
        Start acquiring data.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:

        Parameters
        ----------

        Returns
        ----------

        """
        self.send(self._COMMANDS["start_acq"])
        time.sleep(1e-2)
        self.start()

    def stop_acq(self):
        """
        Stop acquiring data.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:

        Parameters
        ----------

        Returns
        ----------

        """
        self.send(self._COMMANDS["stop_acq"])
        self.stop()




### ==============================添加内容=============================== ###
class BoardIds(enum.IntEnum):
    """Enum to store all supported Board Ids
    Update log: 2025-05-22 by Guangjin Liang <3330635482@qq.com>
    Modified from https://github.com/brainflow-dev/brainflow/blob/master/python_package/brainflow/board_shim.py
    """
    NO_BOARD = -100
    CYTON_BOARD = 0  #:
    CYTON_DAISY_BOARD = 2  #:
    CYTON_WIFI_BOARD = 5  #:
    CYTON_DAISY_WIFI_BOARD = 6  #:
    STREAMING_BOARD = -2  #:
    PLAYBACK_FILE_BOARD = -3  #:


class IpProtocolTypes(enum.IntEnum):
    """Enum to store Ip Protocol types
    Update log: 2025-05-22 by Guangjin Liang <3330635482@qq.com>
    Modified from https://github.com/brainflow-dev/brainflow/blob/master/python_package/brainflow/board_shim.py
    """
    NO_IP_PROTOCOL = 0  #:
    UDP = 1  #:
    TCP = 2  #:


class BrainFlowPresets(enum.IntEnum):
    """Enum to store presets
    Update log: 2025-05-22 by Guangjin Liang <3330635482@qq.com>
    Modified from https://github.com/brainflow-dev/brainflow/blob/master/python_package/brainflow/board_shim.py
    """
    DEFAULT_PRESET = 0  #:
    AUXILIARY_PRESET = 1  #:
    ANCILLARY_PRESET = 2  #:


class BrainFlowExitCodes(enum.IntEnum):
    """Enum to store all possible exit codes
    Update log: 2025-05-22 by Guangjin Liang <3330635482@qq.com>
    Modified from https://github.com/brainflow-dev/brainflow/blob/master/python_package/brainflow/exit_codes.py
    """
    STATUS_OK = 0
    PORT_ALREADY_OPEN_ERROR = 1
    UNABLE_TO_OPEN_PORT_ERROR = 2
    SER_PORT_ERROR = 3
    BOARD_WRITE_ERROR = 4
    INCOMMING_MSG_ERROR = 5
    INITIAL_MSG_ERROR = 6
    BOARD_NOT_READY_ERROR = 7
    STREAM_ALREADY_RUN_ERROR = 8
    INVALID_BUFFER_SIZE_ERROR = 9
    STREAM_THREAD_ERROR = 10
    STREAM_THREAD_IS_NOT_RUNNING = 11
    EMPTY_BUFFER_ERROR = 12
    INVALID_ARGUMENTS_ERROR = 13
    UNSUPPORTED_BOARD_ERROR = 14
    BOARD_NOT_CREATED_ERROR = 15
    ANOTHER_BOARD_IS_CREATED_ERROR = 16
    GENERAL_ERROR = 17
    SYNC_TIMEOUT_ERROR = 18
    JSON_NOT_FOUND_ERROR = 19
    NO_SUCH_DATA_IN_JSON_ERROR = 20
    CLASSIFIER_IS_NOT_PREPARED_ERROR = 21
    ANOTHER_CLASSIFIER_IS_PREPARED_ERROR = 22
    UNSUPPORTED_CLASSIFIER_AND_METRIC_COMBINATION_ERROR = 23


class BrainFlowError(Exception):
    """This exception is raised if non-zero exit code is returned from C code

    :param message: exception message
    :type message: str
    :param exit_code: exit code flow low level API
    :type exit_code: int

    Update log: 2025-05-22 by Guangjin Liang <3330635482@qq.com>
    Modified from https://github.com/brainflow-dev/brainflow/blob/master/python_package/brainflow/exit_codes.py
    """

    def __init__(self, message: str, exit_code: int) -> None:
        detailed_message = '%s:%d %s' % (BrainFlowExitCodes(exit_code).name, exit_code, message)
        super(BrainFlowError, self).__init__(detailed_message)
        self.exit_code = exit_code


class OpenBCIInputParams(object):
    """ inputs parameters for prepare_session method

    :param serial_port: serial port name is used for boards which reads data from serial port
    :type serial_port: str
    :param mac_address: mac address for example its used for bluetooth based boards
    :type mac_address: str
    :param ip_address: ip address is used for boards which reads data from socket connection
    :type ip_address: str
    :param ip_address_aux: ip address is used for boards which reads data from socket connection
    :type ip_address_aux: str
    :param ip_address_anc: ip address is used for boards which reads data from socket connection
    :type ip_address_anc: str
    :param ip_port: ip port for socket connection, for some boards where we know it in front you dont need this parameter
    :type ip_port: int
    :param ip_port_aux: ip port for socket connection, for some boards where we know it in front you dont need this parameter
    :type ip_port_aux: int
    :param ip_port_anc: ip port for socket connection, for some boards where we know it in front you dont need this parameter
    :type ip_port_anc: int
    :param ip_protocol: ip protocol type from IpProtocolTypes enum
    :type ip_protocol: int
    :param other_info: other info
    :type other_info: str
    :param serial_number: serial number
    :type serial_number: str
    :param file: file
    :type file: str
    :param file_aux: file
    :type file_aux: str
    :param file_anc: file
    :type file_anc: str

    Update log: 2025-05-22 by Guangjin Liang <3330635482@qq.com>
    Modified from https://github.com/brainflow-dev/brainflow/blob/master/python_package/brainflow/board_shim.py
    """

    def __init__(self) -> None:
        self.serial_port = ''
        self.mac_address = ''
        self.ip_address = ''
        self.ip_address_aux = ''
        self.ip_address_anc = ''
        self.ip_port = 0
        self.ip_port_aux = 0
        self.ip_port_anc = 0
        self.ip_protocol = IpProtocolTypes.NO_IP_PROTOCOL.value
        self.other_info = ''
        self.timeout = 0
        self.serial_number = ''
        self.file = ''
        self.file_aux = ''
        self.file_anc = ''
        self.master_board = BoardIds.NO_BOARD.value

    def to_json(self) -> None:
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


class OpenBCIControllerDLL(object):
    '''
    Update log: 2025-05-22 by Guangjin Liang <3330635482@qq.com>
    Modified from https://github.com/brainflow-dev/brainflow/blob/master/python_package/brainflow/board_shim.py
    '''
    __instance = None

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

    def __init__(self):
        if platform.system() == 'Windows':
            if struct.calcsize("P") * 8 == 64:
                dll_path = 'lib\\BoardController.dll'
            else:
                dll_path = 'lib\\BoardController32.dll'
        full_path = pkg_resources.resource_filename(__name__, dll_path)
        if os.path.isfile(full_path):
            dir_path = os.path.abspath(os.path.dirname(full_path))
            # for python we load dll by direct path but this dll may depend on other dlls and they will not be found!
            # to solve it we can load all of them before loading the main one or change PATH\LD_LIBRARY_PATH env var.
            # env variable looks better, since it can be done only once for all dependencies
            # for python 3.8 PATH env var doesnt work anymore
            try:
                os.add_dll_directory(dir_path)
            except:
                pass
            if platform.system() == 'Windows':
                os.environ['PATH'] = dir_path + os.pathsep + os.environ.get('PATH', '')
            else:
                os.environ['LD_LIBRARY_PATH'] = dir_path + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')
            self.lib = ctypes.cdll.LoadLibrary(full_path)
        else:
            raise FileNotFoundError(
                'Dynamic library %s is missed, did you forget to compile brainflow before installation of python package?' % full_path)

        self.prepare_session = self.lib.prepare_session
        self.prepare_session.restype = ctypes.c_int
        self.prepare_session.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p
        ]

        self.start_stream = self.lib.start_stream
        self.start_stream.restype = ctypes.c_int
        self.start_stream.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p
        ]

        self.stop_stream = self.lib.stop_stream
        self.stop_stream.restype = ctypes.c_int
        self.stop_stream.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p
        ]

        self.get_num_rows = self.lib.get_num_rows
        self.get_num_rows.restype = ctypes.c_int
        self.get_num_rows.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ndpointer(ctypes.c_int32)
        ]

        self.get_current_board_data = self.lib.get_current_board_data
        self.get_current_board_data.restype = ctypes.c_int
        self.get_current_board_data.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ndpointer(ctypes.c_double),
            ndpointer(ctypes.c_int32),
            ctypes.c_int,
            ctypes.c_char_p
        ]

        self.get_sampling_rate = self.lib.get_sampling_rate
        self.get_sampling_rate.restype = ctypes.c_int
        self.get_sampling_rate.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ndpointer(ctypes.c_int32)
        ]

        self.get_timestamp_channel = self.lib.get_timestamp_channel
        self.get_timestamp_channel.restype = ctypes.c_int
        self.get_timestamp_channel.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ndpointer(ctypes.c_int32)
        ]

        self.get_eeg_names = self.lib.get_eeg_names
        self.get_eeg_names.restype = ctypes.c_int
        self.get_eeg_names.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ndpointer(ctypes.c_ubyte),
            ndpointer(ctypes.c_int32)
        ]

        self.get_eeg_channels = self.lib.get_eeg_channels
        self.get_eeg_channels.restype = ctypes.c_int
        self.get_eeg_channels.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ndpointer(ctypes.c_int32),
            ndpointer(ctypes.c_int32)
        ]


class OpenBCIShim(object):
    """BoardShim class is a primary interface to all boards

    :param board_id: Id of your board
    :type board_id: int
    :param input_params: board specific structure to pass required arguments
    :type input_params: BrainFlowInputParams

    Update log: 2025-05-22 by Guangjin Liang <3330635482@qq.com>
    Modified from https://github.com/brainflow-dev/brainflow/blob/master/python_package/brainflow/board_shim.py
    """

    def __init__(self, board_id: int, input_params: OpenBCIControllerDLL) -> None:
        try:
            self.input_json = input_params.to_json().encode()
        except BaseException:
            self.input_json = input_params.to_json()
        self.board_id = board_id
        # we need it for streaming board
        if board_id == BoardIds.STREAMING_BOARD.value or board_id == BoardIds.PLAYBACK_FILE_BOARD.value:
            if input_params.master_board != BoardIds.NO_BOARD:
                self._master_board_id = input_params.master_board
            else:
                raise BrainFlowError('you need set master board id in BrainFlowInputParams',
                                     BrainFlowExitCodes.INVALID_ARGUMENTS_ERROR.value)
        else:
            self._master_board_id = self.board_id

    @classmethod
    def get_sampling_rate(cls, board_id: int, preset: int = BrainFlowPresets.DEFAULT_PRESET) -> int:
        """get sampling rate for a board

        :param board_id: Board Id
        :type board_id: int
        :param preset: preset
        :type preset: int
        :return: sampling rate for this board id
        :rtype: int
        :raises BrainFlowError: If this board has no such data exit code is UNSUPPORTED_BOARD_ERROR
        """

        sampling_rate = numpy.zeros(1).astype(numpy.int32)
        res = OpenBCIControllerDLL.get_instance().get_sampling_rate(board_id, preset, sampling_rate)
        if res != BrainFlowExitCodes.STATUS_OK.value:
            raise BrainFlowError('unable to request info about this board', res)
        return int(sampling_rate[0])

    @classmethod
    def get_timestamp_channel(cls, board_id: int, preset: int = BrainFlowPresets.DEFAULT_PRESET) -> int:
        """get timestamp channel in resulting data table for a board

        :param board_id: Board Id
        :type board_id: int
        :param preset: preset
        :type preset: int
        :return: number of timestamp channel in returned numpy array
        :rtype: int
        :raises BrainFlowError: If this board has no such data exit code is UNSUPPORTED_BOARD_ERROR
        """

        timestamp_channel = numpy.zeros(1).astype(numpy.int32)
        res = OpenBCIControllerDLL.get_instance().get_timestamp_channel(board_id, preset, timestamp_channel)
        if res != BrainFlowExitCodes.STATUS_OK.value:
            raise BrainFlowError('unable to request info about this board', res)
        return int(timestamp_channel[0])

    @classmethod
    def get_eeg_names(cls, board_id: int, preset: int = BrainFlowPresets.DEFAULT_PRESET) -> List[str]:
        """get names of EEG channels in 10-20 system if their location is fixed

        :param board_id: Board Id
        :type board_id: int
        :param preset: preset
        :type preset: int
        :return: EEG channels names
        :rtype: List[str]
        :raises BrainFlowError: If this board has no such data exit code is UNSUPPORTED_BOARD_ERROR
        """

        string = numpy.zeros(4096).astype(numpy.ubyte)
        string_len = numpy.zeros(1).astype(numpy.int32)
        res = OpenBCIControllerDLL.get_instance().get_eeg_names(board_id, preset, string, string_len)
        if res != BrainFlowExitCodes.STATUS_OK.value:
            raise BrainFlowError('unable to request info about this board', res)
        return string.tobytes().decode('utf-8')[0:string_len[0]].split(',')

    @classmethod
    def get_eeg_channels(cls, board_id: int, preset: int = BrainFlowPresets.DEFAULT_PRESET) -> List[int]:
        """get list of eeg channels in resulting data table for a board

        :param board_id: Board Id
        :type board_id: int
        :param preset: preset
        :type preset: int
        :return: list of eeg channels in returned numpy array
        :rtype: List[int]
        :raises BrainFlowError: If this board has no such data exit code is UNSUPPORTED_BOARD_ERROR
        """

        num_channels = numpy.zeros(1).astype(numpy.int32)
        eeg_channels = numpy.zeros(512).astype(numpy.int32)

        res = OpenBCIControllerDLL.get_instance().get_eeg_channels(board_id, preset, eeg_channels, num_channels)
        if res != BrainFlowExitCodes.STATUS_OK.value:
            raise BrainFlowError('unable to request info about this board', res)
        result = eeg_channels.tolist()[0:num_channels[0]]
        return result

    def prepare_session(self) -> None:
        """prepare streaming sesssion, init resources, you need to call it before any other BoardShim object methods"""

        res = OpenBCIControllerDLL.get_instance().prepare_session(self.board_id, self.input_json)
        if res != BrainFlowExitCodes.STATUS_OK.value:
            raise BrainFlowError('unable to prepare streaming session', res)

    def start_stream(self, num_samples: int = 1800 * 250, streamer_params: str = None) -> None:
        """Start streaming data, this methods stores data in ringbuffer

        :param num_samples: size of ring buffer to keep data
        :type num_samples: int
        :param streamer_params parameter to stream data from brainflow, supported vals: "file://%file_name%:w", "file://%file_name%:a", "streaming_board://%multicast_group_ip%:%port%". Range for multicast addresses is from "224.0.0.0" to "239.255.255.255"
        :type streamer_params: str
        """

        if streamer_params is None:
            streamer = None
        else:
            try:
                streamer = streamer_params.encode()
            except BaseException:
                streamer = streamer_params

        res = OpenBCIControllerDLL.get_instance().start_stream(num_samples, streamer, self.board_id, self.input_json)
        if res != BrainFlowExitCodes.STATUS_OK.value:
            raise BrainFlowError('unable to start streaming session', res)

    def stop_stream(self) -> None:
        """Stop streaming data"""

        res = OpenBCIControllerDLL.get_instance().stop_stream(self.board_id, self.input_json)
        if res != BrainFlowExitCodes.STATUS_OK.value:
            raise BrainFlowError('unable to stop streaming session', res)

    @classmethod
    def get_num_rows(cls, board_id: int, preset: int = BrainFlowPresets.DEFAULT_PRESET) -> int:
        """get number of rows in resulting data table for a board

        :param board_id: Board Id
        :type board_id: int
        :param preset: preset
        :type preset: int
        :return: number of rows in returned numpy array
        :rtype: int
        :raises BrainFlowError: If this board has no such data exit code is UNSUPPORTED_BOARD_ERROR
        """

        num_rows = numpy.zeros(1).astype(numpy.int32)
        res = OpenBCIControllerDLL.get_instance().get_num_rows(board_id, preset, num_rows)
        if res != BrainFlowExitCodes.STATUS_OK.value:
            raise BrainFlowError('unable to request info about this board', res)
        return int(num_rows[0])

    def get_current_board_data(self, num_samples: int, preset: int = BrainFlowPresets.DEFAULT_PRESET):
        """Get specified amount of data or less if there is not enough data, doesnt remove data from ringbuffer

        :param num_samples: max number of samples
        :type num_samples: int
        :param preset: preset
        :type preset: int
        :return: latest data from a board
        :rtype: NDArray[Shape["*, *"], Float64]
        """

        package_length = OpenBCIShim.get_num_rows(self._master_board_id, preset)
        data_arr = numpy.zeros(int(num_samples * package_length)).astype(numpy.float64)
        current_size = numpy.zeros(1).astype(numpy.int32)

        res = OpenBCIControllerDLL.get_instance().get_current_board_data(num_samples, preset, data_arr, current_size,
                                                                       self.board_id, self.input_json)
        if res != BrainFlowExitCodes.STATUS_OK.value:
            raise BrainFlowError('unable to get current data', res)

        if len(current_size) == 0:
            return None

        data_arr = data_arr[0:current_size[0] * package_length].reshape(package_length, current_size[0])
        return data_arr
### ==============================添加内容=============================== ###