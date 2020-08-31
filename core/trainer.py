# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import os
import time
import sys
import traceback

from paddle import fluid

from paddlerec.core.utils import envs


class EngineMode:
    """
    There are various engine designed for different runing environment.
    """
    SINGLE = 1
    CLUSTER = 2
    LOCAL_CLUSTER = 3


class FleetMode:
    """
    Paddle Distributed train support: ParameterServer/Collective/PSlib
    """
    PS = 1
    COLLECTIVE = 2
    PSLIB = 3


class Device:
    """
    PaddleRec Support CPU/GPU, XPU will comming soon
    """
    CPU = 1
    GPU = 2
    # XPU =3


class Trainer(object):
    """
    Trainer Base
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, config=None):
        self._status_processor = {}
        self.model = None
        self.inference_models = []
        self.increment_models = []
        self._exector_context = {}
        self._context = {'status': 'uninit', 'is_exit': False}
        self._context["config_yaml"] = config

        self._model = {}
        self._dataset = {}

        self._runner_name = envs.get_runtime_environ("mode")
        self._context["runner_name"] = self._runner_name

        phase_names = envs.get_global_env(
            "runner." + self._runner_name + ".phases", None)

        _config = envs.load_yaml(config)

        self._context["env"] = _config
        self._context["dataset"] = _config.get("dataset")

        phases = []
        if phase_names is None:
            phases = _config.get("phase")
        else:
            for phase in _config.get("phase"):
                if phase["name"] in phase_names:
                    phases.append(phase)

        self._context["phases"] = phases
        print("PaddleRec: Runner {} Begin".format(self._runner_name))
        self.which_engine()
        self.which_device()
        self.which_fleet_mode()
        self.which_executor_mode()
        self.legality_check()
        '''
        完整的context字典：
        {'status': 'uninit', 'is_exit': False, 'phases': [{'model': '{workspace}/model.py', 'thread_num': 1, 'name': 'phase_train', 'dataset_name': 'data1'}],
         'exe': <paddle.fluid.executor.Executor object at 0x7f304de96d10>, 'is_pslib': False, 'is_infer': False, 
         'dataset': [{'type': 'DataLoader', 'data_converter': '{workspace}/reader.py', 'data_path': '{workspace}/data/train', 'name': 'data1', 'batch_size': 10}, 
                {'type': 'DataLoader', 'data_converter': '{workspace}/reader.py', 'data_path': '{workspace}/data/test', 'name': 'dataset_infer', 'batch_size': 2}], 
         'engine': 1, 'fleet_mode': 'ps', 'place': <paddle.fluid.core_avx.CPUPlace object at 0x7f304de2ca40>, 
         'env': {'phase': [{'model': '{workspace}/model.py', 'thread_num': 1, 'name': 'phase_train', 'dataset_name': 'data1'}, 
                    {'model': '{workspace}/model.py', 'thread_num': 1, 'name': 'phase_infer', 'dataset_name': 'dataset_infer'}], 
                 'hyper_parameters': {'cnn_dim': 128, 'optimizer': {'learning_rate': 0.001, 'class': 'Adagrad'}, 'max_len': 100, 'hid_dim': 96, 'emb_dim': 128, 'cnn_filter_size1': 1,
                     'cnn_filter_size2': 2, 'dict_dim': 33257, 'class_dim': 2, 'cnn_filter_size3': 3, 'is_sparse': False}, 
                 'workspace': 'models/contentunderstanding/classification', 
                 'runner': [{'init_model_path': '', 'phases': 'phase_train', 'save_inference_feed_varnames': [], 
                     'name': 'train_runner', 'save_checkpoint_path': 'increment', 'epochs': 16, 'save_inference_path': 'inference', 'save_checkpoint_interval': 1, 
                     'save_inference_fetch_varnames': [], 'print_interval': 10, 'save_inference_interval': 1, 'device': 'cpu', 'class': 'train'}, 
                     {'init_model_path': 'increment/14', 'phases': 'phase_infer', 'name': 'infer_runner', 'device': 'cpu', 'class': 'infer', 'print_interval': 1}], 
                 'mode': ['train_runner', 'infer_runner'], 
                 'dataset': [{'type': 'DataLoader', 'data_converter': '{workspace}/reader.py', 'data_path': '{workspace}/data/train', 'name': 'data1', 'batch_size': 10}, 
                     {'type': 'DataLoader', 'data_converter': '{workspace}/reader.py', 'data_path': '{workspace}/data/test', 'name': 'dataset_infer', 'batch_size': 2}]},
         'config_yaml': 'models/contentunderstanding/classification/config.yaml', 'device': 'CPU', 'is_fleet': False, 'runner_name': 'train_runner'}
        '''

    def which_device(self):
        """R
        """
        device = envs.get_global_env(
            "runner." + self._runner_name + ".device", default_value="CPU")
        device = device.upper()

        if device == 'GPU':
            self.check_gpu()
            self.device = Device.GPU
            gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
            self._place = fluid.CUDAPlace(gpu_id)
            self._exe = fluid.Executor(self._place)
        elif device == "CPU":
            self.device = Device.CPU
            self._place = fluid.CPUPlace()
            self._exe = fluid.Executor(self._place)
        else:
            raise ValueError("Not Support device {}".format(device))
        self._context["device"] = device
        self._context["exe"] = self._exe
        self._context["place"] = self._place

    def check_gpu(self):
        """
        Log error and exit when set use_gpu=true in paddlepaddle
        cpu version.
        """
        err = "GPU cannot be set as true while you are " \
            "using paddlepaddle cpu version ! \nPlease try: \n" \
            "\t1. Install paddlepaddle-gpu to run model on GPU \n" \
            "\t2. Set device as cpu in config file to run " \
            "model on CPU"

        try:
            if not fluid.is_compiled_with_cuda():
                raise RuntimeError(err)
        except Exception as e:
            pass

    def which_engine(self):
        engine = envs.get_runtime_environ("train.trainer.engine")
        if engine.upper() == "SINGLE":
            self.engine = EngineMode.SINGLE
            self.is_fleet = False
        elif engine.upper() == "LOCAL_CLUSTER":
            self.engine = EngineMode.LOCAL_CLUSTER
            self.is_fleet = True
        elif engine.upper() == "CLUSTER":
            self.engine = EngineMode.CLUSTER
            self.is_fleet = True
        else:
            raise ValueError("Not Support Engine {}".format(engine))
        self._context["is_fleet"] = self.is_fleet
        self._context["engine"] = self.engine

    def which_fleet_mode(self):
        fleet_mode = envs.get_runtime_environ("fleet_mode")
        if fleet_mode.upper() == "PS":
            self.fleet_mode = FleetMode.PS
        elif fleet_mode.upper() == "COLLECTIVE":
            self.fleet_mode = FleetMode.COLLECTIVE
        elif fleet_mode.upper() == "PSLIB":
            self.fleet_mode = FleetMode.PSLIB
        else:
            raise ValueError("Not Support Fleet Mode {}".format(fleet_mode))

        self._context["is_pslib"] = (fleet_mode.upper() == "PSLIB")
        self._context["fleet_mode"] = fleet_mode

    def which_executor_mode(self):
        executor_mode = envs.get_runtime_environ("train.trainer.executor_mode")
        if executor_mode.upper() not in ["TRAIN", "INFER"]:
            raise ValueError("Not Support Executor Mode {}".format(
                executor_mode))
        if executor_mode.upper() == "TRAIN":
            self.is_infer = False
        else:
            self.is_infer = True
        print("Executor Mode: {}".format(executor_mode))
        self._context["is_infer"] = self.is_infer

    def legality_check(self):
        if self.device == Device.CPU:
            assert self.fleet_mode != FleetMode.COLLECTIVE, "Not Support CPU with Collective Mode"

        if self.is_infer:
            assert self.engine == EngineMode.SINGLE, "Not Support Distributed Infer "

    @abc.abstractmethod
    def processor_register(self):
        pass

    def regist_context_processor(self, status_name, processor):
        """
        regist a processor for specify status
        """
        self._status_processor[status_name] = processor

    def context_process(self, context):
        """
        select a processor to deal specify context
        Args:
            context : context with status
        Return:
            None : run a processor for this status
        """
        status = context['status']
        if status in self._status_processor:
            self._status_processor[context['status']](context)
        else:
            self.other_status_processor(context)

    def other_status_processor(self, context):
        """
        if no processor match context.status, use defalut processor
        Return:
            None, just sleep in base
        """
        print('unknow context_status:%s, do nothing' % context['status'])
        time.sleep(60)

    def handle_processor_exception(self, context, exception):
        """
        when exception throwed from processor, will call this func to handle it 
        Return:
            bool exit_app or not
        """
        print("\n--------------------------------\nPaddleRec Error Message "
              "Summary:\n--------------------------------\n")
        print(
            'Exit PaddleRec. catch exception in precoss status: [%s], except: %s'
            % (context['status'], str(exception)))
        return True

    def reload_train_context(self):
        """
        context maybe update timely, reload for update
        """
        pass

    def run(self):
        """
        keep running by statu context.
        """
        while True:
            try:
                self.reload_train_context()
                self.context_process(self._context)
                if self._context['is_exit']:
                    break
            except Exception as err:
                traceback.print_exc()
                print('Catch Exception:%s' % str(err))
                sys.stdout.flush()
                self.handle_processor_exception(self._context, err)
                sys.exit(type(err).__name__)
