# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any, Dict

from git import Optional
import numpy as np
from PIL import Image
from torch import log_
from gr00t.data.dataset import ModalityConfig
from gr00t.eval.service import BaseInferenceClient, BaseInferenceServer
from gr00t.model.policy import BasePolicy
from gr00t.eval.rerun_vis import InferenceLogger


class RobotInferenceServer(BaseInferenceServer):
    """
    Server with three endpoints for real robot policies
    """

    def __init__(self, model, host: str = "*", port: int = 5555, debug=False):
        super().__init__(host, port)
        self.debug = debug
        self.rerun_logger = InferenceLogger() if debug else None
        self.model = model
        self.register_endpoint("get_action", self.get_action)
        self.register_endpoint(
            "get_modality_config", model.get_modality_config, requires_input=False
        )
    
    def get_action(self, obs):
        if self.debug:
            self.rerun_logger.log(obs)
        return self.model.get_action(obs)

    @staticmethod
    def start_server(policy: BasePolicy, port: int):
        server = RobotInferenceServer(policy, port=port)
        server.run()


class RobotInferenceClient(BaseInferenceClient, BasePolicy):
    """
    Client for communicating with the RealRobotServer
    """

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        return self.call_endpoint("get_action", observations)

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        return self.call_endpoint("get_modality_config", requires_input=False)


class DummyInferenceServer(BaseInferenceServer):
    """
    Server output dataset label as action
    """

    def __init__(self, model, dataset, host: str = "*", port: int = 5555, debug=False):
        super().__init__(host, port)
        self.model = model
        self.dataset = dataset
        self.debug = debug
        self.debug_info = {
            'frame_cnt': 0,
        }
        self.init_cache()
        self.register_endpoint("get_action", self.get_action)
        self.register_endpoint(
            "get_modality_config", model.get_modality_config, requires_input=False
        )
    
    def init_cache(self):
        self.step = 0
        self.act_chunks = []
        for i in range(0, 16*60, 16):
            data = self.dataset.get_step_data(0, i)
            action = {}
            for k, v in data.items():
                if k.startswith('action.'):
                    if v.ndim == 2 and v.shape[-1] == 1:
                        v = np.squeeze(v, -1)
                    action[k] = v
            # breakpoint()
            self.act_chunks.append(action)
    
    def get_action(self, *args):
        
        if self.debug:
            here = os.path.basename(__file__)
            log_dir = os.path.join(here, '..', 'logs')
            os.makedirs(log_dir, exist_ok=True)
            obs = args[0]
            video_frame = obs['video.ego_view'][0][..., :3]
            # video_frame = Image.fromarray(video_frame).convert('RGB')
            video_frame = Image.fromarray(video_frame)
            video_frame.save(os.path.join(log_dir, f'frame_{self.debug_info["frame_cnt"]}.jpg'))
            self.debug_info["frame_cnt"] = (self.debug_info["frame_cnt"] + 1) % 100
            print(obs['state.left_hand'])
            
        action = self.act_chunks[self.step % len(self.act_chunks)]
        self.step += 1
        return action

    @staticmethod
    def start_server(policy: BasePolicy, port: int):
        server = RobotInferenceServer(policy, port=port)
        server.run()