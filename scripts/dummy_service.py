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

import argparse

import numpy as np

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer, DummyInferenceServer
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoint directory.",
        default="nvidia/GR00T-N1-2B",
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="gr1",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        help="The name of the data config to use.",
        choices=list(DATA_CONFIG_MAP.keys()),
        default="gr1_arms_waist",
    )

    parser.add_argument("--video_backend", type=str, default="torchvision_av")
    parser.add_argument("--dataset_path", type=str, default="demo_data/robot_sim.PickNPlace/")
    parser.add_argument("--port", type=int, help="Port number for the server.", default=5555)
    parser.add_argument(
        "--host", type=str, help="Host address for the server.", default="localhost"
    )
    # server mode
    parser.add_argument("--server", action="store_true", help="Run the server.")
    # client mode
    parser.add_argument("--client", action="store_true", help="Run the client")
    parser.add_argument("--denoising_steps", type=int, help="Number of denoising steps.", default=4)
    args = parser.parse_args()

    if args.server:
        # Create a policy
        # The `Gr00tPolicy` class is being used to create a policy object that encapsulates
        # the model path, transform name, embodiment tag, and denoising steps for the robot
        # inference system. This policy object is then utilized in the server mode to start
        # the Robot Inference Server for making predictions based on the specified model and
        # configuration.

        # we will use an existing data config to create the modality config and transform
        # if a new data config is specified, this expect user to
        # construct your own modality config and transform
        # see gr00t/utils/data.py for more details
        data_config = DATA_CONFIG_MAP[args.data_config]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
        )
        modality = policy.get_modality_config()
        dataset = LeRobotSingleDataset(
            dataset_path=args.dataset_path,
            modality_configs=modality,
            video_backend=args.video_backend,
            video_backend_kwargs=None,
            transforms=None,  # We'll handle transforms separately through the policy
            embodiment_tag=args.embodiment_tag,
        )

        # Start the server
        server = DummyInferenceServer(policy, dataset, port=args.port)
        server.run()

    elif args.client:
        import time

        # In this mode, we will send a random observation to the server and get an action back
        # This is useful for testing the server and client connection
        # Create a policy wrapper
        policy_client = RobotInferenceClient(host=args.host, port=args.port)

        print("Available modality config available:")
        modality_configs = policy_client.get_modality_config()
        print(modality_configs.keys())

        # Making prediction...
        # - obs: video.ego_view: (1, 256, 256, 3)
        # - obs: state.left_arm: (1, 7)
        # - obs: state.right_arm: (1, 7)
        # - obs: state.left_hand: (1, 6)
        # - obs: state.right_hand: (1, 6)
        # - obs: state.waist: (1, 3)

        # - action: action.left_arm: (16, 7)
        # - action: action.right_arm: (16, 7)
        # - action: action.left_hand: (16, 6)
        # - action: action.right_hand: (16, 6)
        # - action: action.waist: (16, 3)
        if args.embodiment_tag == 'gr1':
            obs = {
                "video.ego_view": np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
                "state.left_arm": np.random.rand(1, 7),
                "state.right_arm": np.random.rand(1, 7),
                "state.left_hand": np.random.rand(1, 6),
                "state.right_hand": np.random.rand(1, 6),
                "state.waist": np.random.rand(1, 3),
                "annotation.human.action.task_description": ["do your thing!"],
            }
        else:
            """
            python3 scripts/inference_service.py --server --data_config g1_stack_block_inference --embodiment_tag new_embodiment --model_path ~/Downloads/GR00T/checkpoint/
            python3 scripts/inference_service.py --client --data_config g1_stack_block --embodiment_tag new_embodiment
            """
            obs = {
                "video.ego_view": np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8),
                "state.left_shoulder": np.random.rand(1, 3),
                "state.right_shoulder": np.random.rand(1, 3),
                "state.left_elbow": np.random.rand(1, 1),
                "state.right_elbow": np.random.rand(1, 1),
                "state.left_wrist": np.random.rand(1, 3),
                "state.right_wrist": np.random.rand(1, 3),
                "state.left_hand": np.random.rand(1, 7),
                "state.right_hand": np.random.rand(1, 7),
                
                # "action.left_shoulder": np.random.rand(1, 3),
                # "action.right_shoulder": np.random.rand(1, 3),
                # "action.left_elbow": np.random.rand(1, 1),
                # "action.right_elbow": np.random.rand(1, 1),
                # "action.left_wrist": np.random.rand(1, 3),
                # "action.right_wrist": np.random.rand(1, 3),
                # "action.left_hand": np.random.rand(1, 7),
                # "action.right_hand": np.random.rand(1, 7),
            }

        time_start = time.time()
        action = policy_client.get_action(obs)
        print(f"Total time taken to get action from server: {time.time() - time_start} seconds")
        print('Keys: ', action.keys())

        for key, value in action.items():
            print(f"Action: {key}: {value.shape}, {value.dtype}")
            print(value)

    else:
        raise ValueError("Please specify either --server or --client")
