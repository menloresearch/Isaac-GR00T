from math import tau
from typing import *

import rerun as rr
import numpy as np
from rerun import blueprint as rrb
from rerun.utilities import build_color_spiral
from rerun.utilities import bounce_lerp


XYZ_AXIS_COLORS = [[(231, 76, 60), (39, 174, 96), (52, 120, 219)]]


def sample():
    rr.init("rerun_example_dna_abacus")
    rr.spawn()
    rr.set_time("stable_time", duration=0)

    NUM_POINTS = 100

    # Points and colors are both np.array((NUM_POINTS, 3))
    points1, colors1 = build_color_spiral(NUM_POINTS)
    points2, colors2 = build_color_spiral(NUM_POINTS, angular_offset=tau*0.5)

    rr.log("dna/structure/left", rr.Points3D(points1, colors=colors1, radii=0.08))
    rr.log("dna/structure/right", rr.Points3D(points2, colors=colors2, radii=0.08))

    rr.log(
        "dna/structure/scaffolding",
        rr.LineStrips3D(np.stack((points1, points2), axis=1), colors=[128, 128, 128])
    )

    offsets = np.random.rand(NUM_POINTS)
    beads = [bounce_lerp(points1[n], points2[n], offsets[n]) for n in range(NUM_POINTS)]
    colors = [[int(bounce_lerp(80, 230, offsets[n] * 2))] for n in range(NUM_POINTS)]
    rr.log(
        "dna/structure/scaffolding/beads",
        rr.Points3D(beads, radii=0.06, colors=np.repeat(colors, 3, axis=-1)),
    )


    time_offsets = np.random.rand(NUM_POINTS)

    for i in range(400):
        time = i * 0.01
        rr.set_time("stable_time", duration=time)

        times = np.repeat(time, NUM_POINTS) + time_offsets
        beads = [bounce_lerp(points1[n], points2[n], times[n]) for n in range(NUM_POINTS)]
        colors = [[int(bounce_lerp(80, 230, times[n] * 2))] for n in range(NUM_POINTS)]
        rr.log(
            "dna/structure/scaffolding/beads",
            rr.Points3D(beads, radii=0.06, colors=np.repeat(colors, 3, axis=-1)),
        )


class InferenceLogger:

    def __init__(self, name="rerun_gr00t_n1.5_inference_request"):
        self.timeline = 'request_t'
        self.step = 0
        rr.init(
            name, 
            default_blueprint=None, 
            spawn=True
        )
        rr.send_blueprint(self.build_blueprint())
        rr.set_time(self.timeline, timestamp=0)
    
    def build_blueprint(self):
        right_panels = [
            "state.left_shoulder",
            "state.right_shoulder",
            "state.left_elbow",
            "state.right_elbow",
        ]
        left_panels = [
            "state.left_wrist",
            "state.right_wrist",
            "state.left_hand",
            "state.right_hand",
        ]
        blueprint = rrb.Horizontal(
            rrb.Vertical(
                *[
                    rrb.TimeSeriesView(
                        origin=name.replace('.', '/'),
                        name=name,
                    )
                    for name in left_panels
                ]
            ),
            rrb.Vertical(
                rrb.Spatial2DView(origin="video/ego_view"),
                *[
                    rrb.TimeSeriesView(
                        origin=name.replace('.', '/'),
                        name=name,
                    )
                    for name in right_panels
                ]
            ),
            # rrb.Spatial3DView(origin="/", name="World position"),
            column_shares=[0.60, 0.40],
        )
        return blueprint

    def debug_log(self, observation: Dict[str, np.ndarray], step=None):
        if step is not None:
            assert step > self.step
            self.step = step
        else:
            self.step += 1
        
        rr.set_time(self.timeline, timestamp=self.step)
        # rr.log("join_a", )
        times = rr.TimeColumn("timestamp", timestamp=[self.step] * 3)
        # values = np.zeros([3])
        values = np.random.uniform(0, 1, [3])
        # rr.send_columns("/left/joint_a", indexes=[times], columns=rr.Scalars.columns(scalars=values))
        # rr.send_columns("/left/joint_a", timesteps=[self.step]*3, columns=rr.Scalars.columns(scalars=values))
        # rr.send_columns("/left/joint_b", indexes=[times], columns=rr.Scalars.columns(scalars=values))
        # rr.send_columns("/joint_c", indexes=[times], columns=rr.Scalars.columns(scalars=values))
        # rr.send_columns("/joint_d", indexes=[times], columns=rr.Scalars.columns(scalars=values))
        rr.log('/left/joint_a/a', rr.Scalars(values[0]))
        rr.log('/left/joint_a/b', rr.Scalars(values[1]))
        rr.log('/left/joint_a/c', rr.Scalars(values[2]))
        
        rr.log('/left/joint_b/', rr.Scalars(values))
        # rr.log('/left/joint_b/b', rr.Scalars(values[1]))
        # rr.log('/left/joint_b/c', rr.Scalars(values[2]))
    
    def log(self, observation: Dict[str, np.ndarray], step=None):
        if step is not None:
            assert step > self.step
            self.step = step
        else:
            self.step += 1
        
        print('rerun log', observation.keys())
        rr.set_time(self.timeline, timestamp=self.step)
        for k, v in observation.items():
            if 'video.' in k:
                rr.log(
                    k.replace('.', '/'), 
                    rr.Image(image=np.squeeze(v, axis=0), color_model=rr.ColorModel.RGB).compress(),
                )
            else:
                rr.log(k.replace('.', '/'), rr.Scalars(np.squeeze(v, axis=0)))


if __name__ == "__main__":
    logger = InferenceLogger()
    for _ in range(5):
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
        logger.log(obs)