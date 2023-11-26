# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

from __future__ import annotations

from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import MultiStepSchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig

from bakedangelo.fields.sdf_field import SDFFieldConfig

from bakedangelo.engine.schedulers  import MultiStepWarmupSchedulerConfig 
from bakedangelo.models.bakedangelo import BakedAngeloModelConfig


bakedangelo = MethodSpecification(
    config=TrainerConfig(
        method_name="bakedangelo",
        steps_per_eval_image=5000,
        steps_per_eval_batch=5000,
        steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
        max_num_iterations=1000_001,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=SDFStudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=BakedAngeloModelConfig(
                near_plane=0.01,
                far_plane=1000.0,
                overwrite_near_far_plane=True,
                sdf_field=SDFFieldConfig(
                    use_grid_feature=True,
                    num_layers=1,
                    num_layers_color=4,
                    hidden_dim=256,
                    hidden_dim_color=256,
                    geometric_init=True,
                    bias=1.5,
                    beta_init=0.1,
                    inside_outside=True,
                    use_appearance_embedding=True,
                    use_numerical_gradients=True,
                    base_res=64,
                    max_res=4096,
                    log2_hashmap_size=22,
                    features_per_level=8,
                    smoothstep=False,
                    use_position_encoding=False,
                ),
                eikonal_loss_mult=0.01,
                background_model="grid",
                proposal_weights_anneal_max_num_iters=10000,
                use_anneal_beta=True,
                eval_num_rays_per_chunk=1024,
                use_spatial_varying_eikonal_loss=False,
                steps_per_level=10_000,
                curvature_loss_warmup_steps=20_000,
                beta_anneal_end=0.0002,
                beta_anneal_max_num_iters=1000_000,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": MultiStepSchedulerConfig(max_steps=1000_000),
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15, weight_decay=1e-2),
                "scheduler": MultiStepWarmupSchedulerConfig(warm_up_end=5000, milestones=[600_000, 800_000], gamma=0.1),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": MultiStepWarmupSchedulerConfig(warm_up_end=5000, milestones=[300_000, 400_000], gamma=0.1),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Implementation of Neuralangelo with BakedSDF",
)