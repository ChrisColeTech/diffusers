<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 파일들을 Hub로 푸시하기

[[open-in-colab]]

🤗 Diffusers는 모델, 스케줄러 또는 파이프라인을 Hub에 업로드할 수 있는 [`~diffusers.utils.PushToHubMixin`]을 제공합니다. 이는 Hub에 당신의 파일을 저장하는 쉬운 방법이며, 다른 사람들과 작업을 공유할 수도 있습니다. 실제적으로 [`~diffusers.utils.PushToHubMixin`]가 동작하는 방식은 다음과 같습니다:

1. Hub에 리포지토리를 생성합니다.
2. 나중에 다시 불러올 수 있도록 모델, 스케줄러 또는 파이프라인 파일을 저장합니다.
3. 이러한 파일이 포함된 폴더를 Hub에 업로드합니다.

이 가이드는 [`~diffusers.utils.PushToHubMixin`]을 사용하여 Hub에 파일을 업로드하는 방법을 보여줍니다.

먼저 액세스 [토큰](https://huggingface.co/settings/tokens)으로 Hub 계정에 로그인해야 합니다:

```py
from huggingface_hub import notebook_login

notebook_login()
```

## 모델

모델을 허브에 푸시하려면 [`~diffusers.utils.PushToHubMixin.push_to_hub`]를 호출하고 Hub에 저장할 모델의 리포지토리 id를 지정합니다:

```py
from diffusers import ControlNetModel

controlnet = ControlNetModel(
    block_out_channels=(32, 64),
    layers_per_block=2,
    in_channels=4,
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
    cross_attention_dim=32,
    conditioning_embedding_out_channels=(16, 32),
)
controlnet.push_to_hub("my-controlnet-model")
```

모델의 경우 Hub에 푸시할 가중치의 [*변형*](loading#checkpoint-variants)을 지정할 수도 있습니다. 예를 들어, `fp16` 가중치를 푸시하려면 다음과 같이 하세요:

```py
controlnet.push_to_hub("my-controlnet-model", variant="fp16")
```

[`~diffusers.utils.PushToHubMixin.push_to_hub`] 함수는 모델의 `config.json` 파일을 저장하고 가중치는 `safetensors` 형식으로 자동으로 저장됩니다.

이제 Hub의 리포지토리에서 모델을 다시 불러올 수 있습니다:

```py
model = ControlNetModel.from_pretrained("your-namespace/my-controlnet-model")
```

## 스케줄러

스케줄러를 허브에 푸시하려면 [`~diffusers.utils.PushToHubMixin.push_to_hub`]를 호출하고 Hub에 저장할 스케줄러의 리포지토리 id를 지정합니다:

```py
from diffusers import DDIMScheduler

scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
)
scheduler.push_to_hub("my-controlnet-scheduler")
```

[`~diffusers.utils.PushToHubMixin.push_to_hub`] 함수는 스케줄러의 `scheduler_config.json` 파일을 지정된 리포지토리에 저장합니다.

이제 허브의 리포지토리에서 스케줄러를 다시 불러올 수 있습니다:

```py
scheduler = DDIMScheduler.from_pretrained("your-namepsace/my-controlnet-scheduler")
```

## 파이프라인

모든 컴포넌트가 포함된 전체 파이프라인을 Hub로 푸시할 수도 있습니다. 예를 들어, 원하는 파라미터로 [`StableDiffusionPipeline`]의 컴포넌트들을 초기화합니다:

```py
from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionPipeline,
)
from transformers import CLIPTextModel, CLIPTextConfig, CLIPTokenizer

unet = UNet2DConditionModel(
    block_out_channels=(32, 64),
    layers_per_block=2,
    sample_size=32,
    in_channels=4,
    out_channels=4,
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
    up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
    cross_attention_dim=32,
)

scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
)

vae = AutoencoderKL(
    block_out_channels=[32, 64],
    in_channels=3,
    out_channels=3,
    down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
    up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
    latent_channels=4,
)

text_encoder_config = CLIPTextConfig(
    bos_token_id=0,
    eos_token_id=2,
    hidden_size=32,
    intermediate_size=37,
    layer_norm_eps=1e-05,
    num_attention_heads=4,
    num_hidden_layers=5,
    pad_token_id=1,
    vocab_size=1000,
)
text_encoder = CLIPTextModel(text_encoder_config)
tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
```

모든 컴포넌트들을 [`StableDiffusionPipeline`]에 전달하고 [`~diffusers.utils.PushToHubMixin.push_to_hub`]를 호출하여 파이프라인을 Hub로 푸시합니다:

```py
components = {
    "unet": unet,
    "scheduler": scheduler,
    "vae": vae,
    "text_encoder": text_encoder,
    "tokenizer": tokenizer,
    "safety_checker": None,
    "feature_extractor": None,
}

pipeline = StableDiffusionPipeline(**components)
pipeline.push_to_hub("my-pipeline")
```

[`~diffusers.utils.PushToHubMixin.push_to_hub`] 함수는 각 컴포넌트를 리포지토리의 하위 폴더에 저장합니다. 이제 Hub의 리포지토리에서 파이프라인을 다시 불러올 수 있습니다:

```py
pipeline = StableDiffusionPipeline.from_pretrained("your-namespace/my-pipeline")
```

## 비공개

모델, 스케줄러 또는 파이프라인 파일들을 비공개로 두려면 [`~diffusers.utils.PushToHubMixin.push_to_hub`] 함수에서 `private=True`를 설정하세요:

```py
controlnet.push_to_hub("my-controlnet-model-private", private=True)
```

비공개 리포지토리는 본인만 볼 수 있으며 다른 사용자는 리포지토리를 복제할 수 없고 리포지토리가 검색 결과에 표시되지 않습니다. 사용자가 비공개 리포지토리의 URL을 가지고 있더라도 `404 - Sorry, we can't find the page you are looking for`라는 메시지가 표시됩니다. 비공개 리포지토리에서 모델을 로드하려면 [로그인](https://huggingface.co/docs/huggingface_hub/quick-start#login) 상태여야 합니다.