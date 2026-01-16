<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Unconditional 이미지 생성

unconditional 이미지 생성은 text-to-image 또는 image-to-image 모델과 달리 텍스트나 이미지에 대한 조건이 없이 학습 데이터 분포와 유사한 이미지만을 생성합니다.

<iframe
	src="https://stevhliu-ddpm-butterflies-128.hf.space"
	frameborder="0"
	width="850"
	height="550"
></iframe>


이 가이드에서는 기존에 존재하던 데이터셋과 자신만의 커스텀 데이터셋에 대해 unconditional image generation 모델을 훈련하는 방법을 설명합니다. 훈련 세부 사항에 대해 더 자세히 알고 싶다면 unconditional image generation을 위한 모든 학습 스크립트를 [여기](https://github.com/huggingface/diffusers/tree/main/examples/unconditional_image_generation)에서 확인할 수 있습니다.

스크립트를 실행하기 전, 먼저 의존성 라이브러리들을 설치해야 합니다.

```bash
pip install diffusers[training] accelerate datasets
```

그 다음 🤗 [Accelerate](https://github.com/huggingface/accelerate/) 환경을 초기화합니다.

```bash
accelerate config
```

별도의 설정 없이 기본 설정으로 🤗 [Accelerate](https://github.com/huggingface/accelerate/) 환경을 초기화해봅시다.

```bash
accelerate config default
```

노트북과 같은 대화형 쉘을 지원하지 않는 환경의 경우, 다음과 같이 사용해볼 수도 있습니다.

```py
from accelerate.utils import write_basic_config

write_basic_config()
```

## 모델을 허브에 업로드하기

학습 스크립트에 다음 인자를 추가하여 허브에 모델을 업로드할 수 있습니다.

```bash
--push_to_hub
```

## 체크포인트 저장하고 불러오기

훈련 중 문제가 발생할 경우를 대비하여 체크포인트를 정기적으로 저장하는 것이 좋습니다. 체크포인트를 저장하려면 학습 스크립트에 다음 인자를 전달합니다:

```bash
--checkpointing_steps=500
```

전체 훈련 상태는 500스텝마다 `output_dir`의 하위 폴더에 저장되며, 학습 스크립트에 `--resume_from_checkpoint` 인자를 전달함으로써 체크포인트를 불러오고 훈련을 재개할 수 있습니다.

```bash
--resume_from_checkpoint="checkpoint-1500"
```

## 파인튜닝

이제 학습 스크립트를 시작할 준비가 되었습니다! `--dataset_name` 인자에 파인튜닝할 데이터셋 이름을 지정한 다음, `--output_dir` 인자에 지정된 경로로 저장합니다. 본인만의 데이터셋를 사용하려면, [학습용 데이터셋 만들기](create_dataset) 가이드를 참조하세요.

학습 스크립트는 `diffusion_pytorch_model.bin` 파일을 생성하고, 그것을 당신의 리포지토리에 저장합니다.

> [!TIP]
> 💡 전체 학습은 V100 GPU 4개를 사용할 경우, 2시간이 소요됩니다.

예를 들어, [Oxford Flowers](https://huggingface.co/datasets/huggan/flowers-102-categories) 데이터셋을 사용해 파인튜닝할 경우:

```bash
accelerate launch train_unconditional.py \
  --dataset_name="huggan/flowers-102-categories" \
  --resolution=64 \
  --output_dir="ddpm-ema-flowers-64" \
  --train_batch_size=16 \
  --num_epochs=100 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \
  --push_to_hub
```

<div class="flex justify-center">
    <img src="https://user-images.githubusercontent.com/26864830/180248660-a0b143d0-b89a-42c5-8656-2ebf6ece7e52.png"/>
</div>
[Naruto](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions) 데이터셋을 사용할 경우:

```bash
accelerate launch train_unconditional.py \
  --dataset_name="lambdalabs/naruto-blip-captions" \
  --resolution=64 \
  --output_dir="ddpm-ema-naruto-64" \
  --train_batch_size=16 \
  --num_epochs=100 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \
  --push_to_hub
```

<div class="flex justify-center">
    <img src="https://user-images.githubusercontent.com/26864830/180248200-928953b4-db38-48db-b0c6-8b740fe6786f.png"/>
</div>

### 여러개의 GPU로 훈련하기

`accelerate`을 사용하면 원활한 다중 GPU 훈련이 가능합니다. `accelerate`을 사용하여 분산 훈련을 실행하려면 [여기](https://huggingface.co/docs/accelerate/basic_tutorials/launch) 지침을 따르세요. 다음은 명령어 예제입니다.

```bash
accelerate launch --mixed_precision="fp16" --multi_gpu train_unconditional.py \
  --dataset_name="lambdalabs/naruto-blip-captions" \
  --resolution=64 --center_crop --random_flip \
  --output_dir="ddpm-ema-naruto-64" \
  --train_batch_size=16 \
  --num_epochs=100 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision="fp16" \
  --logger="wandb" \
  --push_to_hub
```
