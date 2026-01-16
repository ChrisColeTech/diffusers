<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Intel Gaudi에서 Stable Diffusion을 사용하는 방법

🤗 Diffusers는 🤗 [Optimum Habana](https://huggingface.co/docs/optimum/habana/usage_guides/stable_diffusion)를 통해서 Habana Gaudi와 호환됩니다.

## 요구 사항

- Optimum Habana 1.4 또는 이후, [여기](https://huggingface.co/docs/optimum/habana/installation)에 설치하는 방법이 있습니다.
- SynapseAI 1.8.


## 추론 파이프라인

Gaudi에서 Stable Diffusion 1 및 2로 이미지를 생성하려면 두 인스턴스를 인스턴스화해야 합니다:
- [`GaudiStableDiffusionPipeline`](https://huggingface.co/docs/optimum/habana/package_reference/stable_diffusion_pipeline)이 포함된 파이프라인. 이 파이프라인은 *텍스트-이미지 생성*을 지원합니다.
- [`GaudiDDIMScheduler`](https://huggingface.co/docs/optimum/habana/package_reference/stable_diffusion_pipeline#optimum.habana.diffusers.GaudiDDIMScheduler)이 포함된 스케줄러. 이 스케줄러는 Habana Gaudi에 최적화되어 있습니다.

파이프라인을 초기화할 때, HPU에 배포하기 위해 `use_habana=True`를 지정해야 합니다.
또한 가능한 가장 빠른 생성을 위해 `use_hpu_graphs=True`로 **HPU 그래프**를 활성화해야 합니다.
마지막으로, [Hugging Face Hub](https://huggingface.co/Habana)에서 다운로드할 수 있는 [Gaudi configuration](https://huggingface.co/docs/optimum/habana/package_reference/gaudi_config)을 지정해야 합니다.

```python
from optimum.habana import GaudiConfig
from optimum.habana.diffusers import GaudiDDIMScheduler, GaudiStableDiffusionPipeline

model_name = "stabilityai/stable-diffusion-2-base"
scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
pipeline = GaudiStableDiffusionPipeline.from_pretrained(
    model_name,
    scheduler=scheduler,
    use_habana=True,
    use_hpu_graphs=True,
    gaudi_config="Habana/stable-diffusion",
)
```

파이프라인을 호출하여 하나 이상의 프롬프트에서 배치별로 이미지를 생성할 수 있습니다.

```python
outputs = pipeline(
    prompt=[
        "High quality photo of an astronaut riding a horse in space",
        "Face of a yellow cat, high resolution, sitting on a park bench",
    ],
    num_images_per_prompt=10,
    batch_size=4,
)
```

더 많은 정보를 얻기 위해, Optimum Habana의 [문서](https://huggingface.co/docs/optimum/habana/usage_guides/stable_diffusion)와 공식 GitHub 저장소에 제공된 [예시](https://github.com/huggingface/optimum-habana/tree/main/examples/stable-diffusion)를 확인하세요.


## 벤치마크

다음은 [Habana/stable-diffusion](https://huggingface.co/Habana/stable-diffusion) Gaudi 구성(혼합 정밀도 bf16/fp32)을 사용하는 Habana first-generation Gaudi 및 Gaudi2의 지연 시간입니다:

|                        | Latency (배치 크기 = 1) | Throughput (배치 크기 = 8) |
| ---------------------- |:------------------------:|:---------------------------:|
| first-generation Gaudi | 4.29s                    | 0.283 images/s              |
| Gaudi2                 | 1.54s                    | 0.904 images/s              |
