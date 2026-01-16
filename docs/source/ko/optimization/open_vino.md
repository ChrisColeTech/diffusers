<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 추론을 위한 OpenVINO 사용 방법

🤗 [Optimum](https://github.com/huggingface/optimum-intel)은 OpenVINO와 호환되는 Stable Diffusion 파이프라인을 제공합니다.
이제 다양한 Intel 프로세서에서 OpenVINO Runtime으로 쉽게 추론을 수행할 수 있습니다. ([여기](https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html)서 지원되는 전 기기 목록을 확인하세요).

## 설치

다음 명령어로 🤗 Optimum을 설치합니다:

```sh
pip install optimum["openvino"]
```

## Stable Diffusion 추론

OpenVINO 모델을 불러오고 OpenVINO 런타임으로 추론을 실행하려면 `StableDiffusionPipeline`을 `OVStableDiffusionPipeline`으로 교체해야 합니다. PyTorch 모델을 불러오고 즉시 OpenVINO 형식으로 변환하려는 경우 `export=True`로 설정합니다.

```python
from optimum.intel.openvino import OVStableDiffusionPipeline

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = OVStableDiffusionPipeline.from_pretrained(model_id, export=True)
prompt = "a photo of an astronaut riding a horse on mars"
images = pipe(prompt).images[0]
```

[Optimum 문서](https://huggingface.co/docs/optimum/intel/inference#export-and-inference-of-stable-diffusion-models)에서 (정적 reshaping과 모델 컴파일 등의) 더 많은 예시들을 찾을 수 있습니다.
