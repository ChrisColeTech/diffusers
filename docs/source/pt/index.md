<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/diffusers/77aadfee6a891ab9fcfb780f87c693f7a5beeb8e/docs/source/imgs/diffusers_library.jpg" width="400"/>
    <br>
</p>

# Diffusers

🤗 Diffusers é uma biblioteca de modelos de difusão de última geração para geração de imagens, áudio e até mesmo estruturas 3D de moléculas. Se você está procurando uma solução de geração simples ou quer treinar seu próprio modelo de difusão, 🤗 Diffusers é uma caixa de ferramentas modular que suporta ambos. Nossa biblioteca é desenhada com foco em [usabilidade em vez de desempenho](conceptual/philosophy#usability-over-performance), [simples em vez de fácil](conceptual/philosophy#simple-over-easy) e [customizável em vez de abstrações](conceptual/philosophy#tweakable-contributorfriendly-over-abstraction).

A Biblioteca tem três componentes principais:

- Pipelines de última geração para a geração em poucas linhas de código. Há muitos pipelines no 🤗 Diffusers, veja a tabela no pipeline [Visão geral](api/pipelines/overview) para uma lista completa de pipelines disponíveis e as tarefas que eles resolvem.
- Intercambiáveis [agendadores de ruído](api/schedulers/overview) para balancear as compensações entre velocidade e qualidade de geração.
- [Modelos](api/models) pré-treinados que podem ser usados como se fossem blocos de construção, e combinados com agendadores, para criar seu próprio sistema de difusão de ponta a ponta.

<div class="mt-10">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-2 md:gap-y-4 md:gap-x-5">
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./tutorials/tutorial_overview"
      ><div class="w-full text-center bg-gradient-to-br from-blue-400 to-blue-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">Tutoriais</div>
      <p class="text-gray-700">Aprenda as competências fundamentais que precisa para iniciar a gerar saídas, construa seu próprio sistema de difusão, e treine um modelo de difusão. Nós recomendamos começar por aqui se você está utilizando o 🤗 Diffusers pela primeira vez!</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./using-diffusers/loading_overview"
      ><div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">Guias de utilização</div>
      <p class="text-gray-700">Guias práticos para ajudar você carregar pipelines, modelos, e agendadores. Você também aprenderá como usar os pipelines para tarefas específicas, controlar como as saídas são geradas, otimizar a velocidade de geração, e outras técnicas diferentes de treinamento.</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./conceptual/philosophy"
      ><div class="w-full text-center bg-gradient-to-br from-pink-400 to-pink-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">Guias conceituais</div>
      <p class="text-gray-700">Compreenda porque a biblioteca foi desenhada da forma que ela é, e aprenda mais sobre as diretrizes éticas e implementações de segurança para o uso da biblioteca.</p>
   </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./api/models/overview"
      ><div class="w-full text-center bg-gradient-to-br from-purple-400 to-purple-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">Referência</div>
      <p class="text-gray-700">Descrições técnicas de como funcionam as classes e métodos do 🤗 Diffusers</p>
    </a>
  </div>
</div>
