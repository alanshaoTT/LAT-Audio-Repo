# Listening with Time: Precise Temporal Awareness for Long-Form Audio Understanding


<p align="center">
  Mingchen Shao<sup>1</sup>, 
  Hang Su<sup>2</sup>, 
  Wenjie Tian<sup>1</sup>, 
  Bingshen Mu<sup>1</sup>, 
  Zhennan Lin<sup>1</sup>, 
  Lichun Fan<sup>2</sup>, 
  Zhenbo Luo<sup>2</sup>, 
  Jian Luan<sup>2</sup>, 
  Lei Xie<sup>1</sup><sup>†</sup>, 
</p>

<p align="center">
  <sup>1</sup> Audio, Speech and Language Processing Group (ASLP@NPU), Northwestern Polytechnical University <br>
  <sup>2</sup> Independent Researcher <br>
</p>

<!-- 📑 <a href="https://arxiv.org/abs/2601.11027">Paper</a> &nbsp&nbsp | &nbsp&nbsp  -->
<p align="center">
🐙 <a href="https://github.com/alanshaoTT/LAT-Audio-Repo">GitHub</a> &nbsp&nbsp | &nbsp&nbsp 
🤗 <a href="https://huggingface.co/collections/mcshao/lat-audio">HuggingFace</a> &nbsp&nbsp | &nbsp&nbsp 
<!-- 🖥️ <a href="">HuggingFace Space</a> &nbsp&nbsp | &nbsp&nbsp  -->
💬 <a href="https://github.com/alanshaoTT/LAT-Audio-Repo?tab=readme-ov-file#contact">Contact Us</a>
</p>

## 🚀 Overview

Large Audio Language Models (LALMs) perform well on short audio but struggle with long-form audio due to **temporal hallucination** and **timestamp drift**.

We propose **LAT-Audio**, a global-to-local reasoning framework that enables precise temporal awareness through:
- A **global timeline** for structured temporal understanding
- A **Think-With-Audio Chain-of-Thought (TWA-CoT)** for iterative reasoning
- A **tool-augmented mechanism** to retrieve local audio evidence

We further introduce:
- **LAT-Chronicle**: a 1.2k-hour long-form audio dataset
- **LAT-Bench**: the first human-verified long-form temporal benchmark

## 🧠 Framework

<p align="center">
  <img src="figs/lat-audio.png" width="900"/>

</p>

LAT-Audio follows a **progressive global-to-local reasoning paradigm**:
1. Construct a global timeline as temporal-semantic anchors
2. Perform multi-step reasoning via TWA-CoT
3. Iteratively retrieve audio evidence through tool calls

## 📊 Dataset & Benchmark

| Component | Description |
|----------|------------|
| **LAT-Chronicle** | 1.2k-hour long-form dataset (1k zh / 200h en) |
| **LAT-Bench** | Human-verified benchmark (DAC, TAG, TAC) |

👉 Download:
- 🤗 LAT-Chronicle: ...
- 🤗 LAT-Bench: ...

<p align="center">
  <img src="figs/pipeline.png" width="900"/>

</p>

## Download
* The LAT-Chronicle dataset are available at [LAT-Chronicle](https://huggingface.co/datasets/mcshao/LAT-Chronicle).
* The LAT-Bench benchmark are available at [LAT-Bench](https://huggingface.co/datasets/mcshao/LAT-Bench).
* The LAT-Audio model are available at [LAT-Audio](https://huggingface.co/mcshao/LAT-Audio).
* The LAT-Audio-Base model are available at [LAT-Audio-Base](https://huggingface.co/mcshao/LAT-Audio-Base).

## Contact

If you are interested in leaving a message to our research team, feel free to email mcshao@mail.nwpu.edu.cn .
