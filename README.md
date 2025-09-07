# Weftly - Interest clustering for TikTok
Open-source, privacy-first pipelines that turn your 10,000+ **liked TikToks** into **interpretable interest clusters** you can explore and explain.

## Why Weftly? 
Your liked TikToks give us an indicator to your interest. Weftly "weaves" those seemingly unstructured data into a coherent map of your interest.

## Quickstart

This repo is built to run locally, with **no cloud required**. You can start with 100 videos and scale from there.

### 1) Download *your* liked TikToks

Use **myfaveTT “Download all your likes”** feature. 

> Only login with **your own** account or account you have explicit permission to analyze.


### 2) Transcribe (ASR) locally 
**Extract audio**: Extract mono 16kHz with **FFmpeg** (the Whisper front-end expects 16kHz, 16-bit WAV).

**Run Whisper**: 
- [Apple Silicon](https://github.com/ggml-org/whisper.cpp) - Use Core ML (ANE) - Runs on Neural Engine, **3X faster** than CPU only
- [Linux/Windows](https://github.com/SYSTRAN/faster-whisper) - (CTranslate2). It's **up to 4x faster** than orignal Whisper with less memory.



### 3) Run weftly cluster
`python3 cluster_engine.py`

# Contributing
Issues and PRs are welcome! 

# Python
Recommended: Python 3.12+