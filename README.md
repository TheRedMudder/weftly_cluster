# Weftly - Interest clustering for TikTok
Open-source, privacy-first pipelines that turn your 10,000+ **liked TikToks** into **interpretable interest clusters** you can explore and explain.

## Why Weftly? 
Your liked TikToks give us an indicator to your interest. Weftly "weaves" those seemingly unstructured data into a coherent map of your interest.

## Quickstart

This repo is built to run locally, with **no cloud required**. You can start with 100 videos and scale from there.

1. Download *your* liked TikToks: Use **myfaveTT “Download all your likes”** feature. 
    1. Only login with **your own** account or account you have explicit permission to analyze.
2. Transcribe audio:
    1.  **Extract audio**: Extract mono 16kHz with **FFmpeg** (the Whisper front-end expects 16kHz, 16-bit WAV).
    1. **Run Whisper**: 
        - [Apple Silicon](https://github.com/ggml-org/whisper.cpp) - Use Core ML (ANE) - Runs on Neural Engine, **3X faster** than CPU only. 
            > `scripts/whisper_mac.sh` provides example build commands from following the repo's README.md 
        - [Linux/Windows](https://github.com/SYSTRAN/faster-whisper) - (CTranslate2). It's **up to 4x faster** than orignal Whisper with less memory.
3. Run weftly cluster: 
    1. **Modify Configuration File**: Update `transcription_dir`,`video_dir` in `config.toml` to point to the transcription directory and video directory.
    2. `python cluster_engine.py`

## Post
- [High level overview]()

# Contributing
Issues and PRs are welcome! 

# Python
Recommended: Python 3.12+

# Transcription File Format
All transcription files in the "transcription" folder must follow this format.
```json
// Example: 7529497187273739542.json, Note: The file name is using the video ID: {VideoID}.json Credit: The creator is @maya.mental.fitness, https://www.tiktok.com/@maya.mental.fitness/video/7529497187273739542
[
    {
        "start": 0.64,
        "end": 5.44,
        "text": " seven years of neuroscience and psychology in 60 seconds your brain believes what it repeats"
    },
    {
        "start": 5.44,
        "end": 10.24,
        "text": " not what's true thoughts create your feelings your feelings drive your actions and your actions"
    },
    {
        "start": 10.24,
        "end": 15.120000000000001,
        "text": " create your identity and personality but identity is plastic your brain is constantly rewiring"
    },
    {
        "start": 15.120000000000001,
        "end": 19.44,
        "text": " so when you repeat something with enough intention and emotion you can change your identity self-talk"
    },
    {
        "start": 19.44,
        "end": 23.28,
        "text": " isn't harmless it's like casting a spell your brain and body are listening all the time so"
    },
    {
        "start": 23.28,
        "end": 27.52,
        "text": " the more encouraging helpful useful your self-talk is the stronger relationship you're going to have"
    },
    {
        "start": 27.52,
        "end": 31.68,
        "text": " with yourself your life will always reflect who you believe yourself to be which is why you will"
    },
    {
        "start": 31.68,
        "end": 35.519999999999996,
        "text": " never outperform your self image your brain doesn't know the difference between what is real"
    },
    {
        "start": 35.519999999999996,
        "end": 39.28,
        "text": " and what is imagined when you visualize the person you want to be how you want to act how you want"
    },
    {
        "start": 39.28,
        "end": 44.0,
        "text": " to talk how you want to walk you achieving your goals you are rewiring your brain for success you"
    },
    {
        "start": 44.0,
        "end": 48.239999999999995,
        "text": " don't have thoughts you get visited by them and you don't have to believe every thought you think"
    },
    {
        "start": 48.239999999999995,
        "end": 52.480000000000004,
        "text": " you can't think your way out of a feeling but you can feel your way into a new way of thinking"
    },
    {
        "start": 52.480000000000004,
        "end": 57.36,
        "text": " unprocessed emotions stay stored in the nervous system the body is always keeping score about 95"
    },
    {
        "start": 57.36,
        "end": 57.5,
        "text": " of"
    },
    {
        "start": 57.52,
        "end": 60.88,
        "text": " your behavior is automatic run by the subconscious. Your brain is a prediction"
    },
    {
        "start": 60.88,
        "end": 65.08,
        "text": " machine it constantly guesses what's next based on old data so if you can"
    },
    {
        "start": 65.08,
        "end": 69.84,
        "text": " program that data then you can change your future reality."
    }
]
```