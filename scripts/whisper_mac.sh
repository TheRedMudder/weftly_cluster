# NOTE: Refer to repo for latest instructions: https://github.com/ggml-org/whisper.cpp This script is me following the README on 9/7/25. 
# IMPORTANT: For Apple Silicon only! For Windows/Linux Follow: https://github.com/SYSTRAN/faster-whisper 
# GOOD PRACTICE: Please use a virtual environment. Ex:conda create --name weftly_ingest python=3.11
# clone
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp
# download models
./models/download-ggml-model.sh large-v3
./models/download-vad-model.sh silero-v5.1.2
# install 
pip install torch==2.5.0 pip-system-certs ane_transformers coremltools openai-whisper -U
# large model
./models/generate-coreml-model.sh large-v3
cmake -B build -DWHISPER_COREML=1 && cmake --build build -j --config Release
# run example
./build/bin/whisper-cli -m models/ggml-large-v3.bin -f samples/jfk.wav