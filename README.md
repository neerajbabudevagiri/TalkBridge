# Real-Time Speech-to-Text & Multilingual Translation System

A real-time, low-latency speech processing system that captures live microphone audio, converts it into English text using Whisper-based models, translates it into Hindi and Telugu, and optionally generates Text-to-Speech (TTS) audio output.

This project demonstrates end-to-end integration of speech recognition, multilingual translation, tokenizer training, dataset preparation, and real-time streaming inference.


Key Features

- Real-time microphone audio streaming
- Speech-to-Text (STT) using Whisper / Faster-Whisper
- Chunk-wise low-latency transcription
- Multilingual translation (English → Hindi, Telugu)
- Optional Text-to-Speech (TTS) output
- Offline and online inference support
- Multiple Whisper model variants (tiny, small, medium)
- Dataset preprocessing and multilingual tokenizer training
- CPU and GPU compatible execution

System Workflow

Microphone Audio  
→ Audio Chunking  
→ Speech-to-Text (Whisper / Faster-Whisper)  
→ Translation (EN → HI / TE)  
→ Optional Text-to-Speech  
→ Logging & Dataset Storage


Repository Structure

├── stt_stream_tiny_auto.py # Real-time STT using Whisper-tiny
├── stt_stream_small_auto.py # Real-time STT using Whisper-small
├── stt_stream_local.py # Offline STT using local Whisper model
├── fast_stream_stt.py # Faster-Whisper continuous streaming
├── speech_translate1.py # STT → Translation → Hindi TTS pipeline
│
├── merge_all.py # Dataset cleaning, merging, splitting (EN–HI, EN–TE)
├── train_tokenizer.py # SentencePiece tokenizer training (BPE)
├── Training.ipynb # Training & experimentation notebook
│
├── tokenizer/
│ ├── spiece.model # Trained SentencePiece model
│ ├── spiece.vocab # Vocabulary file
│ └── special_tokens.txt # Language & special tokens
│
├── outputs/
│ ├── speech_translations.txt # Logged speech and translations
│ ├── hindi_output.mp3 # Sample Hindi TTS output
│ └── telugu_output.mp3 # Sample Telugu TTS output
│
└── README.md


Technologies Used
- Python
- OpenAI Whisper / Faster-Whisper
- SoundDevice (real-time audio capture)
- SentencePiece (BPE tokenizer)
- HuggingFace Hub
- gTTS / Piper TTS
- NumPy




 Installation

1. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

2. Install required packages
pip install numpy sounddevice faster-whisper transformers sentencepiece gtts huggingface_hub tqdm
Make sure your system microphone is enabled and accessible.



How to Run
Real-Time Speech-to-Text (Fastest – Tiny Model)
python stt_stream_tiny_auto.py

Real-Time Speech-to-Text (Small / Medium Models)
python stt_stream_small_auto.py
python fast_stream_stt.py

Offline Whisper (Local Model Execution)
python stt_stream_local.py

Speech → Translation → Hindi TTS Pipeline
python speech_translate1.py

Translated text outputs are saved in:
speech_translations.txt




Dataset & Tokenizer Pipeline
Dataset Preparation

Cleans, normalizes, merges, shuffles, and splits EN–HI and EN–TE datasets into train, validation, and test sets.
python merge_all.py

Tokenizer Training
Trains a shared SentencePiece BPE tokenizer for English, Hindi, and Telugu.
python train_tokenizer.py --vocab_size 32000 --model_type bpe


Language tokens used:
>>en<<
>>hi<<
>>te<<


Sample Output
English : hello world
Hindi   : हैलो दुनिया आप कैसे हैं
Telugu  : హలో వరల్డ్




Applications

Real-time speech translation systems
Assistive and accessibility tools
Multilingual voice interfaces
Speech-enabled NLP research
Edge-device speech processing



Contributors
Pradyumna Kumar – Speech processing, real-time streaming, translation pipeline
Collaborators – Model integration and experimentation




License
This project is intended for academic and research purposes.
