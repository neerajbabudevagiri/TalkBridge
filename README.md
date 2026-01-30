```text
# Real-Time Speech-to-Text & Multilingual Translation System

A real-time, low-latency speech processing system that captures live microphone audio,
converts it into English text using Whisper-based models, translates it into Hindi and Telugu,
and optionally generates Text-to-Speech (TTS) audio output.

This project demonstrates end-to-end integration of speech recognition, multilingual
translation, tokenizer training, dataset preparation, and real-time streaming inference.

--------------------------------------------------------------------

FEATURES

- Real-time microphone audio streaming
- Speech-to-Text (STT) using Whisper / Faster-Whisper
- Chunk-wise low-latency transcription
- Multilingual translation (English → Hindi, Telugu)
- Optional Text-to-Speech (TTS) output
- Offline and online inference support
- Multiple Whisper model variants (tiny, small, medium)
- Dataset preprocessing and multilingual tokenizer training
- CPU and GPU compatible execution

--------------------------------------------------------------------

SYSTEM WORKFLOW

Microphone Audio
→ Audio Chunking
→ Speech-to-Text (Whisper / Faster-Whisper)
→ Translation (EN → HI / TE)
→ Optional Text-to-Speech
→ Logging & Dataset Storage

--------------------------------------------------------------------

REPOSITORY STRUCTURE

.
├── stt_stream_tiny_auto.py        # Real-time STT using Whisper-tiny (auto-download)
├── stt_stream_small_auto.py       # Real-time STT using Whisper-small
├── stt_stream_local.py            # Offline STT using local Whisper model
├── fast_stream_stt.py             # Faster-Whisper continuous streaming
├── speech_translate1.py           # Speech → Translation → Hindi TTS pipeline
│
├── merge_all.py                   # Dataset cleaning, merging, splitting (EN–HI, EN–TE)
├── train_tokenizer.py             # SentencePiece tokenizer training (BPE)
├── Training.ipynb                 # Training & experimentation notebook
│
├── tokenizer/
│   ├── spiece.model               # Trained SentencePiece model
│   ├── spiece.vocab               # SentencePiece vocabulary
│   └── special_tokens.txt         # Language & special tokens
│
├── outputs/
│   ├── speech_translations.txt    # Logged speech and translations
│   ├── hindi_output.mp3           # Sample Hindi TTS output
│   └── telugu_output.mp3          # Sample Telugu TTS output
│
└── README.md

--------------------------------------------------------------------

TECHNOLOGIES USED

- Python
- OpenAI Whisper / Faster-Whisper
- SoundDevice (real-time audio capture)
- SentencePiece (BPE tokenizer)
- HuggingFace Hub
- gTTS / Piper TTS
- NumPy

--------------------------------------------------------------------

INSTALLATION

1. Create a virtual environment (recommended)

   python -m venv venv
   source venv/bin/activate        # Linux / macOS
   venv\Scripts\activate           # Windows

2. Install required packages

   pip install numpy sounddevice faster-whisper transformers sentencepiece gtts huggingface_hub tqdm

Make sure your system microphone is enabled and accessible.

--------------------------------------------------------------------

HOW TO RUN

Real-Time Speech-to-Text (Fastest – Tiny Model)
   python stt_stream_tiny_auto.py

Real-Time Speech-to-Text (Small / Medium Models)
   python stt_stream_small_auto.py
   python fast_stream_stt.py

Offline Whisper (Local Model Execution)
   python stt_stream_local.py

Speech → Translation → Hindi TTS Pipeline
   python speech_translate1.py

Translated outputs are saved in:
   speech_translations.txt

--------------------------------------------------------------------

DATASET & TOKENIZER PIPELINE

Dataset Preparation
   python merge_all.py

Tokenizer Training
   python train_tokenizer.py --vocab_size 32000 --model_type bpe

Language tokens used:
   >>en<<
   >>hi<<
   >>te<<

--------------------------------------------------------------------

SAMPLE OUTPUT

English : hello world
Hindi   : हैलो दुनिया आप कैसे हैं
Telugu  : హలో వరల్డ్

--------------------------------------------------------------------

APPLICATIONS

- Real-time speech translation systems
- Assistive and accessibility tools
- Multilingual voice interfaces
- Speech-enabled NLP research
- Edge-device speech processing

--------------------------------------------------------------------

CONTRIBUTORS

- Pradyumna Kumar – Speech processing, real-time streaming, translation pipeline
- Collaborators – Model integration and experimentation

--------------------------------------------------------------------

LICENSE

This project is intended for academic and research purposes.
```
