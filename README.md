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


TECHNOLOGIES USED

- Python
- OpenAI Whisper / Faster-Whisper
- SoundDevice (real-time audio capture)
- SentencePiece (BPE tokenizer)
- HuggingFace Hub
- gTTS / Piper TTS
- NumPy

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

- Devagiri Neeraj Babu – Speech processing, real-time streaming, translation pipeline
- Pradyumna Kumar – Model integration and experimentation

--------------------------------------------------------------------

LICENSE

This project is intended for academic and research purposes.
```
