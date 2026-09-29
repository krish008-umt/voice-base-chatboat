Krrish AI: Voice-Enabled Personal Companion with Long-Term Memory

A voice-first AI companion that listens, remembers facts about you across turns, and replies in a personalized way. Built with LangGraph, Whisper, and Qwen2.5-7B-Instruct.

Status: Working CLI prototype (not deployed). Memory is in-process (InMemoryStore) and resets on restart.

Persistent storage is on the roadmap.

How it works

Mic -> Whisper (STT) -> LangGraph:
                          [remember node] -> extract facts (Pydantic structured output),
                                             dedupe vs existing memory, save to store
                          [chat node]     -> personalized streamed reply using memory
                        -> pyttsx3 (TTS) -> Speaker


Speech-to-text: OpenAI Whisper (base) with energy-based silence detection for hands-free turn-taking

Memory extraction: LLM returns a MemoryDecision JSON, validated with Pydantic; malformed output is caught and skipped

Deduplication: new facts are compared against existing memories before saving

Response generation: streamed from Qwen2.5-7B-Instruct via HuggingFace Inference Endpoint

Text-to-speech: pyttsx3 (offline)

Tech stack

Python, LangGraph, LangChain, HuggingFace Inference API, Whisper, Pydantic, sounddevice, pyttsx3

Run it on your machine
Prerequisites
Python 3.10 or 3.11
A working microphone and speakers
FFmpeg installed and on your PATH
Windows: winget install Gyan.FFmpeg
macOS: brew install ffmpeg
Linux: sudo apt install ffmpeg
A free HuggingFace access token: https://huggingface.co/settings/tokens (Read permission is enough)
Setup
bash


python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt

Create a .env file in the project folder (copy from .env.example):

HUGGINGFACEHUB_API_TOKEN=xyz
