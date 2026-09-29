import os
import uuid
from typing import Annotated, List
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import whisper
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import pyttsx3

from langchain_core.messages import SystemMessage, HumanMessage, AIMessageChunk
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langgraph.graph import START, END, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import PydanticOutputParser


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
ffmpeg_path = r"C:\Users\Dell\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin"
if os.path.exists(ffmpeg_path):
    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]


class mystate(TypedDict):
    message: Annotated[list, add_messages]


class MemoryItem(BaseModel):
    is_new_info: bool = Field(description="True if this info is new and not present in memory, otherwise False")
    text: str = Field(description="The information to be added in memory")

class MemoryDecision(BaseModel):
    is_add: bool = Field(description="True if any info required to save in memory otherwise False")
    memories: List[MemoryItem] = Field(description="List of memories to add")


store = InMemoryStore()
import numpy as np
import sounddevice as sd
import whisper
import os
from scipy.io.wavfile import write

whisper_model = whisper.load_model("base")

def speech_text(fs=16000, silence_limit=1.2, threshold=0.02):
    print(f"\n🎙️  Sun raha hoon... (Bolna shuru karein)")
    
    audio_frames = []
    chunk_size = 1024
    silent_chunks = 0
    
    
    with sd.InputStream(samplerate=fs, channels=1, dtype='float32') as stream:
        while True: 
            chunk, overflow = stream.read(chunk_size)
            audio_frames.append(chunk)
            
            
            energy = np.linalg.norm(chunk) / np.sqrt(len(chunk))
            
            
            if energy < threshold:
                silent_chunks += 1
            else:
                silent_chunks = 0
            
            if silent_chunks > int(silence_limit * fs / chunk_size) and len(audio_frames) > 30:
                print(" Silence detected. Processing...")
                break    

    recording = np.concatenate(audio_frames, axis=0)
    
  
    recording_int16 = (recording * 32767).astype(np.int16)
    
   
    temp_file = "temp_krrish.wav"
    write(temp_file, fs, recording_int16)
    
    
    print("Soch raha hoon")
    result = whisper_model.transcribe(temp_file, fp16=False, language="en")
    text = result["text"].strip()
    
    os.remove(temp_file)
    
    if text:
        print(f" Tune kaha: '{text}'")
    else:
        print(" Kuch sunayi nahi diya.")
        
    return text
def text_speech(text):
    engine = pyttsx3.init()
    voice=engine.getProperty('voices')
    engine.setProperty('voice', voice[1].id)
    engine.setProperty('rate', 225)
    engine.setProperty('volume', 1)
    engine.say(text)
    engine.runAndWait()


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.0,
    task="text-generation",
    streaming=True,         
)
llm1 = ChatHuggingFace(llm=llm)


llm_memory = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.0,
    task="text-generation",
    streaming=False,        
)
llm1_memory = ChatHuggingFace(llm=llm_memory)

# ============================================================
# OUTPUT PARSER
# ============================================================
parser = PydanticOutputParser(pydantic_object=MemoryDecision)

# ============================================================
# PROMPTS
# ============================================================
memory_prompt = """You are an Advanced Memory Management Assistant. Your sole purpose is to maintain a clean, non-redundant database of user facts.

### DATA:
1. Existing Memory: {existing_memory}
2. Last Message: {last_message}

### STRICT FILTERING RULES:
- **Deduplication:** Before extracting, compare the new information with "Existing Memory". If the fact is already present (even in different words), set is_add to false. 
- **Update Logic:** If the new message contradicts or updates an existing fact (e.g., "I moved from Delhi to Mumbai"), mark it as an update, not a new entry.
- **Utility Check:** Only extract high-value data (Identity, Preferences, Professional details, Goals). 
- **Strict Exclusion:** Ignore greetings, temporary feelings ("I'm tired"), fillers, or technical questions that don't reveal personal info.

### DECISION PROCESS:
1. Does this message contain a fact/preference? No -> is_add=false.
2. Is this fact already in 'Existing Memory'? Yes -> is_add=false.
3. Is it a new or updated fact? Yes -> is_add=true.

### OUTPUT FORMAT:
You MUST output ONLY a valid JSON object. No markdown, no explanation.

{format_instructions}"""

system_prompt = """You are Krrish, a sophisticated Personal AI Companion. Your unique strength is your ability to remember the user's life details and use them to provide a deeply personalized experience.

### YOUR INPUT DATA:
- Long-Term Memory: {long_term_memory}

### YOUR MISSION:
1. PERSONALIZED GREETING: Address the user by Name if found in memory.
2. MEMORY WEAVING: Weave in references to their past facts, goals, or preferences naturally.
3. THE 3-QUESTION RULE: End your response with exactly 3 personalized follow-up questions.

Be warm, helpful, and conversational. Keep response concise."""

# ============================================================
# NODES
# ============================================================
def remember_node(state: mystate, config: RunnableConfig, store: BaseStore):
    """Memory extract karo aur store karo — no streaming needed here."""
    user_id = config['configurable']['user_id']
    name_space = ("user", user_id, "details")

    stored_items = store.search(name_space)
    existing_memories = "\n".join([item.value.get('content', '') for item in stored_items])
    last_message = state['message'][-1].content

    formatted_prompt = memory_prompt.format(
        existing_memory=existing_memories if existing_memories else "No memory yet.",
        last_message=last_message,
        format_instructions=parser.get_format_instructions()
    )

    
    response = llm1_memory.invoke([SystemMessage(content=formatted_prompt)])

    try:
       
        raw = response.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        decision = parser.parse(raw)
        print(f"\n   [Memory Decision]: is_add={decision.is_add}, memories={[m.text for m in decision.memories]}")
        if decision.is_add:
            for mem in decision.memories:
                if mem.is_new_info and mem.text.strip():
                    store.put(name_space, str(uuid.uuid4()), {"content": mem.text})
                    print(f"\n   [Memory Saved]: {mem.text}")
    except Exception as e:
        print(f"\n   [Memory Note]: Could not parse — {e}")

    return {}


def chat_node(state: mystate, config: RunnableConfig, store: BaseStore):
    e
    user_id = config['configurable']['user_id']
    name_space = ("user", user_id, "details")

    stored_items = store.search(name_space)
    existing_memories = "\n".join([item.value.get('content', '') for item in stored_items])

    # FIX 3: sahi message list — objects, not strings
    messages_for_llm = [
        SystemMessage(content=system_prompt.format(long_term_memory=existing_memories))
    ] + [state['message'][-1].content]

    
    print("\n Krrish: ", end="", flush=True)
    full_response = ""
    for chunk in llm1.stream(messages_for_llm):
        
        if isinstance(chunk, AIMessageChunk) and chunk.content:
            
            print(chunk.content, end="", flush=True)
            full_response += chunk.content
    text_speech(full_response)
    print()  
    from langchain_core.messages import AIMessage
    return {"message": [AIMessage(content=full_response)]}



graph = StateGraph(mystate)
graph.add_node("remember", remember_node)
graph.add_node("chat", chat_node)

graph.add_edge(START, "remember")
graph.add_edge("remember", "chat")
graph.add_edge("chat", END)

graph = graph.compile(store=store)


print("\n" + "="*40)
print("   Krrish AI — Ready to Talk!")
print("   (type 'e' to exit)")
print("="*40)

config = {'configurable': {'user_id': 'user123'}}
count=0
while True:
    
    print("sy somthing")
    user_input=speech_text()
    print("processing...")
    
    if not user_input:
        count=count+1
        text_speech("you are not say anythng how can help you")
        if count>1:
            break
        
        continue
    state_input = {"message": [HumanMessage(content=user_input)]}

    
    graph.invoke(state_input, config=config)

    # Memory display
    memories = store.search(("user", "user123", "details"))
    if memories:
        print("\n   --- Memory DB ---")
        for item in memories:
            print(f"   * {item.value['content']}")
        print("   -----------------")