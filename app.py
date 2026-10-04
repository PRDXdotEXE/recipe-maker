import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st
import whisper
from dotenv import load_dotenv
from ollama import Client as OllamaClient

# Load environment variables
load_dotenv()

# Constants
ASSISTANT_CONFIG_FILE = ".backboard_assistant_id"
DEFAULT_OLLAMA_MODELS = ["llama3.2", "gemma2", "qwen2.5"]

# Streamlit Page Setup
st.set_page_config(
    page_title="Family Recipe Memory",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Backboard Integration Setup
BACKBOARD_ASSISTANT_NAME = "Family Recipe Memory Assistant"
BACKBOARD_SYSTEM_PROMPT = (
    "You are the Family Recipe Memory Assistant. Your job is to remember, retrieve, "
    "and answer questions about saved family recipes and culinary traditions. "
    "Preserve names, dishes, ingredients, instructions, tips, and personal stories "
    "accurately. Never invent recipe details that are not present in memory. "
    "Be warm, helpful, and concise."
)


def get_backboard_client():
    """Create a Backboard client using the API key from .env."""
    api_key = os.getenv("BACKBOARD_API_KEY")
    if not api_key or api_key == "YOUR_BACKBOARD_API_KEY":
        return None

    try:
        from backboard import BackboardClient

        return BackboardClient(api_key=api_key)
    except Exception as e:
        st.warning(f"Failed to initialize Backboard client: {e}")
        return None


def _run_async(coro):
    """Run one Backboard async SDK operation from Streamlit's synchronous code."""
    return asyncio.run(coro)


async def _resolve_assistant_id(client) -> str:
    """Validate the saved assistant ID, then fall back to exact-name lookup or creation."""
    config_path = Path(ASSISTANT_CONFIG_FILE)

    # First, try the locally stored assistant ID.
    if config_path.exists():
        try:
            saved_id = config_path.read_text().strip()
            if saved_id:
                try:
                    assistant = await client.get_assistant(saved_id)
                    assistant_id = getattr(assistant, "assistant_id", saved_id)
                    return assistant_id
                except Exception:
                    # The saved ID may belong to an old/deleted assistant.
                    pass
        except Exception:
            pass

    # Reuse an existing assistant with the exact project name if possible.
    assistants = await client.list_assistants(
        name=BACKBOARD_ASSISTANT_NAME,
        limit=1,
    )
    if assistants:
        assistant = assistants[0]
        assistant_id = getattr(assistant, "assistant_id", None) or getattr(
            assistant, "id", None
        )
        if assistant_id:
            config_path.write_text(assistant_id)
            return assistant_id

    # Otherwise create the assistant using the current SDK.
    assistant = await client.create_assistant(
        name=BACKBOARD_ASSISTANT_NAME,
        system_prompt=BACKBOARD_SYSTEM_PROMPT,
    )
    assistant_id = getattr(assistant, "assistant_id", None) or getattr(
        assistant, "id", None
    )
    if not assistant_id:
        raise RuntimeError("Backboard created an assistant but returned no assistant ID.")

    config_path.write_text(assistant_id)
    return assistant_id


def get_or_create_assistant_id(client) -> Optional[str]:
    """Get or create the shared Backboard assistant used for family memory."""
    if not client:
        return None

    try:
        return _run_async(_resolve_assistant_id(client))
    except Exception as e:
        st.error(f"Failed to create or load Backboard Assistant: {e}")
        return None


@st.cache_resource
def load_whisper_model(model_size: str = "base"):
    """Cache local Whisper model in memory."""
    return whisper.load_model(model_size)


def transcribe_audio_locally(audio_file) -> Optional[str]:
    """Transcribes uploaded audio locally using OpenAI Whisper."""
    model = load_whisper_model("base")
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(audio_file.name).suffix
    ) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path)
        return result.get("text", "").strip()
    except Exception as e:
        st.error(f"Local Whisper transcription failed: {e}")
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def extract_recipe_with_ollama(
    transcript: str, person: str, model_name: str
) -> Optional[Dict[str, Any]]:
    """Structures raw audio transcript into JSON schema using local Ollama."""
    client = OllamaClient()

    prompt = f"""
You are a meticulous culinary assistant. Extract structured recipe details from the following transcript provided by '{person}'.

STRICT RULES:
1. Do NOT invent or hallucinate ingredients, measurements, or cooking times.
2. If any field is missing or not mentioned in the text, output "Not specified".
3. Only extract personal context or story elements actually present in the text.

Transcript:
"{transcript}"

Respond strictly with valid JSON with the following structure:
{{
  "recipe_name": "Name of the dish or 'Not specified'",
  "person": "{person}",
  "ingredients": ["item 1", "item 2"],
  "instructions": ["step 1", "step 2"],
  "cooking_tips": ["tip 1"] or ["Not specified"],
  "personal_story": "Personal anecdote/context or 'Not specified'"
}}
"""
    try:
        response = client.generate(model=model_name, prompt=prompt, format="json")
        return json.loads(response["response"])
    except Exception as e:
        st.error(f"Ollama local extraction failed: {e}")
        return None


def save_recipe_to_backboard(
    client, assistant_id: str, recipe_data: Dict[str, Any], raw_transcript: str
) -> bool:
    """Save a recipe as persistent Backboard memory using the current SDK."""
    if not client or not assistant_id:
        return False

    memory_payload = (
        f"RECIPE RECORD:\n"
        f"Person: {recipe_data.get('person', 'Unknown')}\n"
        f"Dish: {recipe_data.get('recipe_name', 'Untitled')}\n"
        f"Ingredients: {', '.join(recipe_data.get('ingredients', []))}\n"
        f"Instructions: {' '.join(recipe_data.get('instructions', []))}\n"
        f"Cooking Tips: {' '.join(recipe_data.get('cooking_tips', []))}\n"
        f"Personal Context: {recipe_data.get('personal_story', 'None')}\n"
        f"Raw Audio Transcript: {raw_transcript}\n"
    )

    prompt = (
        "Save the following family recipe to persistent memory. "
        "Preserve the factual details exactly as provided. Do not invent or "
        "modify ingredients, quantities, instructions, tips, names, or personal context.\n\n"
        f"{memory_payload}"
    )

    async def _save():
        return await client.send_message(
            prompt,
            assistant_id=assistant_id,
            memory="Auto",
            stream=False,
        )

    try:
        _run_async(_save())
        return True
    except Exception as e:
        st.warning(f"Could not save recipe to Backboard persistent memory: {e}")
        return False


def query_backboard_memory(
    client, assistant_id: str, query: str
) -> Optional[str]:
    """Query persistent Backboard memory without saving the question as a memory."""
    if not client or not assistant_id:
        return None

    async def _query():
        return await client.send_message(
            query,
            assistant_id=assistant_id,
            memory="Readonly",
            stream=False,
        )

    try:
        response = _run_async(_query())
        return getattr(response, "content", None) or "No answer received."
    except Exception as e:
        st.error(f"Failed to query Backboard memory: {e}")
        return None


# --- UI LAYOUT ---

st.title("🍲 Family Recipe Memory")
st.caption("Preserve voice recipes from loved ones using Local AI + Persistent Memory.")

st.info(
    "🔒 **Local-first AI & Privacy Guarantee:**\n"
    "- Voice transcription runs locally via OpenAI Whisper.\n"
    "- Recipe extraction runs locally via Ollama.\n"
    "- Only structured text/transcripts are saved to persistent memory. Audio files are **never** uploaded."
)

# Sidebar Config
with st.sidebar:
    st.header("⚙️ Local AI Settings")
    selected_model = st.selectbox("Ollama Model", DEFAULT_OLLAMA_MODELS, index=0)

    backboard_key = os.getenv("BACKBOARD_API_KEY")
    if backboard_key and backboard_key != "YOUR_BACKBOARD_API_KEY":
        st.success("✅ Backboard API Key detected")
    else:
        st.warning(
            "⚠️ No Backboard API Key found in `.env`. Persistent search disabled."
        )

# Main Application Tabs
tab_extract, tab_search = st.tabs(["🎙️ Extract Voice Recipe", "🧠 Ask Family Memory"])

with tab_extract:
    st.subheader("Extract & Preserve a Recipe")

    col_input1, col_input2 = st.columns([1, 2])
    with col_input1:
        person_name = st.text_input(
            "👤 Who is this recipe from?", placeholder="e.g., Grandma, Dad, Aunt Mary"
        )
    with col_input2:
        audio_file = st.file_uploader(
            "🎙️ Upload Voice Note", type=["mp3", "wav", "m4a", "ogg"]
        )

    if st.button("🚀 Extract Recipe", type="primary", use_container_width=True):
        if not person_name:
            st.error("Please enter who this recipe is from.")
        elif not audio_file:
            st.error("Please upload an audio file.")
        else:
            # Step 1: Transcribe
            with st.status("Processing Voice Recipe...", expanded=True) as status:
                st.write("🎙️ Transcribing locally with Whisper...")
                raw_transcript = transcribe_audio_locally(audio_file)

                if not raw_transcript:
                    status.update(label="Transcription Failed", state="error")
                    st.stop()

                # Step 2: Extract
                st.write("🧠 Structuring recipe with local Ollama...")
                structured_data = extract_recipe_with_ollama(
                    raw_transcript, person_name, selected_model
                )

                if not structured_data:
                    status.update(label="Ollama Extraction Failed", state="error")
                    st.stop()

                # Step 3: Backboard
                st.write("💾 Saving recipe to family memory...")
                bb_client = get_backboard_client()
                assistant_id = get_or_create_assistant_id(bb_client)

                if bb_client and assistant_id:
                    saved = save_recipe_to_backboard(
                        bb_client, assistant_id, structured_data, raw_transcript
                    )
                    if saved:
                        st.write("✅ Saved to Backboard persistent memory!")
                    else:
                        st.write(
                            "⚠️ Memory save skipped (API error), showing local results only."
                        )
                else:
                    st.write("⚠️ Backboard key missing. Showing local results only.")

                status.update(label="✅ Complete!", state="complete")

            # Display Results
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("### 📝 Original Transcript")
                st.info(raw_transcript)

            with col_right:
                st.markdown("### ✨ Structured Recipe")
                st.json(structured_data)

            # Markdown Export
            md_content = f"# {structured_data.get('recipe_name', 'Recipe')} (by {person_name})\n\n"
            md_content += (
                f"**Story:** {structured_data.get('personal_story', 'N/A')}\n\n"
            )
            md_content += (
                "### Ingredients\n"
                + "\n".join([f"- {i}" for i in structured_data.get("ingredients", [])])
                + "\n\n"
            )
            md_content += (
                "### Instructions\n"
                + "\n".join(
                    [f"1. {i}" for i in structured_data.get("instructions", [])]
                )
                + "\n\n"
            )
            md_content += "### Tips\n" + "\n".join(
                [f"- {i}" for i in structured_data.get("cooking_tips", [])]
            )

            st.download_button(
                label="📥 Download Recipe (.md)",
                data=md_content,
                file_name=f"{person_name}_{structured_data.get('recipe_name', 'recipe')}.md",
                mime="text/markdown",
            )

with tab_search:
    st.subheader("🧠 Ask Your Family Recipe Memory")
    query_input = st.text_input(
        "What would you like to remember?",
        placeholder="e.g., What did Grandma say about making momo?",
    )

    if st.button("🔎 Ask Family Memory", type="primary"):
        if not query_input:
            st.error("Please enter a question.")
        else:
            bb_client = get_backboard_client()
            assistant_id = get_or_create_assistant_id(bb_client)

            if not bb_client or not assistant_id:
                st.error(
                    "Backboard is not configured. Please add your `BACKBOARD_API_KEY` to `.env` to enable persistent memory search."
                )
            else:
                with st.spinner("Searching family recipe memory..."):
                    answer = query_backboard_memory(
                        bb_client, assistant_id, query_input
                    )
                    if answer:
                        st.markdown("### Answer")
                        st.success(answer)
