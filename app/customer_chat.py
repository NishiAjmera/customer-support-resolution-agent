"""Customer-facing fictional support chat; operator artifacts live elsewhere."""
import uuid

import streamlit as st
from dotenv import load_dotenv

from agent.case_memory import CaseMemory
from agent.conversation import ConversationService
from agent.graph import GeminiModel, SupportAgent
from learning.lesson_store import LessonStore

load_dotenv()
st.set_page_config(page_title="Fictional shop support")
st.title("Fictional shop support")
st.caption("Recommendations for demo orders only. No purchases or refunds are processed here.")

if "conversation_service" not in st.session_state:
    store = LessonStore("runs/chat/lessons.sqlite")
    agent = SupportAgent(GeminiModel(), case_memory=CaseMemory("runs/chat/case-memory.sqlite"), skill_store=store)
    st.session_state.conversation_service = ConversationService(agent)
    st.session_state.chat_session_id = str(uuid.uuid4())
    st.session_state.chat_messages = []

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("How can we help with your demo order?"):
    turn_id = str(uuid.uuid4())
    result = st.session_state.conversation_service.handle_message(
        st.session_state.chat_session_id, "demo-customer", prompt, turn_id=turn_id)
    st.session_state.chat_messages.extend([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": result.assistant_message},
    ])
    st.rerun()
