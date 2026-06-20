# IA-Front
import streamlit as st
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.db.sqlite import SqliteDb

st.set_page_config(
    page_title="Agente de IA com Memória",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agente de Ia com Memória")
st.markdown("Conversa com um agente inteligente que mantem histórico de iteração")

API_KEY = ""

if "agente" not in st.session_state:
    st.session_state.agente = Agent(
        model=OpenAIChat(
            id="meta/llama-3.1-8b-instruct",
            api_key=API_KEY,
            base_url="https://integrate.api.nvidia.com/v1"
        ),
        instructions="""
        Voce é um assitente útil e profissional.
        responda de forma clara, pedagogica e resumida.
        """,
        db=SqliteDb(db_file="agente.db"),
        add_history_to_context=True,
        num_history_runs=3,
        markdown=True
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

st.subheader("Historico da Conversa")

chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

user_input = st.chat_input("Digite sua mensagem ...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)

    with st.spinner("Agente pensando ..."):
        try:
            response = st.session_state.agente.run(
                user_input,
                session_id="1"
            )
            response_text = (
                response.content
                if hasattr(response, 'content')
                else str(response)
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text
            })
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(response_text)
        except Exception as e:
            st.error(f"Erro ao processar a solicitação")

with st.sidebar:
    st.title("Configurações")
    if st.button("🧹Limpar Conversa"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.subheader("Informações")
    st.info("""
        **MODELO:**llama-3.1-8b-instruct (nvidia)
        **BANCO:** SQLite
        **MEMORIA**Ultimas 3 Interações
        **AUTOR:** Kaiky yoshio
    """)
    st.divider()
    st.caption(f"total de mensagens:{len(st.session_state.messages)}")
