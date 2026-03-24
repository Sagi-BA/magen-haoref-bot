"""
מגן על הזכויות - צ'אטבוט זכויות חיילי מילואים
Streamlit + LangChain RAG app powered by OpenAI gpt-4o-mini
"""

import os
import streamlit as st
from pathlib import Path

# --- LangChain imports ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ============================================================
# Page config - must be first Streamlit call
# ============================================================
st.set_page_config(
    page_title="מגן על הזכויות - מילואים",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# RTL CSS injection
# ============================================================
RTL_CSS = """
<style>
    /* Global RTL — avoid .stApp to not break Streamlit layout */
    .stMainBlockContainer, .stChatMessage, .stMarkdown, .stTextInput > div > div > input {
        direction: rtl;
        text-align: right;
    }
    /* Chat input */
    .stChatInputContainer textarea,
    [data-testid="stChatInput"] textarea,
    .stChatInput textarea {
        direction: rtl;
        text-align: right;
    }
    /* Chat messages */
    [data-testid="stChatMessageContent"] {
        direction: rtl;
        text-align: right;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        direction: rtl;
        text-align: right;
    }
    [data-testid="stSidebar"] > div {
        overflow-x: hidden;
        padding-right: 1rem;
        padding-left: 1rem;
    }
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        direction: rtl;
        text-align: right;
    }
    /* Links stay LTR inside RTL */
    a[href] {
        direction: ltr;
        unicode-bidi: embed;
    }
    /* Sticky header */
    .sticky-header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #1a1a2e;
        color: #e0e0e0;
        text-align: center;
        padding: 8px 0;
        font-size: 0.85em;
        z-index: 9999;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        direction: rtl;
    }
    /* Push page content below the sticky header */
    .stApp {
        margin-top: 40px;
    }
    /* Emergency banner */
    .emergency-banner {
        background-color: #FFEBEE;
        border-right: 4px solid #C62828;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 12px;
        direction: rtl;
    }
    /* Topic buttons inside expander */
    div[data-testid="stExpander"] .stButton > button {
        border-radius: 20px;
        padding: 4px 16px;
        font-size: 0.85em;
        margin: 2px;
    }
    /* Hide Streamlit footer badge and branding */
    footer {visibility: hidden;}
    [data-testid="manage-app-button"],
    .viewerBadge_container__r5tak,
    .styles_viewerBadge__CvC9N,
    ._profileContainer_gzau3_53,
    [data-testid="stStatusWidget"],
    #MainMenu,
    header[data-testid="stHeader"] .stActionButton,
    iframe[title="Streamlit Cloud"] {
        display: none !important;
    }
    /* Disclaimer bar */
    .disclaimer {
        background-color: #FFF3E0;
        border-right: 4px solid #E65100;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.85em;
        direction: rtl;
        margin-bottom: 16px;
    }
</style>
"""
st.markdown(RTL_CSS, unsafe_allow_html=True)
st.markdown('<div class="sticky-header">נוצר ע"י <a href="http://sagiai.co.il/" target="_blank" style="color: #82b1ff; text-decoration: none;">שגיא בר און</a></div>', unsafe_allow_html=True)

# ============================================================
# Constants
# ============================================================
BASE_DIR = Path(__file__).parent
BENEFITS_PATH = BASE_DIR / "benefits.md"
SYSTEM_PROMPT_PATH = BASE_DIR / "system_prompt.md"
FAISS_INDEX_PATH = BASE_DIR / "faiss_index"

# Headers to split on (Markdown-based chunking)
HEADERS_TO_SPLIT = [
    ("#", "נושא_ראשי"),
    ("##", "נושא_משני"),
    ("###", "תת_נושא"),
]

# Top-k retrieval
RETRIEVAL_K = 5

# Available topics for the user to browse
TOPICS = [
    "אובדן או נזק לציוד אישי",
    "תיקונים בבית ומעבר דירה",
    "ביטול חופשות/טיסות",
    "משרת מילואים שאינו עובד",
    "כביש 6 - החזרים",
    "חיילים בודדים",
    "בעלי חיים",
    "סיוע כלכלי - חריגים",
    "עצמאיים ובעלי חברות",
    "קייטנות 2026",
    "בייביסיטר",
    "הארכת חל\"ד / אי חזרה מחל\"ד",
    "בן/בת זוג שאינם עובדים",
    "סיוע לגרושים/גרושות",
    "אובדן הכנסה בן/בת זוג - שכיר/ה",
    "אובדן הכנסה בן/בת זוג - עצמאי/ת",
    "עצמאית - כל מה שאת צריכה לדעת",
    "סטודנטים",
    "טיפולים זוגיים + נפשיים",
    "זכויות במקום העבודה",
    "מענקי פיקוד",
    "נקודות זיכוי 2026",
    "שוברי חופשה",
    "כרטיס fighter",
]

# ============================================================
# Load secrets / API key
# ============================================================
def get_api_key() -> str:
    """Get OpenAI API key from Streamlit secrets or env."""
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        st.error("⚠️ לא הוגדר מפתח OpenAI. הגדר OPENAI_API_KEY ב-secrets או כמשתנה סביבה.")
        st.stop()
    return key


# ============================================================
# Build / load vector store (cached)
# ============================================================
@st.cache_resource(show_spinner="טוען את מאגר הזכויות...")
def build_vector_store(api_key: str):
    """Load benefits.md, split by headers, embed with OpenAI, store in FAISS."""

    # If pre-built index exists on disk, load it
    if FAISS_INDEX_PATH.exists():
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key,
        )
        return FAISS.load_local(
            str(FAISS_INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    # Otherwise build from scratch
    raw_text = BENEFITS_PATH.read_text(encoding="utf-8")

    # Split by Markdown headers
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,
    )
    header_docs = splitter.split_text(raw_text)

    # Convert to LangChain Documents with enriched metadata
    documents = []
    for doc in header_docs:
        meta = doc.metadata.copy()
        # Build a context prefix from header hierarchy
        context_parts = []
        for key in ["נושא_ראשי", "נושא_משני", "תת_נושא"]:
            if key in meta:
                context_parts.append(meta[key])
        context_prefix = " > ".join(context_parts)

        # Prepend context to content for better retrieval
        enriched_content = f"[הקשר: {context_prefix}]\n\n{doc.page_content}" if context_prefix else doc.page_content

        documents.append(
            Document(
                page_content=enriched_content,
                metadata=meta,
            )
        )

    # Embed and store
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
    )
    vector_store = FAISS.from_documents(documents, embeddings)

    # Save to disk for faster reload
    vector_store.save_local(str(FAISS_INDEX_PATH))

    return vector_store


# ============================================================
# Load system prompt (cached)
# ============================================================
@st.cache_data
def load_system_prompt() -> str:
    """Read system_prompt.md."""
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


# ============================================================
# RAG chain - retrieve + generate
# ============================================================
def retrieve_context(vector_store, query: str) -> str:
    """Retrieve top-k relevant chunks from the vector store."""
    results = vector_store.similarity_search(query, k=RETRIEVAL_K)
    if not results:
        return "לא נמצא מידע רלוונטי במסמך."

    context_parts = []
    for i, doc in enumerate(results, 1):
        context_parts.append(f"--- קטע {i} ---\n{doc.page_content}")
    return "\n\n".join(context_parts)


def generate_response(
    llm: ChatOpenAI,
    system_prompt: str,
    retrieved_context: str,
    chat_history: list,
    user_question: str,
) -> str:
    """Build the full prompt and generate a response."""

    # System message with RAG context injected
    full_system = (
        f"{system_prompt}\n\n"
        f"---\n\n"
        f"## מידע שנשלף ממסמך הזכויות (RAG)\n\n"
        f"להלן קטעי מידע רלוונטיים שנשלפו ממסמך הזכויות. "
        f"ענה אך ורק על בסיס מידע זה. "
        f"אם התשובה לא נמצאת בקטעים - אמור זאת בכנות.\n\n"
        f"{retrieved_context}"
    )

    messages = [SystemMessage(content=full_system)]

    # Add chat history (keep last 10 turns for context window management)
    for msg in chat_history[-20:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # Current question
    messages.append(HumanMessage(content=user_question))

    response = llm.invoke(messages)
    return response.content


# ============================================================
# Sidebar
# ============================================================
def render_sidebar():
    """Render the sidebar with contacts and disclaimer."""
    with st.sidebar:
        st.image(
            "logo.png",
            width=80,
        )
        st.markdown("## 🛡️ מגן על הזכויות")
        st.markdown("**צ'אטבוט מידע לחיילי מילואים**")
        st.markdown("מרכז חוסן קהילתי • מחוז חיפה")
        st.divider()

        # Emergency contacts - always visible
        st.markdown("### 🚨 קווי סיוע נפשי")
        st.markdown(
            """
- **ער"ן:** *2201 / *3201
- **נפש אחת:** *8944
- **סה"ר:** 055-957-1399 | sahar.org.il
"""
        )
        st.divider()

        # General contacts
        st.markdown("### 📞 אנשי קשר")
        st.markdown(
            """
- **מוקד מילואים:** 1111 שלוחה 4
- **נטאלי סילבר:** 055-941-0030
- **שגיא בר און:** 054-999-5050
- **שלומי אזולאי:** 052-395-4499
- **אורנה אביקזר:** 052-555-5658
- **נמרוד אסא:** 052-642-6787
- **לילך אביסרור:** 054-774-3672
"""
        )
        st.divider()

        # Professional contacts
        st.markdown("### 👨‍⚖️ אנשי מקצוע")
        st.markdown(
            """
- **עו"ד מוריאל קוט:** 053-354-5552
- **רו"ח יקי קינן:** 054-779-6571
- **רו"ח כוכב אבשלום:** 053-966-6740
"""
        )
        st.divider()

        # Disclaimer
        st.markdown(
            '<div class="disclaimer">'
            "⚠️ <strong>הבהרה:</strong> בוט זה מספק מידע כללי בלבד ואינו מהווה ייעוץ משפטי או פיננסי. "
            "לאימות זכאות ספציפית, פנה למוקד 1111 שלוחה 4."
            "</div>",
            unsafe_allow_html=True,
        )

        # Clear chat button
        if st.button("🗑️ נקה היסטוריית שיחה", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ============================================================
# Main app
# ============================================================
def main():
    api_key = get_api_key()

    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        openai_api_key=api_key,
        max_tokens=2000,
    )

    # Build vector store
    vector_store = build_vector_store(api_key)

    # Load system prompt
    system_prompt = load_system_prompt()

    # Render sidebar
    render_sidebar()

    # Main chat area
    st.markdown("# 🛡️ מגן על הזכויות")
    st.markdown("**שאל אותי כל שאלה על זכויות חיילי מילואים - מענקים, פיצויים, הטבות ועוד.**")
    st.markdown(
        '<div class="disclaimer">'
        "💡 <strong>טיפ:</strong> ציין את שנת השירות (2023/2024/2025/2026) ואת סוג היחידה לתשובה מדויקת יותר."
        "</div>",
        unsafe_allow_html=True,
    )

    # Topics expander with clickable buttons
    with st.expander("📋 נושאים שאני יכול לעזור בהם"):
        cols = st.columns(3)
        for i, topic in enumerate(TOPICS):
            with cols[i % 3]:
                if st.button(topic, key=f"topic_{i}", use_container_width=True):
                    st.session_state.topic_click = topic
                    st.rerun()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Handle topic button click
    if "topic_click" in st.session_state:
        user_input = f"ספר לי על {st.session_state.topic_click}"
        del st.session_state.topic_click
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant", avatar="🛡️"):
            with st.spinner("מחפש במאגר הזכויות..."):
                context = retrieve_context(vector_store, user_input)
                response = generate_response(
                    llm=llm,
                    system_prompt=system_prompt,
                    retrieved_context=context,
                    chat_history=st.session_state.messages[:-1],
                    user_question=user_input,
                )
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🛡️" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("מה תרצה לדעת על הזכויות שלך?"):
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate response
        with st.chat_message("assistant", avatar="🛡️"):
            with st.spinner("מחפש במאגר הזכויות..."):
                # Retrieve relevant context
                context = retrieve_context(vector_store, user_input)

                # Generate LLM response
                response = generate_response(
                    llm=llm,
                    system_prompt=system_prompt,
                    retrieved_context=context,
                    chat_history=st.session_state.messages[:-1],  # exclude current
                    user_question=user_input,
                )

            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    main()
