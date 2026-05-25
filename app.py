import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os

# ==============================
# Load Environment Variables
# ==============================
load_dotenv()

# ==============================
# Streamlit Page Config
# ==============================
st.set_page_config(
    page_title="AI Technical Analyzer",
    page_icon="🤖",
    layout="wide"
)

# ==============================
# Custom CSS Styling
# ==============================
st.markdown("""
<style>

.main {
    background: linear-gradient(to right, #141E30, #243B55);
    color: white;
}

.stTextInput > div > div > input {
    background-color: #1f2937;
    color: white;
    border-radius: 12px;
    padding: 12px;
    border: 2px solid #00E5FF;
}

.stButton>button {
    background: linear-gradient(90deg,#ff512f,#dd2476);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color: white;
}

.card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
}

h1 {
    text-align: center;
    color: #00E5FF;
}

h3 {
    color: #00E5FF;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# Sidebar
# ==============================
with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/4712/4712109.png",
        width=120
    )

    st.title("📚 Topics Covered")

    st.markdown("""
    - Python 🐍
    - SQL 🗄️
    - EDA 📊
    - Statistics 📈
    - Machine Learning 🤖
    - Neural Networks 🧠
    - LLMs 💬
    - Agentic AI 🚀
    """)

    st.info("Built using Streamlit + LangChain + Groq")

# ==============================
# Main Header
# ==============================
st.markdown("""
<h1>🤖 Gen AI Powered Technical Analyzer</h1>

<p style='text-align:center;
color:white;
font-size:18px;'>

Ask anything about Python, SQL, Machine Learning,
LLMs, Agentic AI and more 🚀

</p>
""", unsafe_allow_html=True)

# ==============================
# Prompt Template
# ==============================
template = """
You are a Data Science and AI trainer with 15+ years of experience.

Your task is to:

1. Classify the query into one of these modules:

- Python
- EDA
- Statistics
- PowerBI
- SQL
- Machine Learning
- Neural Networks
- Large Language Models
- Agentic AI

2. Extract 3-4 technical keywords from the query.

3. Explain the answer in 100-200 words.

4. Show a Python code snippet related to the query.

5. Provide some clickable YouTube or Blog links related to the query.

If the query is irrelevant to the above topics, display:
"Insufficient data"

Return the output in valid JSON format:

{{
    "topic":"Python",
    "keywords":["keyword1","keyword2","keyword3"],
    "answer":"Summary about the query",
    "python_code":"print('Hello World')",
    "links":["https://example.com","https://youtube.com"]
}}
"""

# ==============================
# User Input
# ==============================
user_input = st.text_input(
    "💬 Enter your technical question:"
)

# ==============================
# Prompt
# ==============================
prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "{query}")
])

# ==============================
# Groq Model
# ==============================
model = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# ==============================
# JSON Parser
# ==============================
parser = JsonOutputParser()

# ==============================
# Chain
# ==============================
chain = prompt | model | parser

# ==============================
# Submit Button
# ==============================
if st.button("🚀 Analyze Query"):

    if user_input:

        with st.spinner("Analyzing with AI... 🤖"):

            try:

                response = chain.invoke({
                    "query": user_input
                })

                # ==============================
                # Topic
                # ==============================
                st.markdown(f"""
                <div class="card">
                <h3>📌 Topic</h3>
                <p>{response['topic']}</p>
                </div>
                """, unsafe_allow_html=True)

                # ==============================
                # Keywords
                # ==============================
                st.markdown(f"""
                <div class="card">
                <h3>🔑 Keywords</h3>
                <p>{", ".join(response['keywords'])}</p>
                </div>
                """, unsafe_allow_html=True)

                # ==============================
                # Explanation
                # ==============================
                st.markdown(f"""
                <div class="card">
                <h3>🧠 Explanation</h3>
                <p>{response['answer']}</p>
                </div>
                """, unsafe_allow_html=True)

                # ==============================
                # Python Code
                # ==============================
                st.markdown("### 💻 Python Code")

                st.code(
                    response['python_code'],
                    language="python"
                )

                # ==============================
                # Links
                # ==============================
                st.markdown("### 🔗 Useful Resources")

                for link in response['links']:
                    st.markdown(f"- [{link}]({link})")

            except Exception as e:

                st.error(f"Error: {e}")

    else:

        st.warning("⚠️ Please enter a question.")