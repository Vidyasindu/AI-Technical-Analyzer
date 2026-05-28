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
# JSON Parser
# ==============================
parser = JsonOutputParser()



# ==============================
# Prompt Template
# ==============================

template = """
You are an expert AI tutor and technical assistant.

Your task is to:

1. Identify the topic of the user query.

2. Extract 3-4 important keywords from the query.

3. Explain the answer in simple and clear language in 100-200 words.

4. Provide a Python code example if applicable.

5. Provide useful YouTube or blog links related to the topic.

IMPORTANT:
- Return ONLY valid JSON.
- Do not return markdown.
- Do not use triple backticks.
- Do not add explanations outside JSON.
- "keywords" must always be a JSON array.
- "links" must always be a JSON array.

Return JSON in this EXACT structure:

{{ 
    "topic": "Topic Name", 
    "keywords": [ "keyword1", "keyword2" ], 
    "answer": "Detailed explanation here", 
    "python_code": "print('Hello World')", 
    "links": [ "https://example.com" ] 
}}

If Python code is not applicable, return:
"python_code": "Not applicable"

All fields are mandatory.
Never skip any field.

{format_instructions}
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
]).partial( 
    format_instructions=parser.get_format_instructions() 
)


# ==============================
# Groq Model
# ==============================
model = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


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
                <p>{response.get('topic') or response.get('Topic', 'No topic generated')}</p>
                </div>
                """, unsafe_allow_html=True)

                # ==============================
                # Keywords
                # ==============================
                keywords = response.get('keywords') or response.get('Keywords', [])

                st.markdown(f"""
                <div class="card">
                <h3>🔑 Keywords</h3>
                <p>{", ".join(keywords)}</p>
                </div>
                """, unsafe_allow_html=True)

                # ==============================
                # Explanation
                # ==============================
                st.markdown(f"""
                <div class="card">
                <h3>🧠 Explanation</h3>
                <p>{response.get('answer') or response.get('Answer', 'No answer generated')}</p>
                </div>
                """, unsafe_allow_html=True)

                # ==============================
                # Python Code
                # ==============================
                st.markdown("### 💻 Python Code")

                st.code(
                    response.get('python_code') or response.get('Python Code', 'No code generated'),
                    language="python"
                )

                # ==============================
                # Links
                # ==============================
                st.markdown("### 🔗 Useful Resources")

                links = response.get('links') or response.get('Links', [])

                for link in links:
                    st.markdown(f"- [{link}]({link})")

            except Exception as e:

                st.error(f"Error: {e}")

    else:

        st.warning("⚠️ Please enter a question.")

