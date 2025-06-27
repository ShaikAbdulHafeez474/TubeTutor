

import os
import re
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_together import ChatTogether, TogetherEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import initialize_agent, Tool
from langchain_community.tools.tavily_search import TavilySearchResults

# Set API Keys
os.environ["TOGETHER_API_KEY"] = ""

# LLM and Embeddings
llm = ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", temperature=0.2)
embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-32k-retrieval")

# Prompts
note_prompt = PromptTemplate(
    template="""
    You're a note-taking assistant. Convert the following transcript into clear, concise lecture notes:
    - Headings
    - Bullet points
    - Definitions
    - Examples

    Transcript:
    {chunk}
    """,
    input_variables=["chunk"]
)

quiz_prompt = PromptTemplate(
    template="""
    Generate 3 multiple-choice questions from the following transcript. Include correct answers.

    Transcript:
    {chunk}
    """,
    input_variables=["chunk"]
)

assignment_prompt = PromptTemplate(
    template="""
    Based on the transcript below, generate 2 beginner-level coding exercises and short answers.

    Transcript:
    {chunk}
    """,
    input_variables=["chunk"]
)

compare_prompt = PromptTemplate(
    template="""
    Compare the following two transcripts. Highlight:
    - Similarities
    - Differences
    - Unique insights

    Transcript 1:
    {transcript1}

    Transcript 2:
    {transcript2}
    """,
    input_variables=["transcript1", "transcript2"]
)

# Helper Functions
def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\\.be/)([^&?]+)", url)
    return match.group(1) if match else None

def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        return " ".join(chunk["text"] for chunk in transcript_list)
    except TranscriptsDisabled:
        return None

def split_transcript(transcript):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents([transcript])

def create_vector_store(docs):
    return FAISS.from_documents(docs, embeddings)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_notes(chunks):
    return [llm.invoke(note_prompt.invoke({"chunk": chunk.page_content})).content for chunk in chunks]

def generate_quiz(chunks):
    return [llm.invoke(quiz_prompt.invoke({"chunk": chunk.page_content})).content for chunk in chunks]

def generate_assignments(chunks):
    return [llm.invoke(assignment_prompt.invoke({"chunk": chunk.page_content})).content for chunk in chunks]

def find_resources(query):
    tavily = TavilySearchResults()
    tools = [Tool.from_function(name="search", func=tavily.run, description="Web search")]
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=False)
    return agent.run(query)

def compare_videos(t1, t2):
    return llm.invoke(compare_prompt.invoke({"transcript1": t1, "transcript2": t2})).content

# Streamlit UI
st.set_page_config(page_title="🎨 AI Teaching Assistant", layout="centered")
st.markdown("""
    <style>
    .main {
        background: linear-gradient(145deg, #f0f4f8, #c3e0e5);
        padding: 2rem;
        border-radius: 12px;
    }
    .stTextInput>div>div>input {
        border-radius: 0.75rem;
        border: 2px solid #5dade2;
        background-color: #fefefe;
    }
    .stSelectbox>div>div>div {
        border-radius: 0.75rem;
        background-color: #ebf5fb;
        font-weight: bold;
        color: #2e4053;
    }
    .stButton>button {
        border-radius: 0.5rem;
        background: linear-gradient(to right, #00c6ff, #0072ff);
        color: white;
        font-weight: bold;
        padding: 0.6rem 1.2rem;
    }
    .block-container {
        padding: 2rem 3rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style='text-align: center; color: #154360;'>🎓 AI Teaching Assistant</h1>
    <p style='text-align: center; font-size: 18px; color: #1b2631;'>Transform YouTube videos into interactive, intelligent content effortlessly!</p>
    <hr style='border-top: 1px solid #aed6f1;'>
""", unsafe_allow_html=True)

option = st.selectbox("🎯 What do you want to do?", [
    "Summarize",
    "Ask a custom question",
    "Compare with another video",
    "Lecture Notes Generator",
    "Quiz Generator",
    "Assignment / Coding Problems Generator",
    "Follow-up Resource Finder"
])

video_url = st.text_input("🔗 Enter YouTube video URL")
video_id = extract_video_id(video_url)
transcript = get_transcript(video_id) if video_id else None

if option == "Compare with another video":
    second_url = st.text_input("🔁 Enter second video URL to compare")
    if st.button("🧠 Compare Videos"):
        t1, t2 = get_transcript(extract_video_id(video_url)), get_transcript(extract_video_id(second_url))
        if t1 and t2:
            result = compare_videos(t1[:4000], t2[:4000])
            st.markdown(result)
        else:
            st.error("One or both transcripts unavailable.")

elif transcript:
    chunks = split_transcript(transcript)
    if option == "Summarize":
        retriever = create_vector_store(chunks).as_retriever()
        question = "Summarize this video"
        chain = RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }) | PromptTemplate(
            template="""
            You are a helpful assistant. Use only the provided context to answer.

            {context}
            Question: {question}
            """,
            input_variables=["context", "question"]
        ) | llm | StrOutputParser()
        summary = chain.invoke(question)
        st.text_area("📄 Summary", summary, height=300)

    elif option == "Ask a custom question":
        custom_q = st.text_input("💬 Your question about the video")
        if st.button("🧠 Ask"):
            retriever = create_vector_store(chunks).as_retriever()
            chain = RunnableParallel({
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }) | PromptTemplate(
                template="""
                You are a helpful assistant. Use only the provided context to answer.

                {context}
                Question: {question}
                """,
                input_variables=["context", "question"]
            ) | llm | StrOutputParser()
            answer = chain.invoke(custom_q)
            st.text_area("💡 AI Answer", answer, height=300)

    elif option == "Lecture Notes Generator":
        if st.button("📝 Generate Notes"):
            notes = generate_notes(chunks)
            for i, n in enumerate(notes):
                st.markdown(f"### 📘 Section {i+1}")
                st.markdown(n)

    elif option == "Quiz Generator":
        if st.button("🧪 Generate Quiz"):
            quiz = generate_quiz(chunks)
            for i, q in enumerate(quiz):
                st.markdown(f"### ❓ Quiz {i+1}")
                st.markdown(q)
            st.success("✔️ Quiz Generated. (Manual review for answers)")

    elif option == "Assignment / Coding Problems Generator":
        if st.button("👨‍💻 Generate Assignments"):
            tasks = generate_assignments(chunks)
            for i, t in enumerate(tasks):
                st.markdown(f"### ⚙️ Task {i+1}")
                st.markdown(t)

    elif option == "Follow-up Resource Finder":
        if st.button("🌐 Find More Resources"):
            followup = find_resources(f"learning resources about: {transcript[:300]}")
            st.markdown(followup)

else:
    st.warning("⚠️ Please enter a valid YouTube URL with available transcript.")
