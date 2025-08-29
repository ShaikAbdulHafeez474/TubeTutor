# 🧑‍🏫 AI-Powered YouTube Teaching Assistant — Enhanced Colorful UI

import os
import re
import json
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_together import ChatTogether, TogetherEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import initialize_agent, Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow

# -------------------------
# Streamlit Secrets Setup
# -------------------------
# In your secrets.toml, add:

# [youtube_credentials.web]
# client_id = "..."
# project_id = "..."
# auth_uri = "https://accounts.google.com/o/oauth2/auth"
# token_uri = "https://oauth2.googleapis.com/token"
# auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
# client_secret = "..."
# redirect_uris = ["https://TubeTutor-App.streamlit.app"]
# javascript_origins = ["https://TubeTutor-App.streamlit.app"]

# -------------------------
# API Keys
# -------------------------
os.environ["TOGETHER_API_KEY"] = "5c22e5f0d9af71d1cd7dfac4284fcde8260ca7db9c81a678387c74d0679da268"
os.environ["TAVILY_API_KEY"] = "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"

# -------------------------
# LLM & Embeddings
# -------------------------
llm = ChatTogether(model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", temperature=0.2)
embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-32k-retrieval")

# -------------------------
# Prompts
# -------------------------
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

# -------------------------
# Helper Functions
# -------------------------
def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([^&?]+)", url)
    return match.group(1) if match else None

SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]

def get_youtube_client():
    """
    Returns an authenticated YouTube client using Streamlit secrets.
    Works locally and on Streamlit Cloud.
    """
    if "youtube_credentials" not in st.secrets:
        st.error("You must add your YouTube credentials to Streamlit secrets!")
        st.stop()

    creds_dict = st.secrets["youtube_credentials"]["web"]

    # Save credentials as temp JSON for InstalledAppFlow
    temp_file = "temp_credentials.json"
    with open(temp_file, "w") as f:
    # Convert AttrDict to regular dict before writing
      json.dump(dict(st.secrets["youtube_credentials"]["web"]), f)
    try:
        # Detect headless environment
        if os.environ.get("DISPLAY") is None and not os.name == "nt":
            flow = InstalledAppFlow.from_client_secrets_file(temp_file, SCOPES)
            creds = flow.run_console()
        else:
            flow = InstalledAppFlow.from_client_secrets_file(temp_file, SCOPES)
            creds = flow.run_local_server(port=0)
    except Exception as e:
        st.error(f"Failed to authenticate YouTube client: {e}")
        st.stop()

    youtube = build("youtube", "v3", credentials=creds)
    return youtube

def get_transcript(video_id, llm=None, translate_to_english=True):
    """Fetches captions of a YouTube video using OAuth credentials."""
    try:
        youtube = get_youtube_client()

        captions_response = youtube.captions().list(
            part="snippet",
            videoId=video_id
        ).execute()

        captions = captions_response.get("items", [])
        if not captions:
            return None

        caption_id, lang = None, None
        for cap in captions:
            if cap['snippet']['language'] in ['en', 'en-US']:
                caption_id = cap['id']
                lang = 'en'
                break
        if not caption_id:
            caption_id = captions[0]['id']
            lang = captions[0]['snippet']['language']

        caption_response = youtube.captions().download(
            id=caption_id,
            tfmt="srt"
        ).execute()

        srt_text = caption_response.decode("utf-8")
        transcript = re.sub(r"\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n", "", srt_text)
        transcript = re.sub(r"\n+", " ", transcript).strip()

        if translate_to_english and lang != 'en' and llm:
            prompt = f"Translate the following transcript into clear English:\n\n{transcript}"
            transcript = llm.invoke(prompt).content

        return transcript

    except HttpError as e:
        st.error(f"Error fetching captions: {e}")
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

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="🎨 AI Teaching Assistant", layout="centered")

st.markdown("""
<style>
.block-container { padding: 2rem 3rem; }
.stButton>button { border-radius: 0.5rem; background: linear-gradient(to right, #00c6ff, #0072ff); color: white; font-weight: bold; padding: 0.6rem 1.2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;color:#154360;'>🎓 AI Teaching Assistant</h1>", unsafe_allow_html=True)

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
            template="You are a helpful assistant. Use only the provided context to answer.\n\n{context}\nQuestion: {question}",
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
                template="You are a helpful assistant. Use only the provided context to answer.\n\n{context}\nQuestion: {question}",
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
