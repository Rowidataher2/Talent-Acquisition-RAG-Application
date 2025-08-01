import sys
import os



PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "offline_processing"))
sys.path.append(os.path.join(PROJECT_ROOT, "vector_storage"))
sys.path.append(os.path.join(PROJECT_ROOT, "online_processing"))
from retrieve import search_candidates, generate_response
from vectorization import prepare_for_vector_db, process_resume
from milvus_store import CVVectorStore
import streamlit as st

# 🔹 Streamlit UI
st.set_page_config(page_title="GenAI Resume Screening ChatBot", layout="wide")
 
st.title("🧠 GenAI Resume Screening ChatBot 🤖")
st.markdown("#### Ask for the best candidates based on job requirements.")
 
# 🔹 Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "recommended_candidates" not in st.session_state:
    st.session_state.recommended_candidates = []


# 🔹 Display chat history
st.subheader("📝 Chat History")
for entry in st.session_state.chat_history:
    st.markdown(f"**🔹 You:** {entry['query']}")
    st.markdown(f"**🤖 Bot:** {entry['response']}")
    st.markdown("---")
 
st.subheader("📝 Add Resumes")
uploaded_files = st.file_uploader("Upload DOCX/PDF Files", type=["pdf", "docx"], accept_multiple_files=True)
uploaded_folder = st.text_input("Or enter folder path:")

if st.button("Process File(s)"):
    st.write("Processing files...")

    all_files = []
    temp_folder = "temp_files"
    os.makedirs(temp_folder, exist_ok=True)

    # Save uploaded files to temp folder
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_folder, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            all_files.append(file_path)

    # Get files from folder if entered
    if uploaded_folder and os.path.exists(uploaded_folder):
        all_files.extend([os.path.join(uploaded_folder, f) for f in os.listdir(uploaded_folder) if f.endswith((".pdf", ".docx"))])

    if not all_files:
        st.warning("No valid files found.")
    else:
        st.write("🔄 Extracting text and storing in database...")
        store = CVVectorStore()

        for filename in all_files:
            # file_path = os.path.join(directory, filename)

            chunks, name = process_resume(filename)
            vector_chunks = prepare_for_vector_db(chunks, name)
            store.insert_cv_chunks(vector_chunks) 

            st.success(f"✅ Processed and stored {name}'s resume.")

        
        st.success("🚀 All files processed and stored in Milvus!")
        
# 🔹 User Input
st.markdown("---")
query = st.text_input("Enter job description or required skills:", placeholder="Looking for a Python Data Scientist")
 
if st.button("Find Candidates"):
    with st.spinner("Searching for the best candidates..."):
        candidates = search_candidates(query)
        response = generate_response(query, candidates)
 
    # 🔹 Update chat history
    st.session_state.chat_history.append({"query": query, "response": response})
 
    # 🔹 Display Current Response
    st.subheader("🎯 Recommended Candidates:")
    st.write(response)
 
    st.subheader("📋 Detailed Candidate Extracts:")
    for candidate in candidates:
        st.markdown(f"**Candidate: {candidate['name']}**")
        st.write(f"🔍 Match Score: {candidate['score']:.2f}")
        st.write(f"📄 Extract: {candidate['text']}")
        st.markdown("---")

