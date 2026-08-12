import os
import re
import numpy as np
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI



st.set_page_config(
    page_title="Corrective RAG System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Corrective RAG System")
st.write("Ask questions about the uploaded books.")



GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

embedding_model = HuggingFaceEmbeddings(

    model_name="BAAI/bge-small-en-v1.5",

    model_kwargs={
        "device":"cpu"
    },

    encode_kwargs={
        "normalize_embeddings":True
    }

)


vector_db = FAISS.load_local(

    "vector_database",

    embedding_model,

    allow_dangerous_deserialization=True

)



llm = ChatGoogleGenerativeAI(

    model="gemini-3.5-flash",

    google_api_key=GOOGLE_API_KEY,

    temperature=0

)


def retrieve(question, k=3):

    results = vector_db.similarity_search_with_score(

        question,

        k=k

    )

    return results




evaluation_prompt = """
You are a retrieval evaluator.

Question:
{question}

Retrieved Context:
{context}

Evaluate whether the retrieved context is enough to answer the question.

Return exactly:

Relevant: Yes or No
Score: number from 0 to 100
"""

def evaluate(question, retrieved):

    context = "\n\n".join(

        [doc.page_content for doc, score in retrieved]

    )

    prompt = evaluation_prompt.format(

        question=question,

        context=context

    )

    response = llm.invoke(prompt)

    text = response.content

    try:

        score = int(

            re.search(r"Score:\s*(\d+)", text).group(1)

        )

    except:

        score = 0

    return score



# ================================
# Rewrite Query
# ================================

rewrite_prompt = """
Rewrite the user's question to improve document retrieval.

Return ONLY the rewritten question.

Question:
{question}
"""

def rewrite(question):

    response = llm.invoke(
        rewrite_prompt.format(
            question=question
        )
    )

    # إذا كان الرد نصًا
    if isinstance(response.content, str):
        return response.content.strip()

    # إذا كان الرد قائمة
    if isinstance(response.content, list):

        text = ""

        for item in response.content:

            # AIMessageChunk أو TextPart
            if hasattr(item, "text"):
                text += item.text

            # Dictionary يحتوي على text
            elif isinstance(item, dict):
                text += item.get("text", "")

            # أي نوع آخر
            else:
                text += str(item)

        return text.strip()

    # أي نوع غير متوقع
    return str(response.content).strip()



answer_prompt = """
You are an AI assistant.

Answer ONLY using the provided context.

If the answer is not available, reply exactly:

I couldn't find the answer in the uploaded documents.

Context:

{context}

Question:

{question}

Answer:
"""

def generate_answer(question, retrieved):

    context = "\n\n".join(
        [doc.page_content for doc, score in retrieved]
    )

    prompt = answer_prompt.format(
        context=context,
        question=question
    )

    response = llm.invoke(prompt)

    if isinstance(response.content, str):
        return response.content.strip()

    if isinstance(response.content, list):

        text = ""

        for item in response.content:

            if hasattr(item, "text"):
                text += item.text

            elif isinstance(item, dict):
                text += item.get("text", "")

            else:
                text += str(item)

        return text.strip()

    return str(response.content).strip()



def corrective_rag(question):

    retrieved = retrieve(question)

    similarity_scores = [

        score

        for _, score in retrieved

    ]

    evaluator_score = evaluate(

        question,

        retrieved

    )

    rewritten = None

    if evaluator_score < 70:

        rewritten = rewrite(question)

        question = rewritten

        retrieved = retrieve(question)

        similarity_scores = [

            score

            for _, score in retrieved

        ]

        evaluator_score = evaluate(

            question,

            retrieved

        )

    similarity = np.mean(similarity_scores)

    confidence = round(

        evaluator_score * 0.6 +

        ((1-similarity)*100)*0.4,

        2

    )

    answer = generate_answer(

        question,

        retrieved

    )

    docs = [

        doc

        for doc, score in retrieved

    ]

    return answer, docs, confidence, rewritten




st.divider()

question = st.text_input(
    "Enter your question:",
    placeholder="Example: What is Newton's Second Law?"
)

if st.button("Ask"):

    if question.strip() == "":

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching documents..."):

            answer, docs, confidence, rewritten = corrective_rag(question)

        st.success("Answer Generated Successfully!")

        if rewritten:

            st.info(f"Rewritten Question: {rewritten}")

        st.subheader("Answer")

        st.write(answer)

        st.subheader("Confidence")

        st.progress(min(int(confidence), 100))

        st.write(f"{confidence}%")

        st.subheader("Sources")

        sources = sorted(
            list(
                {
                    doc.metadata["source"]
                    for doc in docs
                }
            )
        )

        for source in sources:

            st.write(f"📄 {source}")