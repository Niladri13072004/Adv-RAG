from streamlit.proto import openmetrics_data_model_pb2
import traceback
import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import Docx2txtLoader
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import plotly.express as px
import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI
import traceback
import math
import time
import shutil
from dataclasses import dataclass
from typing import Literal
import random
from groq import Groq
from functools import lru_cache
from src.config import OPENROUTER_API_KEY
from src.config import GEMINI_API_KEY
from src.config import HF_TOKEN
from src.config import GROQ_API_KEY

parent_docs = {}
chat_history = []
pending_clarification = None
conversation_summary = ""

MODEL_NAME = "openai/gpt-oss-20b:free"

ENABLE_REWRITING = True
ENABLE_SELF_REFLECTION = True
ENABLE_QUERY_REWRITE = True
ENABLE_SELF_CORRECTION = False
ENABLE_GENERAL_ROUTING = True
is_subquestion = False

ENABLE_MULTI_QUERY = False
ENABLE_RERANKING = True
ENABLE_PARENT_RETRIEVAL = True
ENABLE_CONTEXT_COMPRESSION = True
ENABLE_HUMAN_CLARIFICATION = False
ENABLE_ANSWER_VERIFICATION = True

judge_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    default_headers={
        "HTTP-Referer": "http://localhost:7860",
        "X-OpenRouter-Title": "Knowledge Worker Judge",
    },
)

judge_llm = ChatOllama(
    model="lukaspetrik/gemma3-tools:4b",
    temperature=0
)

local_llm = ChatOllama(
    model="lukaspetrik/gemma3-tools:4b",
    temperature=0
)

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

@dataclass(frozen=True)
class ModelConfig:
    provider: Literal["gemini", "ollama", "openrouter","groq"]
    model: str
    temperature: float = 0.0

TASK_MODELS = {
    "routing": ModelConfig("gemini", "gemini-3.1-flash-lite"),
    "query_rewrite": ModelConfig("gemini", "gemini-3.1-flash-lite"),
    "conversation_summary": ModelConfig("gemini", "gemini-2.5-flash-lite"),

    "multi_query": ModelConfig("gemini", "gemini-2.5-flash"),

    "answer_generation": ModelConfig( "gemini","gemini-3.5-flash"),
    "self_correction": ModelConfig("gemini","gemini-3.1-flash-lite"),

    "context_compression": ModelConfig("ollama","lukaspetrik/gemma3-tools:4b"),
    "self_evaluation": ModelConfig("groq","openai/gpt-oss-120b"),
    "verification": ModelConfig("gemini","gemini-3.1-flash-lite"),
    "llm_judge": ModelConfig("groq","qwen/qwen3-32b"),
}

TASK_FALLBACKS = {

    "routing": [
        ModelConfig(
            "openrouter",
            "openai/gpt-oss-20b:free"
        )
    ],

    "query_rewrite": [
        ModelConfig(
            "openrouter",
            "openai/gpt-oss-20b:free"
        )
    ],

    "conversation_summary": [
        ModelConfig(
            "openrouter",
            "openai/gpt-oss-20b:free"
        )
    ],

    "multi_query": [
        ModelConfig(
            "groq",
            "llama-3.3-70b-versatile"
        ),
        ModelConfig(
            "openrouter",
            "openai/gpt-oss-20b:free"
        )
    ],

    "answer_generation": [
        ModelConfig(
            "groq",
            "llama-3.3-70b-versatile"
        ),
        ModelConfig(
            "openrouter",
            "openai/gpt-oss-20b:free"
        )
    ],

    "self_correction": [
        ModelConfig(
            "groq",
            "llama-3.3-70b-versatile"
        ),
        ModelConfig(
            "openrouter",
            "openai/gpt-oss-20b:free"
        )
    ],

    "context_compression": [
        ModelConfig(
            "groq",
            "llama-3.3-70b-versatile"
        )
    ],

    "self_evaluation": [
        ModelConfig(
            "groq",
            "openai/gpt-oss-120b"
        ),
        ModelConfig(
            "openrouter",
            "openai/gpt-oss-20b:free"
        )
    ],

    "verification": [
        ModelConfig(
            "groq",
            "llama-3.3-70b-versatile"
        ),
        ModelConfig(
            "openrouter",
            "openai/gpt-oss-20b:free"
        )
    ],

    "llm_judge": [
        ModelConfig(
            "groq",
            "qwen/qwen3-32b"
        ),
        ModelConfig(
            "openrouter",
            "openai/gpt-oss-20b:free"
        )
    ]
}

OLLAMA_CLIENTS = {}
def _call_ollama(model, prompt, temperature=0.0):

    if model not in OLLAMA_CLIENTS:

        OLLAMA_CLIENTS[model] = ChatOllama(
            model=model,
            temperature=temperature
        )

    start = time.time()

    response = OLLAMA_CLIENTS[model].invoke(prompt)

    print(f"OLLAMA RAW INFERENCE: {time.time() - start:.2f} sec")

    return response.content.strip()

def _call_groq(model: str,prompt: str,temperature: float = 0.0) -> str:

    response = groq_client.chat.completions.create(

        model=model,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=temperature,
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError(
            "Groq returned empty content"
        )

    return content.strip()

def _call_openrouter(model: str, prompt: str, temperature: float = 0.0) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("OpenRouter returned empty content")

    return content.strip()

from google import genai
from google.genai import types

groq_client = Groq(
    api_key=GROQ_API_KEY
)

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

def _call_gemini(model, prompt, temperature=0.0):

    if not model.startswith("models/"):
        model = f"models/{model}"

    response = gemini_client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature
        )
    )

    return response.text.strip()

def call_llm(task: str, prompt: str) -> str:

    config = TASK_MODELS[task]
    fallback_chain = TASK_FALLBACKS.get(task, [])

    print("\n" + "=" * 50)
    print(f"TASK      : {task}")
    print(f"PROVIDER  : {config.provider}")
    print(f"MODEL     : {config.model}")
    print("=" * 50)

    last_error = None

    # ==========================
    # Try Primary Model First
    # ==========================
    try:

        if config.provider == "gemini":
            return _call_gemini(
                config.model,
                prompt,
                config.temperature
            )

        elif config.provider == "groq":
            return _call_groq(
                config.model,
                prompt,
                config.temperature
            )

        elif config.provider == "ollama":
            return _call_ollama(
                config.model,
                prompt,
                config.temperature
            )

        else:
            return _call_openrouter(
                config.model,
                prompt,
                config.temperature
            )

    except Exception as e:

        last_error = e

        print(f"{task} failed: {e}")

    # ==========================
    # Try Fallback Chain
    # ==========================
    for fallback in fallback_chain:

        # Don't retry the same model
        if (
            fallback.provider == config.provider
            and fallback.model == config.model
        ):
            continue

        print("\n==============================")
        print("USING FALLBACK MODEL")
        print("==============================")
        print("Provider :", fallback.provider)
        print("Model    :", fallback.model)

        try:

            if fallback.provider == "gemini":
                return _call_gemini(
                    fallback.model,
                    prompt,
                    fallback.temperature
                )

            elif fallback.provider == "groq":
                return _call_groq(
                    fallback.model,
                    prompt,
                    fallback.temperature
                )

            elif fallback.provider == "ollama":
                return _call_ollama(
                    fallback.model,
                    prompt,
                    fallback.temperature
                )

            else:
                return _call_openrouter(
                    fallback.model,
                    prompt,
                    fallback.temperature
                )

        except Exception as e:

            last_error = e

            print("Fallback failed:", e)

    raise RuntimeError(
        f"LLM task '{task}' failed: {last_error}"
    )
            
def generate_answer(prompt):
    """
    Generate answer from the selected LLM.
    Also performs light post-processing.
    """

    response = call_llm(
        task="answer_generation",
        prompt=prompt
    )

    if not isinstance(response, str):
        response = str(response)

    response = response.strip()

    # Remove markdown fences
    if response.startswith("```"):
        response = response.replace("```", "").strip()

    # Remove leading "Answer:"
    if response.lower().startswith("answer:"):
        response = response[7:].strip()

    # Remove surrounding quotes
    response = response.strip('"').strip("'")

    return response

def answer_general_question(question):

    prompt = f"""
    Answer the following question.

    Question:
    {question}
    """

    return generate_answer(
        prompt
    )

def self_evaluate(question,context,answer):
    
    print("=" * 60)
    print("SELF EVAL CONTEXT LENGTH:", len(context))
    print("=" * 60)

    prompt = f"""
        Evaluate the RAG answer.

        Question:
        {question}

        Context:
        {context[:2000]}

        Answer:
        {answer}

        Return ONLY valid JSON:

        {{
          "supported": true,
          "complete": true,
          "confidence": 0-10,
          "feedback": "short explanation"
        }}
    """

    try:

        response = call_llm(
            task="self_evaluation",
            prompt=prompt
        )
        
        print("\nRAW SELF EVAL")
        print(response)

        return _extract_json_object(response)

    except Exception as e:

        print(
            "\nSELF EVAL ERROR:"
        )

        print(str(e))

        return {
            "supported": True,
            "complete": True,
            "confidence": 10,
            "feedback": ""
        }

@lru_cache(maxsize=512)
def classify_question(question):

    prompt = f"""
    You are a RAG query classifier.

    Analyze the user's question and return ONLY valid JSON.

    Question:
    {question}

    Classify the following:

    1. question_source

    Choose ONE:

    DOCUMENT
    GENERAL

    DOCUMENT:
    Questions requiring uploaded documents.

    GENERAL:
    General knowledge.

    ------------------------------------

    2. query_type

    Choose ONE:

    factual
    definition
    concept
    numerical
    spanning
    holistic
    reasoning

    Definitions:

    factual
    Questions asking for names, lists, titles, organizations,
    projects, technologies, people, locations or other explicitly
    stated facts.

    Examples:
    - What is the college name?
    - Which programming languages are listed?
    - Which backend framework is used?
    - What LeetCode badge was achieved?

    definition
    Questions asking "What is ..."

    concept
    Questions asking why, purpose, role or explanation.

    numerical
    Questions asking for numbers, dates,
    percentages, measurements or calculations.

    spanning
    Answer requires combining two or more sections.

    holistic
    Needs summarizing an entire document.

    reasoning
    Requires inference using multiple facts.

    Return ONLY ONE valid JSON object.

    You MUST include ALL THREE keys.

    {{
        "question_source": "DOCUMENT or GENERAL",
        "query_type": "factual | definition | concept | numerical | spanning | holistic | reasoning",
        "intent": "fact_lookup | comparison | summarization | analysis | troubleshooting | howto | find_doc"
    }}

    Never omit any key.
    Never add explanations.
    Never add markdown.
    Never return anything except the JSON object.
    """

    response = call_llm(
        task="routing",
        prompt=prompt
    )

    print("\nRAW CLASSIFICATION")
    print(response)

    classification = _extract_json_object(response)

    question_lower = question.lower()

    document_keywords = [
        "niladri",
        "resume",
        "cgpa",
        "leetcode",
        "codechef",
        "project",
        "internship",
        "college",
        "skill",
        "typing shooter",
        "compressor",
        "refrigeration",
        "compressor efficiency",
        "compression",
        "air compressor"
    ]

    if any(keyword in question_lower for keyword in document_keywords):
        classification["question_source"] = "DOCUMENT"

    classification.setdefault(
        "question_source",
        "DOCUMENT"
    )

    classification.setdefault(
        "query_type",
        "concept"
    )

    classification.setdefault(
        "intent",
        "fact_lookup"
    )

    classification["question_source"] = (
        classification["question_source"].upper()
    )

    classification["query_type"] = (
        classification["query_type"].lower()
    )

    classification["intent"] = (
        classification["intent"].lower()
    )

    return classification

def self_correct_answer(
    question,
    context,
    answer,
    feedback 
    ):

    prompt = f"""
    You are improving an answer.

    Question:
    {question}

    Context:
    {context}

    Current Answer:
    {answer}

    Review Feedback:
    {feedback}

    Improve the answer using ONLY the context.

    Rules:
    - Keep correct information.
    - Fix inaccuracies.
    - Add missing information.
    - Preserve exact numbers.
    - Do not hallucinate.

    Return only the improved answer.
    """

    response = call_llm(
        task="self_correction",
        prompt=prompt
    )

    return response

def verify_answer(
    question,
    context,
    answer
    ):

    prompt = f"""
You are a fact checking system.

Question:
{question}

Context:
{context}

Answer:
{answer}

Return ONLY valid JSON.

{{
    "supported": true,
    "confidence": 10,
    "feedback": "short explanation"
}}
"""

    try:

        response = call_llm(
            task="verification",
            prompt=prompt
        )

        print("\nRAW VERIFY")
        print(response)

        return _extract_json_object(response)

    except Exception as e:

        print("\nVERIFY ERROR:")
        print(str(e))

        return {
            "supported": True,
            "confidence": 10,
            "feedback": "Verification failed"
        }

def improve_answer(
    question,
    context,
    answer,
    feedback
    ):

    prompt = f"""
    Question:
    {question}

    Context:
    {context}

    Previous Answer:
    {answer}

    Feedback:
    {feedback}

    Generate a better answer.

    Use only the context.
    """

    return generate_answer(
        prompt
    )

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
splitter = MarkdownTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

embedding = None

def get_embedding_model():
    global embedding

    if embedding is None:
        embedding = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )

    return embedding

def rewrite_chunk(text):

        prompt = f"""
    Rewrite the following text for retrieval.

    Rules:
    1. Preserve all facts.
    2. Add clarifying context.
    3. Expand abbreviations.
    4. Add searchable keywords.
    5. Do NOT invent information.

    Text:
    {text}
    """

knowledge_bases = {}
vectordb = None
retriever = None
bm25 = None
all_chunks = None

def process_document(files):
    global vectordb
    global retriever
    global bm25
    global all_chunks
    global parent_docs
    global conversation_summary

    print("process_document called")

    try:
        vectordb = None
        retriever = None
        bm25 = None
        all_chunks = None

        parent_docs.clear()
        chat_history.clear()
        pending_clarification = None
        conversation_summary = ""

        if not files:
            return "No files uploaded."

        all_docs = []
        parent_id = 0

        for file in files:
            file_name = file.name.lower()

            if file_name.endswith(".pdf"):
                loader = PyPDFLoader(file.name)
            elif file_name.endswith(".docx"):
                loader = Docx2txtLoader(file.name)
            else:
                continue

            docs = loader.load()

            for doc in docs:
                doc.metadata["filename"] = os.path.basename(file.name)
                doc.metadata["parent_id"] = parent_id
                parent_docs[
                    doc.metadata["parent_id"]
                ] = {

                    "content": doc.page_content,

                    "page": doc.metadata.get(
                        "page",
                        None
                    ),

                    "filename": os.path.basename(
                        file.name
                    )
                }
                parent_id += 1

            all_docs.extend(docs)

        if not all_docs:
            return "No supported content found in the uploaded files."

        chunks = splitter.split_documents(all_docs)

        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx

            chunk.metadata.setdefault(
                "filename",
                chunk.metadata.get(
                    "source",
                    "unknown"
                )
            )

            chunk.metadata.setdefault(
                "document_name",
                chunk.metadata["filename"]
            )

            chunk.metadata.setdefault(
                "parent_id",
                idx
            )
            chunk.metadata["section"] = detect_section(
                chunk.page_content
            )
            chunk.metadata["chunk_id"] = f"chunk_{idx}"

            chunk.metadata["token_count"] = len(
                chunk.page_content.split()
            )

            chunk.metadata["char_count"] = len(
                chunk.page_content
            )
        
        if ENABLE_REWRITING:
            print("Rewriting Chunks...")
            for chunk in chunks:
                try:
                    # rewritten = rewrite_chunk(chunk.page_content)
                    # if not rewritten:
                    #     continue

                    chunk.metadata["original_text"] = chunk.page_content
                    # chunk.metadata["retrieval_text"] = rewritten
                    # chunk.page_content = rewritten

                except Exception:
                    print("\nREWRITE ERROR")
                    traceback.print_exc()

        all_chunks = chunks

        if not chunks:
            return "No chunks were created from the uploaded files."

        tokenized_corpus = [
            chunk.page_content.lower().split()
            for chunk in chunks
        ]
        bm25 = BM25Okapi(tokenized_corpus)

        print("Total Chunks Created:", len(chunks))

        if os.path.exists("chroma_db"):
            shutil.rmtree("chroma_db")

        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=get_embedding_model(),
        )

        knowledge_bases["default"] = vectordb

        retriever = vectordb.as_retriever(
            search_kwargs={"k": 5}
        )

        print("retriever created")
        return f"""
        Documents Loaded: {len(files)}

        Chunks Created: {len(chunks)}

        Knowledge Base Ready
        """

    except Exception:
        error = traceback.format_exc()
        print(error)
        return error

def chat(message, history):
    global retriever

    if history is None:
        history = []

    if retriever is None:
        history.append(
            (
                message,
                "Please upload documents first."
            )
        )
        return history, history, ""

    try:

        start = time.time()

        print("CALLING answer_question")

        result = answer_question(message)

        print("ANSWER RECEIVED")

        answer = result["answer"]
        combined_docs = result["sources"]

        sources = []

        for doc in combined_docs:

            filename = doc.metadata.get(
                "filename",
                "Unknown File"
            )

            page = doc.metadata.get("page")

            if page is not None:
                sources.append(
                    f"{filename} - Page {page + 1}"
                )

        source_text = "\n".join(
            sorted(set(sources))
        )

        final_answer = answer

        if source_text:

            final_answer += (
                "\n\nSources:\n"
                + source_text
            )

        print(
            f"LLM Time: {time.time()-start:.2f} sec"
        )

        history.append({
            "role": "user",
            "content": message
        })

        history.append({
            "role": "assistant",
            "content": final_answer
        })

        return history, history, ""

    except Exception as e:

        print(e)

        history.append(
            (
                message,
                f"Error: {str(e)}"
            )
        )

        return history, history, ""

def detect_ambiguity(question):

    prompt = f"""
    Determine whether the question truly requires clarification.

    Only mark ambiguous if it CANNOT be answered.

    Examples:

    Question:
    Can you provide a summary of Niladri's technical projects?

    Answer:
    {{
        "ambiguous": false,
        "clarification": ""
    }}

    Question:
    What percentage did Niladri score in Higher Secondary?

    Answer:
    {{
        "ambiguous": false,
        "clarification": ""
    }}

    Question:
    How do I update it?

    Answer:
    {{
      "ambiguous": true,
      "clarification": "What does 'it' refer to?"
    }}

    Question:
    Compare them.

    Answer:
    {{
        "ambiguous": true,
        "clarification": "Which items would you like compared?"
    }}

    Question:
    Which skills listed in the resume were applied in the Conversational RAG Knowledge Assistant project?

    Answer:
    {{
        "ambiguous": false,
        "clarification": ""
    }}

    Question:
    Can you provide a summary of Niladri's technical projects?

    Answer:
    {{
    "ambiguous": false,
    "clarification": ""
    }}

    Question:
    {question}
    """

    try:

        response = local_llm.invoke(
            prompt
        )

        return _extract_json_object(
            response.content
        )

    except Exception:

        return {
            "ambiguous": False,
            "clarification": ""
        }
        
def bm25_search(query, top_k=3):

    global bm25
    global all_chunks

    if bm25 is None:
        return []

    tokenized_query = query.split()

    scores = bm25.get_scores(
        tokenized_query
    )

    ranked = sorted(
        zip(all_chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        chunk
        for chunk, score
        in ranked[:top_k]
    ]

def reciprocal_rank_fusion(vector_docs,bm25_docs,k=60):
    """
    Reciprocal Rank Fusion

    https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
    """

    scores = {}

    for rank, doc in enumerate(vector_docs):

        key = doc.page_content

        if key not in scores:
            scores[key] = {
                "doc": doc,
                "score": 0
            }

        scores[key]["score"] += 1 / (k + rank + 1)

    for rank, doc in enumerate(bm25_docs):

        key = doc.page_content

        if key not in scores:
            scores[key] = {
                "doc": doc,
                "score": 0
            }

        scores[key]["score"] += 1 / (k + rank + 1)

    ranked = sorted(
        scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return [
        item["doc"]
        for item in ranked
    ]

reranker = None

def get_reranker():
    global reranker

    if reranker is None:
        reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    return reranker

def rerank_documents(query, docs, top_k=3):
    if not docs:
        return []
    
    pairs = [
        (query, doc.page_content)
        for doc in docs
    ]

    scores = get_reranker().predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        doc
        for doc, score in ranked[:top_k]
    ]

def answer_question(question, is_subquestion=False):
    global retriever
    global chat_history
    global conversation_summary
    global all_chunks

    routing_time = 0.0
    rewrite_time = 0.0
    retrieval_time = 0.0
    generation_time = 0.0

    try:
        # =====================
        # PREPROCESS
        # =====================
        question = normalize_query(question)
        question = strip_prompt_injection(question)
        question = expand_acronyms(question)

        # =====================
        # ROUTING
        # =====================
        stage = time.time()
        classification = classify_question(question)

        question_source = classification.get("question_source", "DOCUMENT")
        query_type = classification.get("query_type", "concept")
        intent = classification.get("intent", "fact_lookup")

        routing_time = time.time() - stage

        print(f"Routing Time: {routing_time:.2f} sec")
        print("QUESTION SOURCE:", question_source)
        print("QUERY TYPE:", query_type)
        print("INTENT:", intent)

        # =====================
        # GENERAL ROUTING
        # =====================
        if (
            ENABLE_GENERAL_ROUTING
            and question_source == "GENERAL"
            and not is_subquestion
        ):
            answer = answer_general_question(question)

            chat_history.append((question, answer))

            return {
                "answer": answer,
                "sources": [],
                "routing_time": routing_time,
                "rewrite_time": 0.0,
                "retrieval_time": 0.0,
                "generation_time": 0.0,
                "query_type": query_type,
                "intent": intent,
                "question_source": question_source,
            }

        # =====================
        # DOCUMENT CHECK
        # =====================
        if retriever is None:
            return {
                "answer": "Please upload documents first.",
                "sources": [],
                "routing_time": routing_time,
                "rewrite_time": 0.0,
                "retrieval_time": 0.0,
                "generation_time": 0.0,
                "query_type": query_type,
                "intent": intent,
                "question_source": question_source,
            }

        # =====================
        # DYNAMIC K
        # =====================
        k = 5

        # =====================
        # RETRIEVAL PARAMETERS
        # =====================

        if query_type == "factual":
            retrieve_k = 12
            rerank_k = 3

        elif query_type == "definition":
            retrieve_k = 10
            rerank_k = 3

        elif query_type == "numerical":
            retrieve_k = 10
            rerank_k = 3

        elif query_type == "concept":
            retrieve_k = 15
            rerank_k = 5

        elif query_type == "reasoning":
            retrieve_k = 15
            rerank_k = 5

        elif query_type == "spanning":
            retrieve_k = 18
            rerank_k = 6

        elif query_type == "holistic":
            retrieve_k = 20
            rerank_k = 8

        else:
            retrieve_k = 12
            rerank_k = 5

        print(f"Retrieve K : {retrieve_k}")
        print(f"Rerank K   : {rerank_k}")

        # =====================
        # QUERY REWRITE
        # =====================
        stage = time.time()

        rewritten_query = question

        # Skip rewriting for very direct factual lookup questions
        if ENABLE_REWRITING and query_type not in ["factual", "numerical"]:
            rewritten_query = rewrite_query(question)

        print("=" * 60)
        print("ORIGINAL:", question)
        print("REWRITTEN:", rewritten_query)
        print("=" * 60)

        rewrite_time = time.time() - stage
        print(f"Rewrite Time: {rewrite_time:.2f} sec")

        # =====================
        # RETRIEVAL
        # =====================

        stage = time.time()

        queries = [rewritten_query]

        if (
            ENABLE_MULTI_QUERY
            and all_chunks is not None
            and len(all_chunks) > 10
        ):
            queries = generate_search_queries(rewritten_query)

        if not queries:
            queries = [rewritten_query]

        all_docs = []

        for query in queries:

            if query_type in ["factual", "numerical", "definition"]:
                retrieve_k = 12
            else:
                retrieve_k = 10

            retriever.search_kwargs["k"] = retrieve_k

            vector_docs = retriever.invoke(query)

            bm25_k = max(5, retrieve_k // 2)
            bm25_docs = bm25_search(query, top_k=bm25_k)

            print("=" * 60)
            print("Vector Docs:", len(vector_docs))
            print("BM25 Docs:", len(bm25_docs))
            print("=" * 60)

            combined_docs = reciprocal_rank_fusion(vector_docs,bm25_docs)

            print("=" * 60)
            print("RRF TOP CHUNKS")
            print("=" * 60)

            for i, doc in enumerate(combined_docs[:5]):
                print(f"\nChunk {i+1}")
                print(doc.page_content[:200])

            all_docs.extend(combined_docs)

        # =====================
        # OPTIONAL NEIGHBOR/PARENT EXPANSION
        # =====================
        if ENABLE_PARENT_RETRIEVAL and query_type in ["holistic", "spanning", "analysis"]:
            print("Using Parent Retrieval")
            combined_docs = expand_to_parent_docs(combined_docs)

        # =====================
        # RERANK
        # =====================
        if ENABLE_RERANKING and combined_docs:
            combined_docs = rerank_documents(
                question,
                combined_docs,
                top_k=rerank_k
            )
            print_retrieval_debug(combined_docs)
            print("\nTOP RETRIEVED CHUNK")
            print(combined_docs[0].page_content[:500])

        if ENABLE_PARENT_RETRIEVAL and query_type in ["holistic", "spanning", "analysis"]:
            combined_docs = expand_neighbor_chunks(combined_docs)

        print("After Reranking:", len(combined_docs))

        # =====================
        # DOCUMENT FILTER
        # =====================

        print("=" * 60)
        print("Combined Docs:", len(combined_docs))

        if combined_docs:
            print("\nFIRST CHUNK:")
            print(combined_docs[0].page_content[:500])
        else:
            print("NO DOCUMENTS RETRIEVED")
        print("=" * 60)

        # =====================
        # CONTEXT
        # =====================
        context = "\n\n".join(doc.page_content for doc in combined_docs)

        print("=" * 60)
        print("Context BEFORE compression:", len(context))
        print(context[:500])
        print("=" * 60)

        if (
            ENABLE_CONTEXT_COMPRESSION
            and query_type in ["holistic", "analysis", "spanning"]
            and len(context) > 6000
        ):
            print("Compressing Context...")
            context = compress_context(question, combined_docs)
            print("=" * 60)
            print("Context AFTER compression:", len(context))
            print(context[:500])
            print("=" * 60)

        # Adaptive context length by query type
        if query_type in ["factual", "definition", "numerical"]:
            combined_docs = combined_docs[:3]

        elif query_type in ["concept", "reasoning"]:
            combined_docs = combined_docs[:5]

        elif query_type in ["spanning", "holistic"]:
            combined_docs = combined_docs[:8]

        context = "\n\n".join(
            doc.page_content
            for doc in combined_docs
        )

        # =====================
        # HISTORY
        # =====================
        history_text = ""

        # =====================
        # NORMAL BRANCH
        # =====================
        print("ENTERED NORMAL BRANCH")

        prompt = f"""

    You are a Retrieval-Augmented Generation (RAG) assistant.

    Answer ONLY using the retrieved context below.

    If the answer is explicitly present, answer it directly.

    If the answer requires combining two or more retrieved facts, combine them.

    Do NOT use outside knowledge.

    Reply exactly "Not found in document." ONLY when the retrieved context truly does not contain enough information.

    Rules:

    - Preserve names exactly.
    - Preserve numbers exactly.
    - Preserve units exactly.
    - Preserve dates exactly.
    - Never invent information.
    - Never explain your reasoning.
    - Never mention the context.
    - Return concise answers.

    Question Type:
    {query_type}

    Intent:
    {intent}

    Conversation Summary:
    {conversation_summary}

    Recent History:
    {history_text}

    Retrieved Context:
    {context}

    Question:
    {question}

    Answer:
    """

        start = time.time()

        answer = generate_answer(prompt)
        answer = normalize_answer(answer)

        generation_time = time.time() - start
        print(f"Generation Time: {generation_time:.2f} sec")

        if ENABLE_SELF_REFLECTION:
            stage = time.time()

            evidence = "\n\n".join(
                doc.page_content
                for doc in combined_docs[:2]
            )

            review = self_evaluate(
                question,
                evidence,
                answer
            )

            print(f"Self Evaluation Time: {time.time() - stage:.2f} sec")
        else:
            review = {
                "supported": True,
                "complete": True,
                "confidence": 10,
                "feedback": ""
            }

        if ENABLE_SELF_CORRECTION and review["confidence"] < 7:
            answer = self_correct_answer(
                question,
                context,
                answer,
                review["feedback"]
            )
            review = self_evaluate(
                question,
                context,
                answer
            )

        if ENABLE_ANSWER_VERIFICATION:
            print("=" * 60)
            print("VERIFY CONTEXT")
            print(context[:1000])
            print("=" * 60)

            verification = verify_answer(
                question,
                context,
                answer
            )

            print("Verification:", verification["supported"])

        llm_time = time.time() - start

        print(f"LLM Time: {llm_time:.2f} sec")
        print("Self Evaluation Confidence:", review["confidence"])

        chat_history.append((question, answer))

        if len(chat_history) >= 10:
            summarize_conversation()
            chat_history = chat_history[-3:]

        return {
            "answer": answer,
            "sources": combined_docs,
            "routing_time": routing_time,
            "rewrite_time": rewrite_time,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "query_type": query_type,
            "intent": intent,
            "question_source": question_source,
        }

    except Exception as e:
        print("\nANSWER QUESTION ERROR:")
        print(str(e))
        traceback.print_exc()

        return {
            "answer": "Answer Generation Failed",
            "sources": [],
            "routing_time": 0,
            "rewrite_time": 0,
            "retrieval_time": 0,
            "generation_time": 0,
            "query_type": "",
            "intent": "",
            "question_source": "",
        }

def normalize_answer(answer):

    if not answer:
        return answer

    answer = answer.strip()

    replacements = {
        "Fast API": "FastAPI",
        "Fast Api": "FastAPI",
        "500+": "500",
        "Github": "GitHub",
        "Javascript": "JavaScript",
        "Node Js": "Node.js",
        "Express Js": "Express.js",
        "Url": "URL",
        "Url Shortener": "URL Shortener",
    }

    for old, new in replacements.items():
        answer = answer.replace(old, new)

    answer = re.sub(r"\s+", " ", answer)

    return answer.strip()

def expand_neighbor_chunks(docs):

    expanded = []
    seen = set()

    for doc in docs:

        idx = doc.metadata["chunk_index"]

        for offset in [-1, 0, 1]:

            new_idx = idx + offset

            if (
                0 <= new_idx < len(all_chunks)
                and new_idx not in seen
            ):

                expanded.append(
                    all_chunks[new_idx]
                )

                seen.add(new_idx)

    return expanded

def detect_section(text):

    lines = text.splitlines()

    for line in lines[:10]:

        line = line.strip()

        if len(line) < 80:

            if re.match(
                r"^[A-Z][A-Za-z0-9\s&/-]+$",
                line
            ):
                return line

    return "Unknown"

@lru_cache(maxsize=512)
def generate_search_queries(question):
    print("MULTI QUERY CALLED")
    prompt = f"""
    Generate 3 different search queries
    for retrieving information.

    Question:
    {question}

    Return one query per line.
    """

    response = call_llm(
        task="multi_query",
        prompt=prompt
    )
    text = response
    queries = [
        q.strip()
        for q in text.split("\n")
        if q.strip()
    ]

    return queries[:3]

def postprocess_answer(answer):

    answer = answer.strip()

    answer = answer.replace(
        "Fast API",
        "FastAPI"
    )

    answer = answer.replace(
        "Node JS",
        "Node.js"
    )

    answer = answer.replace(
        "Express JS",
        "Express.js"
    )

    return answer

def decompose_question(question):

    prompt = f"""
    Break this question into
    independent sub-questions.

    Question:
    {question}

    Return one per line.
    """

    response = local_llm.invoke(prompt)

    return [
        q.strip()
        for q in response.content.split("\n")
        if q.strip()
    ]

def load_test_set(filepath):

    with open(filepath, "r", encoding="utf-8") as f:
        test_questions = json.load(f)

    print(
        f"Loaded {len(test_questions)} test questions"
    )

    return test_questions

def evaluate_retrieval(item, retrieved_docs):

    keywords = item["keywords"]

    retrieved_text = " ".join(
        doc.page_content.lower()
        for doc in retrieved_docs
    )

    matched = 0

    for keyword in keywords:

        if keyword.lower() in retrieved_text:
            matched += 1

    coverage = matched / len(keywords)

    reciprocal_rank = 0

    for rank, doc in enumerate(
        retrieved_docs,
        start=1
    ):

        text = doc.page_content.lower()

        if any(
            keyword.lower() in text
            for keyword in keywords
        ):

            reciprocal_rank = 1 / rank

            break

    recall = matched > 0

    return {
        "coverage": coverage,
        "mrr": reciprocal_rank,
        "recall": recall
    }

def _extract_json_object(text: str) -> dict:
    """
    Tries to parse strict JSON first.
    If the model adds extra text, extracts the first JSON object.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))

def judge_answer(question: str,reference_answer: str,generated_answer: str) -> dict:

    prompt = f"""
    You are a strict evaluation judge.

    Question:
    {question}

    Reference Answer:
    {reference_answer}

    Generated Answer:
    {generated_answer}

    Scoring Rules:

    10 = Perfect

    9 = Excellent

    8 = Very Good

    7 = Good

    6 = Acceptable

    5 = Partially Correct

    4 = Weak

    3 = Poor

    2 = Very Poor

    1 = Completely Incorrect

    0 = No Answer

    Return ONLY JSON:

    {{
        "accuracy": 0-10,
        "completeness": 0-10,
        "relevance": 0-10,
        "feedback": "short explanation"
    }}
    """
    

    response = call_llm(
        task="llm_judge",
        prompt=prompt
    )
    
    content = response.strip()

    result = _extract_json_object(content)

    print("\nRAW JUDGE JSON")
    print(result)

    return result

total_response_time = 0

def evaluate():
    ENABLE_HUMAN_CLARIFICATION = False
    global conversation_summary

    test_questions = load_test_set(
        "test.json"
    )

    conversation_summary = ""
    chat_history.clear()

    correct = 0
    total_coverage = 0
    total_mrr = 0
    total_recall = 0

    total_judge_accuracy = 0
    total_judge_completeness = 0
    total_judge_relevance = 0
    total_response_time = 0
    failed_questions = []

    category_stats = {}
    detailed_results = []
    for item in test_questions:

        start = time.time()

        result = answer_question(item["question"])

        generated_answer = result["answer"]
        combined_docs = result["sources"]

        response_time = (
            result["routing_time"]
            + result["rewrite_time"]
            + result["retrieval_time"]
            + result["generation_time"]
        )

        routing_time = result["routing_time"]
        rewrite_time = result["rewrite_time"]
        retrieval_time = result["retrieval_time"]
        generation_time = result["generation_time"]

        query_type = result["query_type"]
        intent = result["intent"]
        question_source = result["question_source"]

        response_time = time.time() - start
        
        total_response_time += response_time
        
        print("\nAFTER answer_question()")
        print(repr(generated_answer))

        chat_history.clear()
        
        try:
            print("\nBEFORE JUDGE")
            print(repr(generated_answer))

            judge = judge_answer(
                item["question"],
                item["reference_answer"],
                generated_answer
            )

        except Exception as e:

            print("Judge Error:", e)

            judge = {
                "accuracy": 0,
                    "completeness": 0,
                    "relevance": 0,
                    "feedback": "Judge failed"
                }

        total_judge_accuracy += judge["accuracy"]
        total_judge_completeness += judge["completeness"]
        total_judge_relevance += judge["relevance"]

        print("\n-------------------")
        print("\nBEFORE FINAL REPORT")
        print(repr(generated_answer))

        print("Question:", item["question"])
        print("Expected:", item["reference_answer"])
        print("Actual:", generated_answer)

        print("\nJUDGE SCORES")
        print("Accuracy:", judge["accuracy"])
        print("Completeness:", judge["completeness"])
        print("Relevance:", judge["relevance"])
        print("Feedback:", judge["feedback"])

        metrics = evaluate_retrieval(
            item,
            combined_docs
        )
        
        total_coverage += metrics["coverage"]
        total_mrr += metrics["mrr"]
        total_recall += metrics["recall"]
        
        print(
            f"Coverage={metrics['coverage']:.2f}"
            )

        print(
            f"MRR={metrics['mrr']:.2f}"
            )

        print(
            f"Recall={metrics['recall']}"
            )

        keywords = item["keywords"]
        category = item["category"]

        if category not in category_stats:

            category_stats[category] = {
                "correct": 0,
                "total": 0,
                "total_time": 0.0,
                "judge_accuracy": 0,
                "judge_completeness": 0,
                "judge_relevance": 0
            }

        category_stats[category]["total"] += 1
        category_stats[category]["total_time"] += response_time
        category_stats[category]["judge_accuracy"] += judge["accuracy"]
        category_stats[category]["judge_completeness"] += judge["completeness"]
        category_stats[category]["judge_relevance"] += judge["relevance"]

        if metrics["recall"]:

            category_stats[category]["correct"] += 1

        detailed_results.append(
            {
                "question": item["question"],

                "category": category,

                "expected": item["reference_answer"],

                "actual": generated_answer,

                "judge_accuracy": judge["accuracy"],

                "judge_completeness":judge["completeness"],

                "judge_relevance": judge["relevance"],

                "judge_feedback": judge["feedback"],
                
                "response_time": response_time,
                
                "routing_time": routing_time,
                "rewrite_time": rewrite_time,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "response_time": response_time,

                "predicted_query_type": query_type,
                "predicted_intent": intent,
                "predicted_source": question_source,

                "retrieved_chunks": [
                    doc.page_content[:500]
                    for doc in combined_docs
                ]
            }
        )

        semantic_result = judge

        score = (
            (
                semantic_result["accuracy"]
                +
                semantic_result["completeness"]
                +
                semantic_result["relevance"]
            )
            / 30
        )

        if score >= 0.7:

            correct += 1

            category_stats[category]["correct"] += 1

        else:

            failed_questions.append(
                {
                    "question": item["question"],
                    "expected": item["reference_answer"],
                    "actual": generated_answer,
                    "category": category
                }
            )

    accuracy = (
        correct /
        len(test_questions)
    ) * 100

    print(
        f"\nAccuracy: {accuracy:.2f}%"
    )

    print("\n========================")
    print("CATEGORY PERFORMANCE")
    print("========================")

    for category, stats in category_stats.items():

        category_accuracy = (
            stats["correct"] /
            stats["total"]
        ) * 100

    avg_coverage = (total_coverage /len(test_questions))

    avg_time = stats["total_time"] / stats["total"]

    avg_acc = stats["judge_accuracy"] / stats["total"]

    avg_comp = stats["judge_completeness"] / stats["total"]

    avg_rel = stats["judge_relevance"] / stats["total"]

    print(f"\n{category}")

    print(f"Accuracy : {category_accuracy:.2f}%")

    print(f"Average Time : {avg_time:.2f} sec")

    print(f"Judge Accuracy : {avg_acc:.2f}/10")

    print(f"Judge Completeness : {avg_comp:.2f}/10")

    print(f"Judge Relevance : {avg_rel:.2f}/10")
    
    avg_mrr = (total_mrr /len(test_questions))

    recall_at_k = (total_recall /len(test_questions))
    
    print("\n========================")
    print("RETRIEVAL PERFORMANCE")
    print("========================")

    print(
        f"Keyword Coverage: "
        f"{avg_coverage:.2%}"
    )

    print(
        f"Recall@K: "
        f"{recall_at_k:.2%}"
    )

    print(
        f"MRR: "
        f"{avg_mrr:.4f}"
    )

    print("\n========================")
    print("FAILED QUESTIONS")
    print("========================")

    for item in failed_questions:

        print("\nQuestion:")
        print(item["question"])

        print("\nCategory:")
        print(item["category"])

        print("\nExpected:")
        print(item["expected"])

        print("\nActual:")
        print(item["actual"])

        print("\n-------------------")

    n = len(test_questions)

    print("\n========================")
    print("JUDGE PERFORMANCE")
    print("========================")

    print(f"Average Accuracy Score: {total_judge_accuracy / n:.2f}/10")
    print(f"Average Completeness Score: {total_judge_completeness / n:.2f}/10")
    print(f"Average Relevance Score: {total_judge_relevance / n:.2f}/10")

    avg_response_time = total_response_time / n

    print("\n========================")
    print("LATENCY")
    print("========================")

    print(f"Average Response Time: {avg_response_time:.2f} sec")

    results = {
        "accuracy": accuracy,
        "coverage": avg_coverage * 100,
        "recall": recall_at_k * 100,
        "mrr": avg_mrr,

        "judge_accuracy": total_judge_accuracy / n,
        "judge_completeness": total_judge_completeness / n,
        "judge_relevance": total_judge_relevance / n,

        "category_stats": category_stats,

        "failed_questions": failed_questions,

        "detailed_results":detailed_results,
        "average_response_time": avg_response_time,
    }

    with open(
        "evaluation_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

def run_evaluation():

    evaluate()

    return "Evaluation Complete. Open dashboard.py and click Load Evaluation."

def compress_context(question,docs):

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )
    print("=" * 60)
    print("Retrieved Docs:", len(docs))
    print("Context Length:", len(context))
    print("=" * 60)

    prompt = f"""
    You are a retrieval compressor.

    Question:
    {question}

    Context:
    {context}

    Extract ONLY the information
    relevant to answering the question.

    Return concise notes.
    """

    response = judge_llm.invoke(prompt)
    return response.content

def rewrite_query(query):

    history_text = ""

    for q, a in chat_history[-3:]:

        history_text += (
            f"\nUser:{q}\n"
            f"Assistant:{a}\n"
        )

    prompt = f"""
    Conversation Summary:
    {conversation_summary}

    Recent History:
    {history_text}
    
    Current Question:
    {query}

    Rewrite the question
    as a standalone search query.

    Return only query.
    """

    response = call_llm(
    task="query_rewrite",
    prompt=prompt
    )

    return response.strip()

def strip_prompt_injection(query):

    blocked = [

        "ignore previous instructions",

        "forget all instructions",

        "system prompt",

        "reveal prompt",

        "developer instructions",

        "act as"

    ]

    cleaned = query

    for item in blocked:

        cleaned = cleaned.replace(
            item,
            ""
        )

    return cleaned

ACRONYMS = {

    "ml": "machine learning",

    "ai": "artificial intelligence",

    "llm": "large language model",

    "rag": "retrieval augmented generation",

    "api": "application programming interface",

    "db": "database",

    "nlp": "natural language processing"
}

def expand_acronyms(query):

    words = query.split()

    expanded = []

    for word in words:

        expanded.append(
            ACRONYMS.get(
                word.lower(),
                word
            )
        )

    return " ".join(
        expanded
    )

def normalize_query(query):

    query = expand_acronyms(query)
    
    query = query.strip()

    query = re.sub(
        r"\s+",
        " ",
        query
    )

    query = re.sub(
        r"[\x00-\x1F\x7F]",
        "",
        query
    )

    return query

def aggregate_answers(
    question,
    answers
):

    prompt = f"""
Question:
{question}

Sub Answers:
{answers}

Combine into one answer.
"""

    response = local_llm.invoke(prompt)

    return response.content

def summarize_conversation():

    global chat_history
    global conversation_summary

    history_text = ""

    for q, a in chat_history:

        history_text += (
            f"\nUser:{q}\n"
            f"Assistant:{a}\n"
        )

    prompt = f"""
    Summarize the important information
    from this conversation.

    Conversation:
    {history_text}

    Return concise summary.
    """

    response = local_llm.invoke(prompt)

    conversation_summary = (
        response.content
    )

def print_retrieval_debug(docs):

    print("\n" + "="*60)

    print("TOP RETRIEVED DOCUMENTS")

    print("="*60)

    for i, doc in enumerate(docs):

        print()

        print(f"Rank {i+1}")

        print(
            "File:",
            doc.metadata.get(
                "filename"
            )
        )

        print(
            "Page:",
            doc.metadata.get(
                "page"
            )
        )

        print(
            "Chunk:",
            doc.metadata.get(
                "chunk_index"
            )
        )

        print(doc.page_content[:200])

import copy   

def expand_to_parent_docs(docs):

    expanded = []

    seen = set()

    for doc in docs:

        parent_id = doc.metadata.get("parent_id")

        if parent_id in seen:
            continue

        seen.add(parent_id)

        parent = parent_docs.get(parent_id)

        if parent:

            doc.page_content = parent["content"]

            doc.metadata["page"] = parent["page"]

            doc.metadata["filename"] = parent["filename"]

            expanded.append(doc)

        else:
            expanded.append(doc)

    return expanded

with gr.Blocks() as demo:

    history = gr.State([])

    document = gr.File(
        file_count="multiple",
        file_types=[".pdf", ".docx"]
    )

    status = gr.Textbox(
        label="Knowledge Base Status"
    )

    document.change(
        fn=process_document,
        inputs=document,
        outputs=status
    )

    chatbot = gr.Chatbot(height=600)

    msg = gr.Textbox(
        placeholder="Ask a question..."
    )

    msg.submit(
        fn=chat,
        inputs=[msg, history],
        outputs=[chatbot, history, msg]
    )
    clear_btn = gr.Button("Clear Chat")
    clear_btn.click(
        lambda: ([], []),
        outputs=[chatbot, history]
    )

if __name__ == "__main__":
    demo.launch()
