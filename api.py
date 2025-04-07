from fastapi import FastAPI, UploadFile, File, Form, APIRouter
from langchain_core.documents import Document
from fastapi.responses import JSONResponse
import os
from rag_connect import RagConnect
from typing import Literal
import shutil

rag_client = RagConnect()

app = FastAPI(
    title="RAG API",
    description="API for RAG system with file upload, embedding generation and query with local LLMs.",
    version="1.0.0"
)

router = APIRouter(tags=["RAG"])

DOCS_PATH = "docs"
os.makedirs(DOCS_PATH, exist_ok=True)

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(DOCS_PATH, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    docs = rag_client.load_documents_from_paths([file_path])
    texts = rag_client.split_documents(docs)
    rag_client.append_to_elasticsearch_index(texts)

    return {"message": f"Arquivo {file.filename} enviado com sucesso!"}

@router.post("/embed/")
async def embed_documents():
    files = [os.path.join(DOCS_PATH, f) for f in os.listdir(DOCS_PATH) if f.endswith((".pdf", ".txt"))]
    
    if not files:
        return JSONResponse(status_code=400, content={"error": "Nenhum arquivo PDF ou TXT encontrado na pasta 'docs'."})

    docs = rag_client.load_documents_from_paths(files)
    texts = rag_client.split_documents(docs)
    rag_client.save_to_elasticsearch(texts)

    return {"message": f"{len(docs)} documentos processados e enviados ao Elasticsearch."}

# @router.post("/ask/")
# async def ask_question(
#     question: str = Form(...),
#     k: int = Form(3),
#     llm_source: Literal["ollama", "lmstudio"] = Form("lmstudio")
# ):
#     qa_chain = rag_client.load_rag_pipeline(llm_source=llm_source, k=k)
#     response = qa_chain.invoke(question)
#     return response


# @router.post("/ask/")
# async def ask_question(
#     question: str = Form(...),
#     k: int = Form(3),
#     llm_source: Literal["ollama", "lmstudio"] = Form("lmstudio")
# ):
#     retriever = rag_client.get_vectorstore().as_retriever(search_kwargs={"k": k})
#     llm = rag_client.get_llm(llm_source)
#     prompt_template = rag_client.get_prompt(llm_source)

#     # Recupera os documentos relevantes
#     context_docs = await retriever.aget_relevant_documents(question)

#     # Extrai só o conteúdo dos docs (ou se quiser metadados também)
#     context_texts = [doc.page_content for doc in context_docs]

#     # Monta o contexto manualmente
#     context_combined = "\n\n".join(context_texts)

#     # Substitui manualmente os campos no prompt (simula o RAG)
#     prompt = prompt_template.format(context=context_combined, question=question)

#     # Chama o modelo com o prompt direto
#     response = llm.invoke(prompt)

#     return {
#         "question": question,
#         "answer": response.content if hasattr(response, "content") else str(response),
#         "context": context_texts  # você também pode retornar metadata aqui se quiser
#     }

@router.post("/ask/")
async def ask_question(
    question: str = Form(...),
    k: int = Form(3),
    llm_source: Literal["ollama", "lmstudio"] = Form("lmstudio")
):
    retriever = rag_client.get_vectorstore().as_retriever(search_kwargs={"k": k})
    llm = rag_client.get_llm(llm_source)
    prompt_template = rag_client.get_prompt(llm_source)

    docs: list[Document] = await retriever.ainvoke(question)

    context_texts = [doc.page_content for doc in docs]
    context_combined = "\n\n".join(context_texts)

    prompt = prompt_template.format(context=context_combined, question=question)

    response = llm.invoke(prompt)

    context_response = [
        {
            "rank": idx + 1,
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        for idx, doc in enumerate(docs)
    ]

    return {
        "question": question,
        "answer": response.content if hasattr(response, "content") else str(response),
        "context_documents": context_response
    }



app.include_router(router)
