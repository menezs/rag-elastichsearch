from pathlib import Path
from typing import Literal
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_elasticsearch import ElasticsearchStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from elasticsearch.helpers import BulkIndexError

from elasticsearch import Elasticsearch

ES_URL = "http://localhost:9200"
ES_USER = "elastic"
ES_PASSWORD = "senha-elasticsearch"

class RagConnect:
    def __init__(self):
        self.es_url = ES_URL
        self.es_user = ES_USER
        self.es_password = ES_PASSWORD
        self.es_client = self.__get_es_client()
        self.embeddings = self.__get_embeddings()

    @staticmethod
    def __get_embeddings():
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    

    def __get_es_client(self):
        return Elasticsearch(
            self.es_url,
            basic_auth=(self.es_user, self.es_password)
        )

    @staticmethod
    def split_documents(documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "|", "  "]
        )
        return splitter.split_documents(documents)
    
    def load_documents_from_paths(self, paths: list[str]) -> list[Document]:
        all_docs = []

        for path_str in paths:
            path = Path(path_str)

            if path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(path), extract_images=False, headers=True)

            elif path.suffix.lower() == ".txt":
                loader = TextLoader(str(path), autodetect_encoding=True)

            else:
                continue

            try:
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = str(path)
                    doc.metadata.pop("creationdate", None)

                all_docs.extend(docs)

            except Exception as e:
                print(f"Erro ao carregar {path.name}: {e}")

        return all_docs



    def save_to_elasticsearch(self, docs: list[Document], index_name="rag_docs"):

        return ElasticsearchStore.from_documents(
            documents=docs,
            embedding=self.embeddings,
            index_name=index_name,
            es_connection=self.es_client,
            es_url=self.es_url,
            es_user=self.es_user,
            es_password=self.es_password
        )

    def append_to_elasticsearch_index(self, docs: list[Document], index_name="rag_docs"):
        try:
            if self.es_client.indices.exists(index=index_name):
                print(f"✅ Índice '{index_name}' já existe. Adicionando documentos...")
            
                store = ElasticsearchStore(
                    index_name=index_name,
                    embedding=self.embeddings,
                    es_connection=self.es_client,
                    es_url=self.es_url,
                    es_user=self.es_user,
                    es_password=self.es_password
                )
        
                store.add_documents(docs)
        
            else:
                print(f"🆕 Índice '{index_name}' não existe. Criando novo índice com os documentos...")

                store = ElasticsearchStore.from_documents(
                    documents=docs,
                    embedding=self.embeddings,
                    index_name=index_name,
                    es_connection=self.es_client,
                    es_url=self.es_url,
                    es_user=self.es_user,
                    es_password=self.es_password
                )

            return store

        except BulkIndexError as e:
            print("❌ Alguns documentos falharam ao indexar:")
        
            for i, err in enumerate(e.errors):
                print(f"[{i+1}] {err}")
            
            raise e

    def get_vectorstore(self, index_name="rag_docs"):
        return ElasticsearchStore(
            index_name=index_name,
            embedding=self.embeddings,
            es_connection=self.es_client
        )

    def get_llm(self, source: Literal["lmstudio", "ollama"] = "lmstudio"):
        if source == "lmstudio":
            return ChatOpenAI(
                base_url="http://localhost:1234/v1",
                api_key="lm-studio",
                model="gemma-2-2b-it"
            )
        elif source == "ollama":
            return ChatOpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # valor qualquer, não é necessário
                model="gemma:2b"
            )
        else:
            raise ValueError("source must be 'lmstudio' or 'ollama'")

    def load_rag_pipeline(self, llm_source="lmstudio", k=3):
        vectorstore = self.get_vectorstore()
        llm = self.get_llm(llm_source)
        prompt_template = """
            Você é um assistente que sempre resepnde em português-BR e fornecer
            informações baseadas no contexto fornecido. Responda APENAS com o conteúdo
            exato encontrado no contexto apresentado:
            Contexto: {context}

            Pergunta: {question}

            Resposta:
        """

        if llm_source == 'lmstudio':
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template=prompt_template
            )

        elif llm_source == 'ollama':
            prompt_template = ChatPromptTemplate.from_template(
                template=prompt_template
            )


        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt_template}
        )

        return qa_chain
    
    def get_prompt(self, llm_source="lmstudio"):
        template = """
            Você é um assistente que sempre responde em português-BR e fornece
            informações baseadas no contexto fornecido. Responda APENAS com o conteúdo
            exato encontrado no contexto apresentado:

            Contexto: {context}

            Pergunta: {question}

            Resposta:
        """
        if llm_source == "lmstudio":
            return PromptTemplate(input_variables=["context", "question"], template=template)
        elif llm_source == "ollama":
            return ChatPromptTemplate.from_template(template=template)

