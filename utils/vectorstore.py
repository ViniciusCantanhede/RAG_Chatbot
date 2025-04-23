from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import re
import os
import shutil

def create_vector_store(texts, chunk_size=1000, chunk_overlap=200, persist_directory="chroma_db"):
    """
    Cria um vector store Chroma a partir de textos.
    
    Args:
        texts (list): Lista de textos para processar.
        chunk_size (int, optional): Tamanho de cada chunk. Default é 1000.
        chunk_overlap (int, optional): Sobreposição entre chunks. Default é 200.
        persist_directory (str): Diretório para persistir o vector store.
        
    Returns:
        Chroma: Vector store contendo os embeddings dos textos.
    """
    # Dividir em chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    all_chunks = []
    for text in texts:
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)
    
    # Remover diretório anterior se existir
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    
    # Criar embeddings e vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(
        texts=all_chunks, 
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Persistir o vector store
    vectorstore.persist()
    
    return vectorstore

def save_vector_store(vectorstore, path="chroma_db"):
    """
    Salva o vector store Chroma no disco.
    
    Args:
        vectorstore (Chroma): Vector store a ser salvo.
        path (str, optional): Caminho para salvar. Default é "chroma_db".
    """
    # O Chroma já salva automaticamente quando criado com persist_directory
    vectorstore.persist()
    print(f"Vector store salvo em {path}")

def load_vector_store(path="chroma_db"):
    """
    Carrega um vector store Chroma do disco.
    
    Args:
        path (str, optional): Caminho do vector store. Default é "chroma_db".
        
    Returns:
        Chroma: Vector store carregado.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=path, embedding_function=embeddings)
    return vectorstore