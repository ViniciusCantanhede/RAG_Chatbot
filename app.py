import os
import streamlit as st
import tempfile
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Importar funções dos módulos utils
from utils.pdf_converter import convert_pdf_to_markdown
from utils.data_augmentation import translate_augmentation
from utils.vector_store import create_vector_store, save_vector_store, load_vector_store

# Configurar página
st.set_page_config(page_title="Chatbot RAG com Data Augmentation", layout="wide")

# Título da aplicação
st.title("Chatbot RAG com Data Augmentation")

# Sidebar para configurações
st.sidebar.header("Configurações")
api_key = st.sidebar.text_input("Insira sua chave de API OpenAI", value=os.getenv("OPENAI_API_KEY", ""), type="password")
os.environ["OPENAI_API_KEY"] = api_key

# Inicializar variáveis da sessão
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Área para upload do PDF
st.header("Configuração do Chatbot")
pdf_file = st.file_uploader("Carregar PDF sobre a empresa", type=["pdf"])

if pdf_file and api_key:
    # Salvar arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(pdf_file.read())
        pdf_path = temp_file.name
    
    # Processar documento
    with st.spinner("Processando documento..."):
        try:
            # Converter PDF para markdown
            markdown_content = convert_pdf_to_markdown(pdf_path)
            
            if markdown_content:
                st.success("PDF convertido para markdown com sucesso!")
                
                # Realizar data augmentation
                with st.spinner("Realizando data augmentation..."):
                    augmented_texts = translate_augmentation(markdown_content)
                    st.success(f"Data augmentation concluído! Criados {len(augmented_texts)} textos.")
                
                # Criar vector store
                with st.spinner("Criando vector store..."):
                    vectorstore = create_vector_store(
                        augmented_texts,
                        persist_directory="chroma_db"
                    )
                    st.session_state.vectorstore = vectorstore
                    st.success("Vector store criado com sucesso!")
                    
                    # Opção para salvar o vector store
                    if st.button("Salvar Vector Store"):
                        save_vector_store(vectorstore)
                        st.success("Vector store salvo com sucesso!")
            
            # Remover arquivo temporário
            os.unlink(pdf_path)
            
        except Exception as e:
            st.error(f"Erro ao processar o documento: {e}")

# Opção para carregar vector store salvo
st.sidebar.header("Carregar Vector Store Salvo")
if st.sidebar.button("Carregar Vector Store") and os.path.exists("chroma_db"):
    try:
        st.session_state.vectorstore = load_vector_store("chroma_db")
        st.sidebar.success("Vector store carregado com sucesso!")
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar vector store: {e}")

# Interface do chatbot
st.header("Chatbot")

# Não permitir interação até que o vector store esteja pronto
if not api_key:
    st.warning("Por favor, insira sua chave de API OpenAI para continuar.")
elif st.session_state.vectorstore is None:
    st.warning("Por favor, carregue um PDF para configurar o chatbot ou carregue um vector store salvo.")
else:
    # Exibir histórico de chat
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])
    
    # Input do usuário
    user_input = st.chat_input("Digite sua mensagem aqui...")
    
    if user_input:
        # Adicionar mensagem do usuário ao histórico
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        
        # Criar modelo e chain
        llm = ChatOpenAI(model="o3-mini", temperature=0.2)
        retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
        )
        
        # Obter resposta
        with st.spinner("Gerando resposta..."):
            # Preparar histórico para o formato correto
            chat_history = [(q["content"], a["content"]) 
                          for q, a in zip(
                              st.session_state.chat_history[::2], 
                              st.session_state.chat_history[1::2]
                          ) if a["role"] == "assistant"]
            
            response = retrieval_chain.invoke({
                "question": user_input,
                "chat_history": chat_history
            })
            
            answer = response["answer"]
        
        # Adicionar resposta ao histórico
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)