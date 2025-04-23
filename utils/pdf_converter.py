from pypdf import PdfReader
import re

def convert_pdf_to_markdown(pdf_path):
    """
    Converte um arquivo PDF para formato markdown usando pypdf.
    
    Args:
        pdf_path (str): Caminho para o arquivo PDF.
        
    Returns:
        str: Conteúdo do PDF convertido para markdown.
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        
        # Extrair texto de cada página
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"## Página {i+1}\n\n{page_text}\n\n"
        
        # Processamento para identificar perguntas (linhas que terminam com '?')
        text = re.sub(r'([^\n]+\?)\n', r'### \1\n', text)
        
        return text
    except Exception as e:
        raise Exception(f"Erro ao converter PDF para markdown: {e}")