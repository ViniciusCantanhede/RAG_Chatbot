from deep_translator import GoogleTranslator

def translate_augmentation(markdown_text):
    """
    Realiza data augmentation via tradução para textos grandes.
    
    Args:
        markdown_text (str): Texto em markdown para realizar augmentation.
        
    Returns:
        list: Lista contendo o texto original e versões traduzidas/retraduzidas.
    """
    augmented_texts = [markdown_text]  # Texto original
    
    # Dividir o texto em partes menores para respeitar os limites da API de tradução
    chunk_size = 4000  # Ajustado para ficar abaixo do limite de 5000 caracteres
    chunks = [markdown_text[i:i+chunk_size] for i in range(0, len(markdown_text), chunk_size)]
    
    # Tradução para espanhol e retradução
    try:
        spanish_chunks = []
        for chunk in chunks:
            translated = GoogleTranslator(source='pt', target='es').translate(chunk)
            spanish_chunks.append(translated)
        
        spanish_text = ''.join(spanish_chunks)
        
        # Retraduzir para português
        pt_chunks = []
        es_chunks = [spanish_text[i:i+chunk_size] for i in range(0, len(spanish_text), chunk_size)]
        for chunk in es_chunks:
            translated = GoogleTranslator(source='es', target='pt').translate(chunk)
            pt_chunks.append(translated)
        
        spanish_to_pt = ''.join(pt_chunks)
        augmented_texts.append(spanish_to_pt)
    except Exception as e:
        print(f"Erro na tradução ES: {e}")
    
    # Processo similar para italiano
    try:
        italian_chunks = []
        for chunk in chunks:
            translated = GoogleTranslator(source='pt', target='it').translate(chunk)
            italian_chunks.append(translated)
        
        italian_text = ''.join(italian_chunks)
        
        # Retraduzir para português
        pt_chunks = []
        it_chunks = [italian_text[i:i+chunk_size] for i in range(0, len(italian_text), chunk_size)]
        for chunk in it_chunks:
            translated = GoogleTranslator(source='it', target='pt').translate(chunk)
            pt_chunks.append(translated)
        
        italian_to_pt = ''.join(pt_chunks)
        augmented_texts.append(italian_to_pt)
    except Exception as e:
        print(f"Erro na tradução IT: {e}")
    
    return augmented_texts