def delete_htmls(folder):
    """Delete htmls in a folder regressiveley

    Args:
        folder (_type_): _description_
    """    
    import os
    for dirpath, dirnames, filenames in os.walk(folder):
        for file in filenames:
            if file.endswith(".html"):
                os.remove(os.path.join(dirpath, file))