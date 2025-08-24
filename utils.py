import os
import shutil
import time
import concurrent.futures
from typing import List
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader, PyPDFLoader
from config import VECTOR_DB_DIR

def allowed_file(filename):
    return filename.lower().endswith(('.pdf', '.txt'))

def ensure_directory_permissions(directory):
    """
    Ensure directory has proper write permissions
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory, mode=0o755, exist_ok=True)
        else:
            # Ensure write permissions
            os.chmod(directory, 0o755)
        return True
    except Exception as e:
        print(f"Error setting permissions for {directory}: {e}")
        return False

def cleanup_previous_session(uploaded_dir="uploaded", vector_db_dir=None):
    """
    Clean up previous session files and vector database
    """
    if vector_db_dir is None:
        vector_db_dir = VECTOR_DB_DIR
    
    cleanup_success = True
    
    # Clean up uploaded files directory
    if os.path.exists(uploaded_dir):
        try:
            shutil.rmtree(uploaded_dir, ignore_errors=True)
            print(f"✅ Cleaned up {uploaded_dir} directory")
        except Exception as e:
            print(f"❌ Error cleaning up {uploaded_dir}: {e}")
            cleanup_success = False
    
    # Clean up vector database directory with enhanced retry mechanism
    if os.path.exists(vector_db_dir):
        max_retries = 5  # Increased retries
        for attempt in range(max_retries):
            try:
                # Try different cleanup methods
                if attempt < 3:
                    shutil.rmtree(vector_db_dir, ignore_errors=True)
                else:
                    # More aggressive cleanup for persistent files
                    for root, dirs, files in os.walk(vector_db_dir, topdown=False):
                        for file in files:
                            try:
                                os.chmod(os.path.join(root, file), 0o777)
                                os.remove(os.path.join(root, file))
                            except:
                                pass
                        for dir in dirs:
                            try:
                                os.rmdir(os.path.join(root, dir))
                            except:
                                pass
                    try:
                        os.rmdir(vector_db_dir)
                    except:
                        pass
                
                # Check if cleanup was successful
                if not os.path.exists(vector_db_dir):
                    print(f"✅ Cleaned up {vector_db_dir} directory (attempt {attempt + 1})")
                    break
                    
            except Exception as e:
                print(f"❌ Error cleaning up {vector_db_dir} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                else:
                    print(f"⚠️ Failed to clean up {vector_db_dir} after {max_retries} attempts")
                    cleanup_success = False
    
    # Clean up any cache files
    cache_dirs = ["cache", "__pycache__", ".chroma"]
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir, ignore_errors=True)
                print(f"✅ Cleaned up {cache_dir} directory")
            except Exception as e:
                print(f"❌ Error cleaning up {cache_dir}: {e}")
    
    # Clean up conversation memory file
    memory_file = "conversation_memory.json"
    if os.path.exists(memory_file):
        try:
            os.remove(memory_file)
            print(f"✅ Cleaned up {memory_file}")
        except Exception as e:
            print(f"❌ Error cleaning up {memory_file}: {e}")
    
    # Ensure directories don't exist and create fresh ones with proper permissions
    ensure_directory_permissions(uploaded_dir)
    ensure_directory_permissions(vector_db_dir)
    
    return cleanup_success

def reset_chromadb(vector_db_dir=None):
    """
    Specifically reset ChromaDB when it has database errors
    """
    if vector_db_dir is None:
        vector_db_dir = VECTOR_DB_DIR
    
    print("🔄 Resetting ChromaDB database...")
    
    try:
        # Force close any existing connections
        import gc
        gc.collect()
        
        # Remove the entire ChromaDB directory
        if os.path.exists(vector_db_dir):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(vector_db_dir, ignore_errors=True)
                    time.sleep(1)
                    
                    if not os.path.exists(vector_db_dir):
                        print(f"✅ ChromaDB directory removed (attempt {attempt + 1})")
                        break
                        
                except Exception as e:
                    print(f"❌ Attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
        
        # Recreate the directory
        os.makedirs(vector_db_dir, exist_ok=True)
        print(f"✅ ChromaDB directory recreated: {vector_db_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error resetting ChromaDB: {e}")
        return False

def save_uploaded_files(uploaded_files, temp_dir="uploaded"):
    # Clean up previous session before saving new files
    cleanup_previous_session(temp_dir)
    
    os.makedirs(temp_dir, exist_ok=True)
    file_paths = []
    for file in uploaded_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        file_paths.append(file_path)
    return file_paths

def load_single_document(file_path):
    """Load a single document with error handling"""
    try:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            return loader.load()
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
            return loader.load()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def load_documents(file_paths: List[str], use_parallel=True):
    """
    Load documents with optional parallel processing for better performance
    """
    if not use_parallel or len(file_paths) == 1:
        # Sequential loading for single file or when parallel is disabled
        docs = []
        for path in file_paths:
            docs.extend(load_single_document(path))
        return docs
    
    # Parallel loading for multiple files
    docs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(file_paths))) as executor:
        future_to_path = {executor.submit(load_single_document, path): path for path in file_paths}
        
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                doc_pages = future.result()
                docs.extend(doc_pages)
                print(f"Successfully loaded: {os.path.basename(path)} ({len(doc_pages)} pages)")
            except Exception as e:
                print(f"Error loading {path}: {e}")
    
    print(f"Total documents loaded: {len(docs)} pages from {len(file_paths)} files")
    return docs

def load_documents_from_directory(directory_path: str):
    """
    Load all supported documents from a directory
    """
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist")
        return []
    
    file_paths = []
    for filename in os.listdir(directory_path):
        if allowed_file(filename):
            file_paths.append(os.path.join(directory_path, filename))
    
    if not file_paths:
        print(f"No supported files found in {directory_path}")
        return []
    
    return load_documents(file_paths)