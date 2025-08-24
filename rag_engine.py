import os
import shutil
import time
import tempfile
import hashlib
import pickle
import concurrent.futures
from typing import List, Dict, Any, Optional
import threading

os.environ["USE_TF"] = "0"   # ensure TF is disabled

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from groq import Groq
from config import GROQ_API_KEY, EMBEDDING_MODEL, LLM_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_DB_DIR, SSL_VERIFY
import httpx

def create_groq_client():
    """Create Groq client with SSL configuration for corporate networks"""
    try:
        if not SSL_VERIFY:
            # Create HTTP client that bypasses SSL verification
            http_client = httpx.Client(verify=False)
            return Groq(api_key=GROQ_API_KEY, http_client=http_client)
        else:
            return Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"Error creating Groq client: {e}")
        # Fallback: try without custom HTTP client
        return Groq(api_key=GROQ_API_KEY)

class OptimizedChromaDB:
    """
    Optimized ChromaDB wrapper with caching and parallel processing
    """
    def __init__(self, persist_dir=None, collection_name="studymate_collection"):
        self.persist_dir = persist_dir or VECTOR_DB_DIR
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.cache_dir = os.path.join(self.persist_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._lock = threading.Lock()
    
    def _get_document_hash(self, documents):
        """Generate hash for document set for caching"""
        content = "".join([doc.page_content for doc in documents])
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_cached_chunks(self, doc_hash):
        """Load cached processed chunks"""
        cache_file = os.path.join(self.cache_dir, f"{doc_hash}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
        return None
    
    def _save_cached_chunks(self, doc_hash, chunks):
        """Save processed chunks to cache"""
        cache_file = os.path.join(self.cache_dir, f"{doc_hash}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(chunks, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def _parallel_chunk_processing(self, documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        """Process documents in parallel for faster chunking"""
        def process_doc(doc):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            return splitter.split_documents([doc])
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_doc = {executor.submit(process_doc, doc): doc for doc in documents}
            all_chunks = []
            
            for future in concurrent.futures.as_completed(future_to_doc):
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Error processing document: {e}")
        
        return all_chunks
    
    def build_database(self, documents, embedding_model):
        """Build ChromaDB with optimizations"""
        with self._lock:
            try:
                # Check cache first
                doc_hash = self._get_document_hash(documents)
                cached_chunks = self._load_cached_chunks(doc_hash)
                
                if cached_chunks:
                    print(f"Using cached chunks ({len(cached_chunks)} chunks)")
                    docs_split = cached_chunks
                else:
                    print("Processing documents with parallel chunking...")
                    docs_split = self._parallel_chunk_processing(documents)
                    self._save_cached_chunks(doc_hash, docs_split)
                    print(f"Processed and cached {len(docs_split)} chunks")
                
                # Initialize ChromaDB client with error handling
                os.makedirs(self.persist_dir, exist_ok=True)
                
                try:
                    self.client = chromadb.PersistentClient(path=self.persist_dir)
                except Exception as e:
                    print(f"Error initializing ChromaDB: {e}")
                    print("Attempting to reset ChromaDB directory...")
                    
                    # Clean up corrupted database
                    import shutil
                    if os.path.exists(self.persist_dir):
                        shutil.rmtree(self.persist_dir, ignore_errors=True)
                        time.sleep(1)
                    
                    # Recreate directory and try again
                    os.makedirs(self.persist_dir, exist_ok=True)
                    self.client = chromadb.PersistentClient(path=self.persist_dir)
                    print("✅ ChromaDB reset and reinitialized successfully")
                
                # Delete existing collection if it exists
                try:
                    self.client.delete_collection(name=self.collection_name)
                    print(f"Deleted existing collection: {self.collection_name}")
                except Exception as e:
                    print(f"No existing collection to delete: {e}")
                
                # Create vector store with batching for better performance
                batch_size = 100
                vector_db = None
                
                for i in range(0, len(docs_split), batch_size):
                    batch = docs_split[i:i+batch_size]
                    if vector_db is None:
                        vector_db = Chroma.from_documents(
                            documents=batch,
                            embedding=embedding_model,
                            client=self.client,
                            collection_name=self.collection_name
                        )
                    else:
                        vector_db.add_documents(batch)
                    print(f"Processed batch {i//batch_size + 1}/{(len(docs_split) + batch_size - 1)//batch_size}")
                
                self.collection = vector_db
                print(f"Successfully built ChromaDB with {len(docs_split)} chunks")
                return vector_db
                
            except Exception as e:
                error_msg = str(e)
                print(f"Error building ChromaDB: {error_msg}")
                
                # Check for specific database errors
                if "no such table: tenants" in error_msg or "database" in error_msg.lower():
                    print("🔧 Detected ChromaDB database corruption, attempting auto-reset...")
                    
                    # Try to reset and rebuild
                    try:
                        import shutil
                        if os.path.exists(self.persist_dir):
                            shutil.rmtree(self.persist_dir, ignore_errors=True)
                            time.sleep(1)
                        
                        os.makedirs(self.persist_dir, exist_ok=True)
                        self.client = chromadb.PersistentClient(path=self.persist_dir)
                        
                        print("✅ ChromaDB auto-reset successful, retrying...")
                        
                        # Retry building the vector store
                        vector_db = None
                        for i in range(0, len(docs_split), batch_size):
                            batch = docs_split[i:i+batch_size]
                            if vector_db is None:
                                vector_db = Chroma.from_documents(
                                    documents=batch,
                                    embedding=embedding_model,
                                    client=self.client,
                                    collection_name=self.collection_name
                                )
                            else:
                                vector_db.add_documents(batch)
                            print(f"Processed batch {i//batch_size + 1}/{(len(docs_split) + batch_size - 1)//batch_size}")
                        
                        self.collection = vector_db
                        print(f"✅ Successfully rebuilt ChromaDB after reset with {len(docs_split)} chunks")
                        return vector_db
                        
                    except Exception as retry_e:
                        print(f"❌ Auto-reset failed: {retry_e}")
                        raise Exception(f"Failed to build ChromaDB even after reset: {retry_e}")
                else:
                    raise Exception(f"Failed to build ChromaDB: {error_msg}")
    
    def close(self):
        """Close ChromaDB connection"""
        try:
            if self.client:
                self.client = None
            print("ChromaDB connection closed")
        except Exception as e:
            print(f"Error closing ChromaDB: {e}")

def build_vector_db(documents, persist_dir=None, embedding_model=None, collection_name=None):
    """
    Build optimized ChromaDB vector database with caching and parallel processing
    """
    if embedding_model is None:
        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL, 
                model_kwargs={
                    "device": "cpu",
                    "trust_remote_code": True
                },
                encode_kwargs={"normalize_embeddings": True}
            )
        except Exception as e:
            print(f"Error loading primary embedding model: {e}")
            # Fallback to reliable model
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
    
    if persist_dir is None:
        persist_dir = VECTOR_DB_DIR
    
    if collection_name is None:
        collection_name = "studymate_collection"
    
    chroma_db = OptimizedChromaDB(persist_dir, collection_name)
    vector_db = chroma_db.build_database(documents, embedding_model)
    
    # Attach the ChromaDB instance for proper cleanup
    vector_db._chroma_instance = chroma_db
    return vector_db

def get_relevant_context(user_query, vector_db, top_k=5):
    """Enhanced context retrieval with better ranking"""
    try:
        retriever = vector_db.as_retriever(
            search_type='similarity_score_threshold',
            search_kwargs={'k': top_k, 'score_threshold': 0.1}
        )
        relevant_docs = retriever.invoke(user_query)
        
        # If no docs meet threshold, fallback to regular similarity
        if not relevant_docs:
            retriever = vector_db.as_retriever(search_type='similarity', search_kwargs={'k': top_k})
            relevant_docs = retriever.invoke(user_query)
        
        return ". ".join([doc.page_content for doc in relevant_docs])
    except Exception as e:
        print(f"Error in context retrieval: {e}")
        # Fallback to basic retrieval
        try:
            retriever = vector_db.as_retriever(search_kwargs={'k': top_k})
            relevant_docs = retriever.invoke(user_query)
            return ". ".join([doc.page_content for doc in relevant_docs])
        except Exception as e2:
            print(f"Fallback retrieval also failed: {e2}")
            return "Unable to retrieve relevant context from documents."

def close_vector_db(vector_db):
    """
    Properly close ChromaDB connections
    """
    try:
        if hasattr(vector_db, '_chroma_instance'):
            vector_db._chroma_instance.close()
        elif hasattr(vector_db, '_client') and vector_db._client:
            vector_db._client = None
        print("Vector database closed successfully")
    except Exception as e:
        print(f"Error closing vector database: {e}")

def generate_qa_answer(query, context):
    """Generate precise Q&A answers based on document content"""
    system_message = """You are a precise AI assistant for Q&A mode.
    
    INSTRUCTIONS:
    - Answer ONLY based on the provided context
    - Be direct and factual
    - If the answer is not in the context, say "I don't have enough information to answer this question based on the provided documents."
    - Keep answers concise but complete
    - Use bullet points or numbered lists when appropriate"""
    
    user_message = f"###Context from Documents\n{context}\n\n###Question\n{query}\n\nProvide a precise answer based only on the context above."
    
    try:
        client = create_groq_client()
        print(f"Using LLM model: {LLM_MODEL}")
        print(f"Generating response for query: {query[:100]}...")
        
        # Retry logic for connection errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.1,
                    max_tokens=1024
                )
                
                result = response.choices[0].message.content.strip()
                print(f"Generated response length: {len(result)} characters")
                return result
                
            except Exception as retry_e:
                print(f"Attempt {attempt + 1} failed: {retry_e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    raise retry_e
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"Error generating LLM response ({error_type}): {error_msg}")
        print(f"API Key (first 10 chars): {GROQ_API_KEY[:10]}...")
        print(f"Model: {LLM_MODEL}")
        
        # Provide specific error messages based on error type
        if "connection" in error_msg.lower() or "network" in error_msg.lower():
            return """🌐 **Connection Issue Detected**
            
I'm unable to connect to the AI service. This could be due to:
- Internet connectivity issues
- Firewall or proxy blocking the connection
- Groq API service temporary outage

**Please try:**
1. Check your internet connection
2. Wait a few minutes and try again
3. Use the Reset button to start fresh

*Error: Connection to AI service failed*"""
        elif "api" in error_msg.lower() or "auth" in error_msg.lower():
            return """🔑 **API Authentication Issue**
            
There's a problem with the AI service authentication. Please check:
- API key validity in config.py
- Account permissions and credits

*Error: API authentication failed*"""
        elif "rate" in error_msg.lower() or "limit" in error_msg.lower():
            return """⏰ **Rate Limit Reached**
            
The AI service is currently rate-limited. Please:
- Wait a few minutes before trying again
- Consider upgrading your API plan for higher limits

*Error: API rate limit exceeded*"""
        else:
            return f"""❌ **Service Error**
            
I encountered an error while generating a response:
{error_msg}

**Please try:**
- Using the Reset button to clear the session
- Checking your internet connection
- Trying again in a few minutes

*If the problem persists, there may be a temporary service issue.*"""

def generate_quiz_questions(context, num_questions=5):
    """Generate quiz questions based on document content"""
    system_message = """You are a quiz generator. Create educational quiz questions based on the provided context.
    
    INSTRUCTIONS:
    - Generate exactly the requested number of questions
    - Questions should test understanding, not just memorization
    - Include a mix of difficulty levels
    - Format each question with its answer
    - Use this EXACT format for each question:
    
    Q1: [Question text]
    A1: [Answer text]
    
    Q2: [Question text]
    A2: [Answer text]
    
    And so on..."""
    
    user_message = f"###Context from Documents\n{context}\n\n###Task\nGenerate {num_questions} quiz questions based on the context above. Make them educational and varied in difficulty."
    
    try:
        client = create_groq_client()
        
        # Retry logic for connection errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.3,
                    max_tokens=1024
                )
                return response.choices[0].message.content.strip()
                
            except Exception as retry_e:
                print(f"Quiz generation attempt {attempt + 1} failed: {retry_e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    raise retry_e
                    
    except Exception as e:
        error_msg = str(e)
        print(f"Error generating quiz questions: {error_msg}")
        
        if "connection" in error_msg.lower() or "network" in error_msg.lower():
            return "I'm experiencing connection issues. Please check your internet connection and try generating the quiz again."
        elif "rate" in error_msg.lower() or "limit" in error_msg.lower():
            return "The AI service is currently busy. Please wait a moment and try generating the quiz again."
        else:
            return f"I encountered an error generating quiz questions: {error_msg}. Please try again."

def generate_chat_response(query, context, chat_history=[]):
    """Generate conversational responses for chat mode"""
    system_message = """You are a friendly and knowledgeable AI tutor in chat mode.
    
    INSTRUCTIONS:
    - Engage in natural conversation about the document content
    - Be educational but conversational
    - Ask follow-up questions when appropriate
    - Explain concepts clearly
    - Use examples from the documents when helpful
    - If the user asks something not in the documents, acknowledge this and offer to discuss related topics that ARE in the documents"""
    
    # Build conversation context
    conversation_context = ""
    if chat_history:
        conversation_context = "\n\n###Previous Conversation\n"
        for entry in chat_history[-3:]:  # Last 3 exchanges
            conversation_context += f"User: {entry.get('user', '')}\nAssistant: {entry.get('bot', '')}\n"
    
    user_message = f"###Document Content\n{context}{conversation_context}\n\n###Current User Message\n{query}\n\nRespond conversationally while staying focused on the document content."
    
    try:
        client = create_groq_client()
        
        # Retry logic for connection errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.7,
                    max_tokens=1024
                )
                return response.choices[0].message.content.strip()
                
            except Exception as retry_e:
                print(f"Chat response attempt {attempt + 1} failed: {retry_e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    raise retry_e
                    
    except Exception as e:
        error_msg = str(e)
        print(f"Error generating chat response: {error_msg}")
        
        if "connection" in error_msg.lower() or "network" in error_msg.lower():
            return "I'm experiencing connection issues. Please check your internet connection and try again."
        elif "rate" in error_msg.lower() or "limit" in error_msg.lower():
            return "The AI service is currently busy. Please wait a moment and try again."
        else:
            return f"I encountered an error generating a response: {error_msg}. Please try again."

def generate_offline_response(query, context, mode="qa"):
    """
    Generate basic offline response when API is unavailable
    """
    if mode == "quiz":
        return f"""📝 **Offline Quiz Mode**
        
I'm currently unable to connect to the AI service to generate custom quiz questions. 

**However, here are some study questions you can consider based on your query: "{query}"**

1. What are the main concepts covered in the uploaded documents?
2. How do the key topics relate to each other?
3. What are the practical applications mentioned?
4. What are the important definitions or terms?
5. What processes or procedures are described?

**Context from your documents:**
{context[:500]}...

*Note: Connect to the internet and try again for AI-generated quiz questions.*"""
    
    else:  # qa or chat mode
        return f"""🤖 **Offline Mode Response**

I'm currently unable to connect to the AI service, but I can share the relevant content from your documents:

**Your question:** {query}

**Relevant content found:**
{context[:1000]}...

📋 **Manual Analysis Suggestions:**
- Review the content above for information related to your question
- Look for key terms and concepts that match your query
- Consider the context and relationships between different parts

🔄 **To get AI-powered answers:**
- Check your internet connection
- Try the "🌐 Test Connection" button in the sidebar
- Use the "🗑️ Reset All" button if issues persist

*This is a basic content retrieval - full AI analysis requires an active connection.*"""

def generate_answer(query, context, mode="qa", chat_history=None):
    """
    Unified answer generation function that routes to appropriate mode
    """
    try:
        if mode == "qa":
            return generate_qa_answer(query, context)
        elif mode == "quiz":
            return generate_quiz_questions(context)
        elif mode == "chat":
            return generate_chat_response(query, context, chat_history or [])
        else:
            return generate_qa_answer(query, context)  # Default to Q&A
    except Exception as e:
        # If all API methods fail, provide offline response
        print(f"All API methods failed, switching to offline mode: {e}")
        return generate_offline_response(query, context, mode)
