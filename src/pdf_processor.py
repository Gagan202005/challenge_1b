"""
PDF Processing - Handles loading and reading PDF files efficiently
Takes care of all the heavy lifting when it comes to extracting text from PDFs
"""

import os
import concurrent.futures
from typing import List, Dict
from langchain_community.document_loaders import UnstructuredPDFLoader


class PDFProcessor:
    """Loads PDF files and extracts text content using parallel processing for speed"""
    
    def __init__(self, pdf_directory: str = "data/pdfs", max_workers: int = 4, config: dict = None):
        self.pdf_directory = pdf_directory
        self.max_workers = max_workers
        self.config = config or {}
        
        # Figure out what file types we can handle
        processing_settings = self.config.get('processing_settings', {})
        self.supported_extensions = processing_settings.get('supported_file_extensions', ['.pdf'])
        
    def load_pdfs(self) -> List[Dict]:
        """
        Loads all PDF files from our directory using multiple workers for faster processing
        
        Returns:
            A list of dictionaries containing the text content and file info for each PDF
        """
        documents = []
        
        if not os.path.exists(self.pdf_directory):
            print(f"❌ PDF directory not found: {self.pdf_directory}")
            return documents
            
        # Find all the PDF files we can work with
        pdf_files = [f for f in os.listdir(self.pdf_directory) 
                    if any(f.lower().endswith(ext) for ext in self.supported_extensions)]
        
        if not pdf_files:
            print(f"❌ No PDF files found in {self.pdf_directory}")
            return documents
            
        print(f"📄 Found {len(pdf_files)} PDF files:")
        print(f"⚡ Using parallel processing with {self.max_workers} workers")
        
        # Process multiple PDFs at the same time for better performance
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Start processing all PDFs simultaneously
            future_to_pdf = {
                executor.submit(self._load_single_pdf, pdf_file): pdf_file 
                for pdf_file in pdf_files
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_pdf):
                pdf_file = future_to_pdf[future]
                try:
                    pdf_documents = future.result()
                    documents.extend(pdf_documents)
                    print(f"   ✅ Loaded: {pdf_file}")
                except Exception as e:
                    print(f"   ❌ Failed to load {pdf_file}: {e}")
        
        print(f"✅ Successfully loaded {len(documents)} document pages using parallel processing")
        return documents
    
    def _load_single_pdf(self, pdf_file: str) -> List[Dict]:
        """
        Load a single PDF file (used by parallel processing)
        
        Args:
            pdf_file: Name of the PDF file to load
            
        Returns:
            List of document dictionaries from this PDF
        """
        documents = []
        
        try:
            pdf_path = os.path.join(self.pdf_directory, pdf_file)
            
            # Load PDF using UnstructuredPDFLoader
            loader = UnstructuredPDFLoader(pdf_path)
            docs = loader.load()
            
            # Convert to our format
            for i, doc in enumerate(docs):
                doc_dict = {
                    "filename": pdf_file,
                    "content": doc.page_content,
                    "page": doc.metadata.get("page_number", i + 1),
                    "source": pdf_path
                }
                documents.append(doc_dict)
                
        except Exception as e:
            print(f"   ❌ Error loading {pdf_file}: {e}")
            
        return documents
                
        print(f"✅ Successfully loaded {len(documents)} document pages")
        return documents
    
    def load_pdfs_from_input(self, document_list: List[Dict]) -> List[Dict]:
        """
        Load PDF files specified in the input configuration using parallel processing
        
        Args:
            document_list: List of document dictionaries from input config
            
        Returns:
            List of document dictionaries with content and metadata
        """
        documents = []
        
        if not os.path.exists(self.pdf_directory):
            print(f"❌ PDF directory not found: {self.pdf_directory}")
            return documents
            
        print(f"📄 Loading {len(document_list)} specified PDF files:")
        print(f"⚡ Using parallel processing with {self.max_workers} workers")
        
        # Extract filenames for parallel processing
        filenames = [doc_info["filename"] for doc_info in document_list]
        
        # Use ThreadPoolExecutor for parallel PDF loading
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all PDF loading tasks
            future_to_pdf = {
                executor.submit(self._load_single_pdf_from_input, filename): filename 
                for filename in filenames
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_pdf):
                filename = future_to_pdf[future]
                try:
                    pdf_documents = future.result()
                    documents.extend(pdf_documents)
                    if pdf_documents:
                        print(f"   ✅ Loaded: {filename}")
                except Exception as e:
                    print(f"   ❌ Failed to load {filename}: {e}")
        
        print(f"✅ Successfully loaded {len(documents)} document pages using parallel processing")
        return documents
    
    def _load_single_pdf_from_input(self, filename: str) -> List[Dict]:
        """
        Load a single PDF file from input specification (used by parallel processing)
        
        Args:
            filename: Name of the PDF file to load
            
        Returns:
            List of document dictionaries from this PDF
        """
        documents = []
        
        try:
            pdf_path = os.path.join(self.pdf_directory, filename)
            
            if not os.path.exists(pdf_path):
                print(f"   ❌ File not found: {filename}")
                return documents
                
            # Load PDF using UnstructuredPDFLoader
            loader = UnstructuredPDFLoader(pdf_path)
            docs = loader.load()
            
            # Convert to our format
            for i, doc in enumerate(docs):
                doc_dict = {
                    "filename": filename,
                    "content": doc.page_content,
                    "page": doc.metadata.get("page_number", i + 1),
                    "source": pdf_path
                }
                documents.append(doc_dict)
                
        except Exception as e:
            print(f"   ❌ Error loading {filename}: {e}")
            
        return documents
    
    def get_document_summary(self, documents: List[Dict]) -> Dict:
        """Get summary statistics of loaded documents"""
        if not documents:
            return {"total_docs": 0, "total_pages": 0, "filenames": []}
            
        filenames = list(set([doc["filename"] for doc in documents]))
        total_chars = sum(len(doc["content"]) for doc in documents)
        
        return {
            "total_docs": len(filenames),
            "total_pages": len(documents),
            "filenames": filenames,
            "total_characters": total_chars,
            "avg_chars_per_page": total_chars // len(documents) if documents else 0
        }
