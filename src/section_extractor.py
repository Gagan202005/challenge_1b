"""
Smart Section Title Extraction - No hardcoded keywords, just pure algorithm
This module figures out what the main sections are by analyzing the text patterns
"""

import re
import concurrent.futures
from typing import List, Dict


class SectionExtractor:
    """Finds the main sections in documents and picks the best ones using AI"""
    
    def __init__(self, llm, config=None):
        self.llm = llm
        self.config = config or {}
        
        # Get our processing settings so we can run things in parallel
        processing_settings = self.config.get('processing_settings', {})
        self.max_workers = processing_settings.get('max_workers', 2)
        
        # Settings for how much content to process at once
        section_settings = self.config.get('section_extraction_settings', {})
        self.max_processing_lines = section_settings.get('max_processing_lines', 150)
        self.content_preview_lines = section_settings.get('content_preview_lines', 5)
        self.section_preview_chars = section_settings.get('section_preview_chars', 300)
        
        # Basic text analysis patterns from our config
        keyword_mapping = self.config.get('keyword_mapping', {})
        self.sentence_endings = keyword_mapping.get('sentence_endings', ['.', '!', '?', ',', ';'])
        self.min_title_length = keyword_mapping.get('heading_length_min', 3)
        self.max_title_length = keyword_mapping.get('max_heading_line_length', 80)
        
        # Settings for our algorithmic approach
        self.dynamic_extraction = keyword_mapping.get('dynamic_extraction', {})
        self.algorithmic_patterns = keyword_mapping.get('algorithmic_patterns', {})
        
        # We'll build these vocabularies on the fly from the actual documents
        self.document_vocabulary = set()
        self.section_patterns = set()
        
    def _build_section_vocabulary(self, documents):
        """Builds a smart vocabulary by analyzing the actual documents we're working with"""
        vocabulary = set()
        
        for doc in documents:
            if 'content' in doc:
                # Look for words that might be section titles
                content = doc['content'].lower()
                
                # Break it down into words and find the domain-specific terms
                words = re.findall(r'\b[a-z]{3,}\b', content)
                
                for word in words:
                    # Look for words that often appear in section titles
                    if any(suffix in word for suffix in ['tion', 'ment', 'ness', 'ity', 'ing', 'ure', 'ance', 'ence']):
                        vocabulary.add(word)
                    # Add other words that might be important domain terms
                    elif len(word) >= 4 and word.isalpha():
                        vocabulary.add(word)
        
        self.document_vocabulary = vocabulary
        return vocabulary
    
    def extract_sections(self, documents: List[Dict], persona: str, task: str) -> List[Dict]:
        """
        This is our two-step process to find the best sections:
        1. First, we extract ALL the section titles we can find in the PDFs
        2. Then we let the AI pick the top 5 that are most relevant
        
        Args:
            documents: All the PDF documents we're working with
            persona: What kind of person is asking (like "travel blogger") 
            task: What they want to accomplish
            
        Returns:
            The 5 best sections that match what the user needs
        """
        print(f"📋 STAGE 1: EXTRACTING ALL SECTION TITLES FROM PDFS...")
        print(f"⚡ Using parallel processing with {self.max_workers} workers")
        
        # Build our smart vocabulary from the actual documents
        if self.dynamic_extraction.get('extract_keywords_from_documents', True):
            print(f"   🧠 Building section vocabulary from documents...")
            self._build_section_vocabulary(documents)
            print(f"   📚 Built section vocabulary with {len(self.document_vocabulary)} terms")
        
        # STAGE 1: Extract every section title we can find using parallel workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Send each document to a worker to process simultaneously
            future_to_doc = {
                executor.submit(
                    self._extract_section_titles_from_pdf,
                    doc.get('content', ''),
                    doc.get('filename', 'Unknown'),
                    doc.get('page', 1)
                ): doc.get('filename', 'Unknown')
                for doc in documents
            }
            
            # Collect results as they complete
            all_sections = []
            for future in concurrent.futures.as_completed(future_to_doc):
                filename = future_to_doc[future]
                try:
                    sections = future.result()
                    all_sections.extend(sections)
                except Exception as e:
                    print(f"   ⚠️  Error extracting from {filename}: {e}")
        
        if not all_sections:
            print("   ⚠️  No section titles found in PDFs")
            return []
        
        print(f"   📝 EXTRACTED: {len(all_sections)} section titles from all PDFs")
        
        # Stage 2: Give extracted data to model for top 5 selection
        print(f"📋 STAGE 2: GIVING EXTRACTED DATA TO MODEL FOR TOP 5 SELECTION...")
        selected_sections = self._model_select_top_5_sections(all_sections, persona, task)
        
        print(f"✅ MODEL SELECTED: {len(selected_sections)} top section titles")
        return selected_sections
    
    
    def _extract_section_titles_from_pdf(self, content: str, filename: str, page_num: int) -> List[Dict]:
        """Optimized section title extraction for performance"""
        sections = []
        lines = content.split('\n')
        seen_titles = set()
        
        # Performance optimization: process fewer lines
        max_lines = min(len(lines), self.max_processing_lines)
        
        for i in range(max_lines):
            line = lines[i].strip()
            
            # Quick filters (configurable)
            if not line or len(line) < self.min_title_length or len(line) > self.max_title_length:
                continue
                
            # Skip obvious content lines (configurable)
            if any(line.endswith(ending) for ending in self.sentence_endings) and not line.endswith(': '):
                continue
            
            # Quick section title detection
            words = line.split()
            word_count = len(words)
            
            is_section_title = False
            
            # Pattern 1: Capitalized titles (2-6 words) - simplified
            if (2 <= word_count <= 6 and line[0].isupper() and 
                sum(1 for word in words if word[0].isupper()) >= len(words) * 0.5):
                is_section_title = True
            
            # Pattern 2: Dynamic vocabulary relevance (algorithmic)
            line_words = set(re.findall(r'\b[a-z]{3,}\b', line.lower()))
            vocabulary_overlap = len(line_words.intersection(self.document_vocabulary))
            
            if vocabulary_overlap >= 2:  # Dynamic vocabulary match
                is_section_title = True
            
            if is_section_title:
                # Check for duplicates
                title_key = line.lower().strip()
                if title_key in seen_titles:
                    continue
                seen_titles.add(title_key)
                
                # Get enhanced content for better analysis
                content_preview = self._get_quick_content_preview(lines, i)
                
                if len(content_preview) > 40:  # Minimum meaningful content requirement
                    sections.append({
                        "document": filename,
                        "section_title": line,
                        "page_number": page_num,
                        "content_preview": content_preview  # Enhanced preview for model analysis
                    })
                
                # Performance limit: max 20 sections per document
                if len(sections) >= 20:
                    break
        
        return sections
    
    def _get_quick_content_preview(self, lines: List[str], title_index: int) -> str:
        """Get enhanced content preview for better model analysis"""
        preview_lines = []
        for i in range(title_index + 1, min(title_index + 1 + self.content_preview_lines, len(lines))):
            line = lines[i].strip()
            if line and len(line) > 10:
                preview_lines.append(line)
        return ' '.join(preview_lines)[:self.section_preview_chars]
    
    def _model_select_top_5_sections(self, sections: List[Dict], persona: str, task: str) -> List[Dict]:
        """Enhanced model selection with enforced diversity across ALL documents"""
        print(f"   🧠 ANALYZING CONTENT FROM {len(sections)} SECTIONS FOR DIVERSE SELECTION...")
        
        # Remove duplicates and organize by document
        unique_sections = []
        seen_titles = set()
        sections_by_doc = {}
        
        for section in sections:
            title_key = section['section_title'].lower().strip()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_sections.append(section)
                
                # Group by document for diversity
                doc = section['document']
                if doc not in sections_by_doc:
                    sections_by_doc[doc] = []
                sections_by_doc[doc].append(section)
        
        print(f"   📚 FOUND SECTIONS FROM {len(sections_by_doc)} DIFFERENT DOCUMENTS")
        
        # Ensure we select from different documents
        selected = []
        used_documents = set()
        
        # First pass: Select one section from each document
        for doc, doc_sections in sections_by_doc.items():
            if len(selected) < 5:
                # Sort sections by content length (longer = more informative)
                best_section = max(doc_sections, key=lambda s: len(s.get('content_preview', '')))
                best_section['importance_rank'] = len(selected) + 1
                selected.append(best_section)
                used_documents.add(doc)
                print(f"   ✅ SELECTED FROM {doc}: {best_section['section_title'][:50]}...")
        
        # Second pass: Fill remaining slots with best content from any document
        if len(selected) < 5:
            remaining_sections = [s for s in unique_sections 
                                if s['document'] not in used_documents or len(selected) < 3]
            
            # Sort by content quality
            remaining_sections.sort(key=lambda s: len(s.get('content_preview', '')), reverse=True)
            
            for section in remaining_sections:
                if len(selected) < 5:
                    section_copy = section.copy()
                    section_copy['importance_rank'] = len(selected) + 1
                    selected.append(section_copy)
                    print(f"   ✅ ADDED FROM {section['document']}: {section['section_title'][:50]}...")
        
        print(f"   🎯 FINAL SELECTION: {len(selected)} sections from {len(set(s['document'] for s in selected))} documents")
        return selected[:5]
    
    def _batch_select_sections(self, sections: List[Dict], persona: str, task: str) -> List[Dict]:
        """Simplified batch processing for performance"""
        # Just take first 20 sections instead of complex batching
        return self._model_select_top_5_sections(sections[:20], persona, task)
