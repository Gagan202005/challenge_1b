"""
Smart Subsection Extraction - Finds detailed subsections within larger sections
This module breaks down sections into smaller, more focused pieces that users actually need
"""

import re
import concurrent.futures
from typing import List, Dict


class SubsectionExtractor:
    """Finds the best subsections within documents using AI to pick the most relevant ones"""
    
    def __init__(self, llm, config=None):
        self.llm = llm
        self.config = config or {}
        
        # Set up parallel processing for faster results
        processing_settings = self.config.get('processing_settings', {})
        self.max_workers = processing_settings.get('max_workers', 2)
        
        # Configure how much content we process at once
        content_settings = self.config.get('content_extraction_settings', {})
        self.max_content_lines = content_settings.get('max_content_lines', 300)
        self.max_content_chars = content_settings.get('max_content_chars', 500)
        self.content_context_lines = content_settings.get('content_context_lines', 5)
        self.max_subsections_per_doc = content_settings.get('max_subsections_per_doc', 60)
        self.top_subsections_per_doc = content_settings.get('top_subsections_per_doc', 40)
        self.min_meaningful_content_length = content_settings.get('min_meaningful_content_length', 60)
        self.heading_length_min = content_settings.get('heading_length_min', 3)
        self.heading_length_max = content_settings.get('heading_length_max', 100)
        
        # Get our algorithmic analysis patterns
        keyword_mapping = self.config.get('keyword_mapping', {})
        self.dynamic_extraction = keyword_mapping.get('dynamic_extraction', {})
        self.algorithmic_patterns = keyword_mapping.get('algorithmic_patterns', {})
        self.content_analysis = keyword_mapping.get('content_analysis', {})
        self.filtering_rules = keyword_mapping.get('filtering_rules', {})
        
        # Text patterns we look for when analyzing content
        self.list_markers = keyword_mapping.get('list_markers', ['1.', '2.', '3.', '4.', '5.', '•', '-', '*'])
        self.exclude_prefixes = keyword_mapping.get('exclude_prefixes', ['Note:', 'Tip:', 'Warning:'])
        self.sentence_endings = keyword_mapping.get('sentence_endings', ['.', '!', '?', ',', ';'])
        self.stop_words = keyword_mapping.get('stop_words', ['the', 'and', 'for', 'with', 'from', 'that', 'this'])
        self.min_word_length = self.filtering_rules.get('min_word_length', 3)
        self.max_heading_words = keyword_mapping.get('max_heading_words', 10)
        self.max_heading_line_length = keyword_mapping.get('max_heading_line_length', 80)
        
        # We build these dynamically from the content we're analyzing
        self.document_vocabulary = set()
        self.extracted_keywords = {}
        self.domain_terms = set()
        
    def extract_subsections(self, documents: List[Dict], persona: str, task: str) -> List[Dict]:
        """
        Two-stage process:
        1. Extract ALL subsections from PDFs with full content
        2. Give extracted data to model to select top 5
        """
        print(f"🔍 STAGE 1: EXTRACTING ALL SUBSECTIONS FROM PDFS...")
        print(f"⚡ Using parallel processing with {self.max_workers} workers")
        
        # Build document vocabulary for dynamic keyword extraction
        if self.content_analysis.get('extract_keywords_from_documents', True):
            print(f"   🧠 Building dynamic vocabulary from documents...")
            self._build_document_vocabulary(documents)
            print(f"   📚 Built vocabulary with {len(self.document_vocabulary)} terms")
        
        # Stage 1: Extract all subsections using parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_doc = {
                executor.submit(
                    self._extract_subsections_from_pdf,
                    doc.get('content', ''),
                    doc.get('filename', 'Unknown'),
                    doc.get('page', 1)
                ): doc.get('filename', 'Unknown')
                for doc in documents
            }
            
            all_subsections = []
            for future in concurrent.futures.as_completed(future_to_doc):
                filename = future_to_doc[future]
                try:
                    subsections = future.result()
                    all_subsections.extend(subsections)
                except Exception as e:
                    print(f"   ⚠️  Error extracting from {filename}: {e}")
        
        if not all_subsections:
            return []
        
        print(f"   📝 EXTRACTED: {len(all_subsections)} subsections from all PDFs")
        
        # Stage 2: Give extracted data to model for top 5 selection
        print(f"🔍 STAGE 2: GIVING EXTRACTED DATA TO MODEL FOR TOP 5 SELECTION...")
        selected_subsections = self._model_select_top_5_subsections(all_subsections, persona, task)
        
        print(f"✅ MODEL SELECTED: {len(selected_subsections)} top subsections")
        return selected_subsections
    
    def _extract_subsections_from_pdf(self, content: str, filename: str, page_num: int) -> List[Dict]:
        """Enhanced subsection extraction with better content analysis"""
        subsections = []
        lines = content.split('\n')
        
        # Enhanced processing: analyze more content but with smarter filtering
        max_lines = min(len(lines), self.max_content_lines)
        
        i = 0
        current_section_context = ""
        
        while i < max_lines:
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Enhanced heading detection with context awareness
            if self._is_enhanced_heading(line, current_section_context):
                # Found a heading, collect enhanced content
                heading = line
                content_lines = []
                i += 1
                
                # Collect more comprehensive content
                content_char_count = 0
                line_count = 0
                while i < max_lines and line_count < self.content_context_lines:
                    content_line = lines[i].strip()
                    
                    # Stop if we hit another heading
                    if content_line and self._is_enhanced_heading(content_line, heading):
                        break
                    
                    if content_line and content_char_count < self.max_content_chars:
                        content_lines.append(content_line)
                        content_char_count += len(content_line)
                        line_count += 1
                    
                    i += 1
                
                # Create enhanced subsection with better content
                if content_lines:
                    combined_text = f"{heading}: {' '.join(content_lines)[:self.max_content_chars]}"
                else:
                    combined_text = heading
                
                # Enhanced quality filtering
                if self._is_quality_subsection(combined_text, filename):
                    subsections.append({
                        "document": filename,
                        "refined_text": combined_text,
                        "page_number": page_num,
                        "content_type": "heading_with_content",
                        "quality_score": self._calculate_quality_score(combined_text)
                    })
                
                current_section_context = heading
                
            else:
                # Enhanced standalone content detection
                if self._is_quality_standalone_content(line):
                    # Look for related content in next few lines
                    extended_content = self._extract_related_content(lines, i, max_lines)
                    
                    if len(extended_content) >= self.min_meaningful_content_length:
                        subsections.append({
                            "document": filename,
                            "refined_text": extended_content[:self.max_content_chars],
                            "page_number": page_num,
                            "content_type": "standalone_content",
                            "quality_score": self._calculate_quality_score(extended_content)
                        })
                i += 1
            
            # Performance limit with quality consideration
            if len(subsections) >= self.max_subsections_per_doc:
                break
        
        # Sort by quality score and return top subsections
        subsections.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        return subsections[:self.top_subsections_per_doc]
    
    def _is_enhanced_heading(self, line: str, context: str = "") -> bool:
        """Enhanced heading detection with context awareness"""
        if not line or len(line) < self.heading_length_min or len(line) > self.heading_length_max:
            return False
        
        # Enhanced patterns for better heading detection (all configurable)
        patterns = [
            # Traditional headings
            (len(line) <= self.max_heading_line_length and len(line.split()) <= self.max_heading_words and 
             line[0].isupper() and not any(line.endswith(ending) for ending in self.sentence_endings if ending != ':')),
            # Colon-ending headings
            (line.endswith(':') and len(line.split()) <= self.max_heading_words),
            # Numbered/bulleted items (configurable)
            any(line.startswith(marker) for marker in self.list_markers),
            # Topic-specific headings (dynamic pattern detection)
            self._is_topic_heading_dynamic(line) and len(line.split()) <= self.max_heading_words and line[0].isupper()
        ]
        
        return any(patterns)
    
    def _is_topic_heading_dynamic(self, line: str) -> bool:
        """Dynamically determine if line is a topic heading using algorithmic patterns"""
        line_lower = line.lower()
        
        # Check for domain terms from vocabulary
        words = line_lower.split()
        domain_score = 0
        
        for word in words:
            cleaned_word = ''.join(c for c in word if c.isalnum())
            if cleaned_word in self.document_vocabulary:
                domain_score += 1
            # Check for conceptual suffixes
            noun_suffixes = self.algorithmic_patterns.get('noun_indicators', ['tion', 'sion', 'ness', 'ment'])
            if any(cleaned_word.endswith(suffix) for suffix in noun_suffixes):
                domain_score += 2
        
        # Check structural patterns
        structural_patterns = [
            line.endswith(':'),  # Section indicators
            any(line_lower.startswith(prefix) for prefix in ['overview', 'introduction', 'conclusion']),
            len(words) <= 5 and all(word[0].isupper() for word in words if word),  # Title case
            any(char.isdigit() for char in line) and ':' in line  # Numbered sections
        ]
        
        return domain_score >= 1 or any(structural_patterns)
    
    def _is_quality_subsection(self, text: str, filename: str) -> bool:
        """Enhanced quality assessment for subsections"""
        if len(text) < 40 or len(text) > 700:
            return False
        
        # Quality indicators (dynamic extraction)
        quality_indicators = [
            ':' in text,  # Structured content
            '.' in text and len(text.split('.')) >= 2,  # Complete sentences
            self._has_quality_content_dynamic(text),  # Dynamic descriptive content detection
            len(text.split()) >= 8,  # Substantial content
        ]
        
        return sum(quality_indicators) >= 2
    
    def _has_quality_content_dynamic(self, text: str) -> bool:
        """Dynamically assess content quality using algorithmic patterns"""
        text_lower = text.lower()
        quality_score = 0
        
        # Extract and analyze patterns
        patterns = self._analyze_text_patterns(text)
        
        # Score based on different types of quality indicators
        quality_score += len(patterns['descriptive_words']) * 2  # Descriptive words are valuable
        quality_score += len(patterns['action_words'])  # Action words indicate activity/process
        quality_score += len(patterns['topic_words'])  # Topic words indicate subject matter
        
        # Check for domain-specific terms
        domain_terms = self._extract_domain_terms(text)
        quality_score += len(domain_terms) * 3  # Domain terms are highly valuable
        
        # Check vocabulary overlap
        words = text_lower.split()
        vocab_overlap = sum(1 for word in words if ''.join(c for c in word if c.isalnum()) in self.document_vocabulary)
        quality_score += vocab_overlap
        
        # Structural quality indicators
        if any(indicator in text_lower for indicator in ['features', 'offers', 'provides', 'includes']):
            quality_score += 5
        if any(indicator in text_lower for indicator in ['experience', 'explore', 'discover', 'enjoy']):
            quality_score += 3
        if any(indicator in text_lower for indicator in ['popular', 'famous', 'known', 'recommended']):
            quality_score += 2
        
        return quality_score >= 5  # Threshold for quality content
    
    def _is_quality_standalone_content(self, line: str) -> bool:
        """Enhanced detection of quality standalone content (configurable)"""
        return (50 <= len(line) <= 600 and 
                ('.' in line or ':' in line) and
                len(line.split()) >= 6 and
                not any(line.startswith(prefix) for prefix in self.exclude_prefixes))
    
    def _extract_related_content(self, lines: List[str], start_idx: int, max_lines: int) -> str:
        """Extract related content for standalone sections"""
        content_parts = [lines[start_idx].strip()]
        
        for i in range(start_idx + 1, min(start_idx + 4, max_lines)):
            if i < len(lines):
                line = lines[i].strip()
                if line and len(line) > 20 and not self._is_enhanced_heading(line):
                    content_parts.append(line)
                else:
                    break
        
        return ' '.join(content_parts)
    
    def _calculate_quality_score(self, text: str) -> float:
        """Calculate quality score for subsection ranking"""
        score = 0.0
        
        # Length score (optimal range)
        if 100 <= len(text) <= 400:
            score += 2.0
        elif 50 <= len(text) <= 600:
            score += 1.0
        
        # Content structure score
        if ':' in text:
            score += 1.5
        if text.count('.') >= 2:
            score += 1.0
        
        # Descriptive content score (dynamic analysis)
        patterns = self._analyze_text_patterns(text)
        domain_terms = self._extract_domain_terms(text)
        
        # Score based on pattern analysis
        score += len(patterns['descriptive_words']) * 0.3
        score += len(patterns['action_words']) * 0.2
        score += len(patterns['topic_words']) * 0.2
        score += len(domain_terms) * 0.4
        
        # Sentence completeness
        if len(text.split()) >= 10:
            score += 1.0
        
        return min(score, 10.0)  # Cap at 10
    
    def _build_document_vocabulary(self, documents: List[Dict]) -> None:
        """Build vocabulary from all documents dynamically"""
        if not self.content_analysis.get('extract_keywords_from_documents', True):
            return
            
        for doc in documents:
            content = doc.get('content', '').lower()
            words = content.split()
            
            # Extract meaningful words
            for word in words:
                cleaned_word = ''.join(c for c in word if c.isalnum())
                if (len(cleaned_word) >= self.min_word_length and 
                    cleaned_word not in self.stop_words and
                    cleaned_word.isalpha()):
                    self.document_vocabulary.add(cleaned_word)
    
    def _extract_domain_terms(self, text: str) -> List[str]:
        """Extract domain-specific terms using algorithmic patterns"""
        text_lower = text.lower()
        domain_terms = []
        
        # Extract terms with specific suffixes that indicate concepts
        noun_suffixes = self.algorithmic_patterns.get('noun_indicators', ['tion', 'sion', 'ness', 'ment', 'ity'])
        quality_suffixes = self.algorithmic_patterns.get('quality_suffixes', ['ful', 'able', 'ive', 'ant', 'ent'])
        
        words = text_lower.split()
        for word in words:
            cleaned_word = ''.join(c for c in word if c.isalnum())
            if len(cleaned_word) >= self.min_word_length:
                # Check for conceptual suffixes
                if any(cleaned_word.endswith(suffix) for suffix in noun_suffixes + quality_suffixes):
                    domain_terms.append(cleaned_word)
                # Check for capitalized terms (proper nouns, places, etc.)
                elif word[0].isupper() and len(word) > 3:
                    domain_terms.append(cleaned_word)
        
        return list(set(domain_terms))
    
    def _analyze_text_patterns(self, text: str) -> Dict[str, List[str]]:
        """Analyze text to extract patterns algorithmically"""
        patterns = {
            'action_words': [],
            'descriptive_words': [],
            'topic_words': [],
            'contextual_words': []
        }
        
        words = text.lower().split()
        verb_indicators = self.algorithmic_patterns.get('verb_indicators', ['ing', 'ed', 'en', 'er'])
        quality_suffixes = self.algorithmic_patterns.get('quality_suffixes', ['ful', 'able', 'ive', 'ant', 'ent'])
        
        for i, word in enumerate(words):
            cleaned_word = ''.join(c for c in word if c.isalnum())
            if len(cleaned_word) < self.min_word_length or cleaned_word in self.stop_words:
                continue
                
            # Action words (verbs)
            if any(cleaned_word.endswith(indicator) for indicator in verb_indicators):
                patterns['action_words'].append(cleaned_word)
            
            # Descriptive words (adjectives)
            elif any(cleaned_word.endswith(suffix) for suffix in quality_suffixes):
                patterns['descriptive_words'].append(cleaned_word)
            
            # Topic words (nouns in specific contexts)
            elif i > 0 and words[i-1] in ['the', 'a', 'an', 'this', 'that']:
                patterns['topic_words'].append(cleaned_word)
            
            # Contextual words (words near colons, periods)
            elif i < len(words) - 1 and any(p in words[i+1] for p in [':', '.']):
                patterns['contextual_words'].append(cleaned_word)
        
        # Remove duplicates and return top terms
        for key in patterns:
            patterns[key] = list(set(patterns[key]))[:self.dynamic_extraction.get('max_keywords_per_pattern', 10)]
        
        return patterns
    
    def _model_select_top_5_subsections(self, subsections: List[Dict], persona: str, task: str) -> List[Dict]:
        """Enhanced selection with comprehensive content analysis and quality scoring"""
        print(f"   🧠 ANALYZING {len(subsections)} SUBSECTIONS WITH ENHANCED QUALITY ASSESSMENT...")
        
        # Filter and organize high-quality subsections by document
        quality_subsections = []
        subsections_by_doc = {}
        
        for subsection in subsections:
            content = subsection.get('refined_text', '')
            quality_score = subsection.get('quality_score', 0)
            
            # Enhanced quality filtering
            if (len(content) >= 50 and 
                quality_score >= 2.0 and 
                ('.' in content or ':' in content)):
                
                quality_subsections.append(subsection)
                
                # Group by document for diversity
                doc = subsection['document']
                if doc not in subsections_by_doc:
                    subsections_by_doc[doc] = []
                subsections_by_doc[doc].append(subsection)
        
        print(f"   📚 FOUND {len(quality_subsections)} QUALITY SUBSECTIONS FROM {len(subsections_by_doc)} DOCUMENTS")
        
        # Enhanced selection with quality and relevance analysis
        selected = []
        used_documents = set()
        
        # First pass: Select highest quality subsection from each document
        for doc, doc_subsections in subsections_by_doc.items():
            if len(selected) < 5:
                # Sort by quality score and content relevance
                best_subsection = max(doc_subsections, 
                                    key=lambda s: (
                                        s.get('quality_score', 0) * 2 +  # Quality weight
                                        len(s.get('refined_text', '')) / 100 +  # Length bonus
                                        (1 if self._is_task_relevant(s.get('refined_text', ''), task) else 0)
                                    ))
                
                # Add enhanced metadata
                best_subsection['selection_reason'] = f"Best quality from {doc}"
                best_subsection['relevance_score'] = self._calculate_relevance_score(
                    best_subsection.get('refined_text', ''), persona, task)
                
                selected.append(best_subsection)
                used_documents.add(doc)
                print(f"   ✅ SELECTED FROM {doc}: Quality={best_subsection.get('quality_score', 0):.1f}")
        
        # Second pass: Fill remaining slots with best overall content
        if len(selected) < 5:
            remaining_subsections = [s for s in quality_subsections 
                                   if s['document'] not in used_documents]
            
            # Sort by combined quality and relevance score
            remaining_subsections.sort(
                key=lambda s: (
                    s.get('quality_score', 0) + 
                    self._calculate_relevance_score(s.get('refined_text', ''), persona, task)
                ), 
                reverse=True
            )
            
            for subsection in remaining_subsections:
                if len(selected) < 5:
                    subsection['selection_reason'] = "High quality and relevance"
                    subsection['relevance_score'] = self._calculate_relevance_score(
                        subsection.get('refined_text', ''), persona, task)
                    selected.append(subsection)
                    print(f"   ✅ ADDED FROM {subsection['document']}: "
                          f"Quality={subsection.get('quality_score', 0):.1f}")
        
        # Final ranking by combined score
        selected.sort(key=lambda s: (
            s.get('quality_score', 0) + s.get('relevance_score', 0)
        ), reverse=True)
        
        print(f"   🎯 FINAL SELECTION: {len(selected)} high-quality subsections from "
              f"{len(set(s['document'] for s in selected))} documents")
        
        return selected[:5]
    
    def _batch_select_subsections(self, subsections: List[Dict], persona: str, task: str) -> List[Dict]:
        """Simplified batch processing - remove complex batching to improve performance"""
        # Instead of complex batching, just take first 30 and process directly
        return self._model_select_top_5_subsections(subsections[:30], persona, task)
    
    def _extract_task_keywords(self, task: str) -> List[str]:
        """Dynamically extract keywords based on task content using algorithmic analysis"""
        task_lower = task.lower()
        keywords = []
        
        # Extract patterns from task text
        patterns = self._analyze_text_patterns(task)
        
        # Combine all pattern types
        for pattern_type, pattern_words in patterns.items():
            keywords.extend(pattern_words)
        
        # Extract domain terms from task
        domain_terms = self._extract_domain_terms(task)
        keywords.extend(domain_terms)
        
        # Extract meaningful words directly from task (algorithmic filtering)
        task_words = task_lower.split()
        meaningful_words = []
        for word in task_words:
            cleaned_word = ''.join(c for c in word if c.isalnum())
            if (len(cleaned_word) > self.min_word_length and 
                cleaned_word not in self.stop_words and
                cleaned_word.isalpha() and
                # Additional algorithmic filters
                not cleaned_word.isdigit() and
                len(set(cleaned_word)) > 2):  # Not repetitive characters
                meaningful_words.append(cleaned_word)
        
        keywords.extend(meaningful_words[:5])  # Add up to 5 meaningful words from task
        
        return list(set(keywords))  # Remove duplicates
    
    def _extract_persona_keywords(self, persona: str) -> List[str]:
        """Dynamically extract keywords based on persona content using algorithmic analysis"""
        persona_lower = persona.lower()
        keywords = []
        
        # Extract patterns from persona text
        patterns = self._analyze_text_patterns(persona)
        
        # Focus on descriptive and topic words for personas
        keywords.extend(patterns['descriptive_words'])
        keywords.extend(patterns['topic_words'])
        
        # Extract domain terms from persona
        domain_terms = self._extract_domain_terms(persona)
        keywords.extend(domain_terms)
        
        # Extract meaningful words from persona (algorithmic filtering)
        persona_words = persona_lower.split()
        meaningful_words = []
        for word in persona_words:
            cleaned_word = ''.join(c for c in word if c.isalnum())
            if (len(cleaned_word) > self.min_word_length and 
                cleaned_word not in self.stop_words and
                cleaned_word.isalpha() and
                # Persona-specific filters
                not cleaned_word.endswith('er') or len(cleaned_word) > 6):  # Professional roles
                meaningful_words.append(cleaned_word)
        
        keywords.extend(meaningful_words)
        
        return list(set(keywords))  # Remove duplicates
    
    def _is_task_relevant(self, text: str, task: str) -> bool:
        """Check if content is relevant to the specific task using dynamic keyword extraction"""
        text_lower = text.lower()
        task_keywords = self._extract_task_keywords(task)
        
        return any(keyword in text_lower for keyword in task_keywords)
    
    def _calculate_relevance_score(self, text: str, persona: str, task: str) -> float:
        """Calculate relevance score based on persona and task using dynamic keyword extraction"""
        score = 0.0
        text_lower = text.lower()
        
        # Dynamic persona-specific relevance
        persona_keywords = self._extract_persona_keywords(persona)
        score += sum(0.5 for keyword in persona_keywords if keyword in text_lower)
        
        # Dynamic task-specific relevance
        if self._is_task_relevant(text, task):
            score += 2.0
        
        # Content informativeness
        if len(text.split()) >= 15:
            score += 1.0
        if ':' in text and '.' in text:
            score += 1.0
        
        return min(score, 8.0)  # Cap at 8
