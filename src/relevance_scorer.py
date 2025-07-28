"""
Smart Relevance Scoring - Figures out which content is most useful for the user
Fast and efficient scoring that actually understands what the user needs
"""

import re
from typing import List, Dict


class RelevanceScorer:
    """Uses AI to score and rank content based on how well it matches what the user wants"""
    
    def __init__(self, llm, config=None):
        self.llm = llm
        self.config = config or {}
        
        # Get our scoring configuration settings
        scorer_settings = self.config.get('relevance_scorer_settings', {})
        self.max_items_to_evaluate = scorer_settings.get('max_items_to_evaluate', 30)
        self.target_selections = scorer_settings.get('target_selections', 5)
        self.score_step = scorer_settings.get('score_step', 5)
        self.content_preview_length = scorer_settings.get('content_preview_length', 200)
        self.dedup_title_prefix_chars = scorer_settings.get('dedup_title_prefix_chars', 50)
        self.dedup_content_chars = scorer_settings.get('dedup_content_chars', 150)
        self.dedup_content_start_chars = scorer_settings.get('dedup_content_start_chars', 80)
        self.min_meaningful_content_chars = scorer_settings.get('min_meaningful_content_chars', 30)
        self.section_content_preview_chars = scorer_settings.get('section_content_preview_chars', 400)
        self.subsection_content_preview_chars = scorer_settings.get('subsection_content_preview_chars', 500)
        self.batch_top_selections = scorer_settings.get('batch_top_selections', 3)
        self.max_balanced_candidates = scorer_settings.get('max_balanced_candidates', 60)
        self.max_final_candidates = scorer_settings.get('max_final_candidates', 30)
        self.max_relevance_score = scorer_settings.get('max_relevance_score', 100.0)
        self.content_dedup_chars = scorer_settings.get('content_dedup_chars', 100)
        
    def score_sections(self, sections: List[Dict], persona: str, task: str) -> List[Dict]:
        """
        Simple and effective: show all sections to our AI and let it pick the best ones
        """
        print(f"   🎯 Model-based selection for {len(sections)} sections...")
        
        if len(sections) <= 5:
            print(f"   ⚠️  Only {len(sections)} sections available, returning all")
            for i, section in enumerate(sections):
                section['importance_rank'] = i + 1
            return sections
        
        # Get rid of any duplicate content first
        unique_sections = self._remove_duplicate_sections(sections)
        print(f"   📝 After deduplication: {len(unique_sections)} unique sections")
        
        # Let our AI model pick the top 5 based on all the content
        selected_sections = self._model_select_top_sections(unique_sections, persona, task)
        
        print(f"   ✅ Model selected {len(selected_sections)} sections")
        return selected_sections
    
    def score_subsections(self, subsections: List[Dict], persona: str, task: str) -> List[Dict]:
        """
        Simple approach: Give all subsections to model and ask for top 5 based on content
        """
        print(f"   🎯 Model-based selection for {len(subsections)} subsections...")
        
        if len(subsections) <= 5:
            print(f"   ⚠️  Only {len(subsections)} subsections available, returning all")
            return subsections
        
        # Remove duplicates first
        unique_subsections = self._remove_duplicate_subsections(subsections)
        print(f"   📝 After deduplication: {len(unique_subsections)} unique subsections")
        
        # Let model select top 5 based on all available content
        selected_subsections = self._model_select_top_subsections(unique_subsections, persona, task)
        
        print(f"   ✅ Model selected {len(selected_subsections)} subsections")
        return selected_subsections
    
    def _model_select_top_sections(self, sections: List[Dict], persona: str, task: str) -> List[Dict]:
        """
        Give all sections to model and let it choose top 5 based on content analysis
        """
        print(f"      🤖 Model analyzing ALL {len(sections)} sections from all PDFs...")
        
        # Create comprehensive section data for model
        section_data = []
        for i, section in enumerate(sections, 1):
            title = section['section_title']
            doc = section['document']
            content = section.get('section_content', section.get('refined_text', ''))
            
            if content and len(content.strip()) > 20:
                # Include actual content for better analysis
                content_preview = content[:400].replace('\n', ' ').strip()
                section_info = f"{i}. TITLE: {title}\nDOCUMENT: {doc}\nCONTENT: {content_preview}..."
            else:
                # Title only if no content
                section_info = f"{i}. TITLE: {title}\nDOCUMENT: {doc}\nCONTENT: [Title only]"
            
            section_data.append(section_info)
        
        # Simple batching for large collections to avoid token limits
        if len(sections) > 50:
            return self._batch_model_selection_sections(sections, section_data, persona, task)
        
        # Direct model selection for manageable size
        return self._direct_model_selection_sections(sections, section_data, persona, task)
    
    def _model_select_top_subsections(self, subsections: List[Dict], persona: str, task: str) -> List[Dict]:
        """
        Give all subsections to model and let it choose top 5 based on content analysis
        """
        print(f"      🤖 Model analyzing ALL {len(subsections)} subsections from all PDFs...")
        
        # Create comprehensive subsection data for model
        subsection_data = []
        for i, subsection in enumerate(subsections, 1):
            doc = subsection['document']
            content = subsection['refined_text']
            
            # Include full content for analysis (truncated for model limits)
            content_preview = content[:500].replace('\n', ' ').strip()
            subsection_info = f"{i}. DOCUMENT: {doc}\nCONTENT: {content_preview}..."
            
            subsection_data.append(subsection_info)
        
        # Simple batching for large collections to avoid token limits
        if len(subsections) > 40:
            return self._batch_model_selection_subsections(subsections, subsection_data, persona, task)
        
        # Direct model selection for manageable size
        return self._direct_model_selection_subsections(subsections, subsection_data, persona, task)
    
    def _direct_model_selection_sections(self, sections: List[Dict], section_data: List[str], persona: str, task: str) -> List[Dict]:
        """
        Direct model selection for sections when collection is manageable size
        """
        prompt = f"""You are a {persona} working on: {task}

TASK: Analyze ALL {len(sections)} sections below and select the TOP 5 MOST RELEVANT ones based on their actual content and titles.

Consider sections from ALL documents. Focus on content that would be most useful for your specific task.

SECTIONS FROM ALL PDFS:
{chr(10).join(section_data)}

CRITERIA:
1. Most useful for: "{task}"
2. Best for a {persona}
3. Actionable, practical content
4. Consider all PDFs equally
5. Focus on content quality and relevance

INSTRUCTION: Select exactly 5 sections that would be MOST helpful for your task.

RESPONSE FORMAT: Return only the numbers separated by commas (e.g., "3, 7, 12, 18, 25")

YOUR TOP 5 SECTIONS:"""

        try:
            response = self.llm.invoke(prompt)
            numbers = re.findall(r'\d+', response.content.strip())
            
            # Convert to selected sections
            selected_sections = []
            for num_str in numbers[:5]:
                idx = int(num_str) - 1  # Convert to 0-based index
                if 0 <= idx < len(sections):
                    section_copy = sections[idx].copy()
                    section_copy['importance_rank'] = len(selected_sections) + 1
                    selected_sections.append(section_copy)
            
            # Fill remaining if needed
            if len(selected_sections) < 5:
                used_indices = set()
                for section in selected_sections:
                    for i, orig_section in enumerate(sections):
                        if (section['section_title'] == orig_section['section_title'] and 
                            section['document'] == orig_section['document']):
                            used_indices.add(i)
                            break
                
                for i, section in enumerate(sections):
                    if i not in used_indices and len(selected_sections) < 5:
                        section_copy = section.copy()
                        section_copy['importance_rank'] = len(selected_sections) + 1
                        selected_sections.append(section_copy)
            
            print(f"      ✅ Model selected {len(selected_sections)} sections from all PDFs")
            return selected_sections[:5]
            
        except Exception as e:
            print(f"      ⚠️  Error in model selection: {e}")
            # Fallback: return first 5 sections
            fallback_sections = []
            for i, section in enumerate(sections[:5]):
                section_copy = section.copy()
                section_copy['importance_rank'] = i + 1
                fallback_sections.append(section_copy)
            return fallback_sections
    
    def _direct_model_selection_subsections(self, subsections: List[Dict], subsection_data: List[str], persona: str, task: str) -> List[Dict]:
        """
        Direct model selection for subsections when collection is manageable size
        """
        prompt = f"""You are a {persona} working on: {task}

TASK: Analyze ALL {len(subsections)} content sections below and select the TOP 5 MOST RELEVANT ones based on their actual content.

Consider content from ALL documents. Focus on content that would be most useful for your specific task.

CONTENT SECTIONS FROM ALL PDFS:
{chr(10).join(subsection_data)}

CRITERIA:
1. Most useful for: "{task}"
2. Best for a {persona}
3. Contains specific, actionable information
4. Consider all PDFs equally
5. Focus on content quality and practical value

INSTRUCTION: Select exactly 5 content sections that would be MOST helpful for your task.

RESPONSE FORMAT: Return only the numbers separated by commas (e.g., "2, 8, 15, 22, 35")

YOUR TOP 5 CONTENT SECTIONS:"""

        try:
            response = self.llm.invoke(prompt)
            numbers = re.findall(r'\d+', response.content.strip())
            
            # Convert to selected subsections
            selected_subsections = []
            for num_str in numbers[:5]:
                idx = int(num_str) - 1  # Convert to 0-based index
                if 0 <= idx < len(subsections):
                    selected_subsections.append(subsections[idx])
            
            # Fill remaining if needed
            if len(selected_subsections) < 5:
                used_content = set()
                for subsection in selected_subsections:
                    used_content.add(subsection['refined_text'][:100])
                
                for subsection in subsections:
                    content_key = subsection['refined_text'][:100]
                    if content_key not in used_content and len(selected_subsections) < 5:
                        selected_subsections.append(subsection)
                        used_content.add(content_key)
            
            print(f"      ✅ Model selected {len(selected_subsections)} subsections from all PDFs")
            return selected_subsections[:5]
            
        except Exception as e:
            print(f"      ⚠️  Error in model selection: {e}")
            # Fallback: return first 5 subsections
            return subsections[:5]
    
    def _batch_model_selection_sections(self, sections: List[Dict], section_data: List[str], persona: str, task: str) -> List[Dict]:
        """
        Batch processing for large section collections
        """
        batch_size = 30
        all_candidates = []
        
        # Process in batches to get candidates
        for i in range(0, len(sections), batch_size):
            batch_sections = sections[i:i+batch_size]
            batch_data = section_data[i:i+batch_size]
            
            # Get top 3 from this batch
            batch_selected = self._direct_model_selection_sections(batch_sections, batch_data, persona, task)
            all_candidates.extend(batch_selected[:3])  # Top 3 from each batch
        
        # Final selection from all candidates
        if len(all_candidates) <= 15:
            # Small enough for final round
            candidate_data = []
            for i, section in enumerate(all_candidates, 1):
                title = section['section_title']
                doc = section['document']
                content = section.get('section_content', section.get('refined_text', ''))
                if content and len(content.strip()) > 20:
                    content_preview = content[:400].replace('\n', ' ').strip()
                    candidate_data.append(f"{i}. TITLE: {title}\nDOCUMENT: {doc}\nCONTENT: {content_preview}...")
                else:
                    candidate_data.append(f"{i}. TITLE: {title}\nDOCUMENT: {doc}\nCONTENT: [Title only]")
            
            return self._direct_model_selection_sections(all_candidates, candidate_data, persona, task)
        else:
            # Return top candidates
            for i, section in enumerate(all_candidates[:5]):
                section['importance_rank'] = i + 1
            return all_candidates[:5]
    
    def _batch_model_selection_subsections(self, subsections: List[Dict], subsection_data: List[str], persona: str, task: str) -> List[Dict]:
        """
        Batch processing for large subsection collections
        """
        batch_size = 25
        all_candidates = []
        
        # Process in batches to get candidates
        for i in range(0, len(subsections), batch_size):
            batch_subsections = subsections[i:i+batch_size]
            batch_data = subsection_data[i:i+batch_size]
            
            # Get top 3 from this batch
            batch_selected = self._direct_model_selection_subsections(batch_subsections, batch_data, persona, task)
            all_candidates.extend(batch_selected[:3])  # Top 3 from each batch
        
        # Final selection from all candidates
        if len(all_candidates) <= 12:
            # Small enough for final round
            candidate_data = []
            for i, subsection in enumerate(all_candidates, 1):
                doc = subsection['document']
                content = subsection['refined_text']
                content_preview = content[:500].replace('\n', ' ').strip()
                candidate_data.append(f"{i}. DOCUMENT: {doc}\nCONTENT: {content_preview}...")
            
            return self._direct_model_selection_subsections(all_candidates, candidate_data, persona, task)
        else:
            # Return top candidates
            return all_candidates[:5]
    
    def _simple_llm_selection(self, summaries: List[str], items: List[Dict], persona: str, task: str, content_type: str) -> List[int]:
        """
        Simple, fast LLM-based selection - select most relevant regardless of source
        """
        # Use more items for better selection (but still limit for speed)
        max_evaluate = min(len(summaries), 40)  # Increased from default for better selection
        items_to_evaluate = summaries[:max_evaluate]
        
        # Create simple numbered list
        numbered_items = []
        for i, summary in enumerate(items_to_evaluate, 1):
            numbered_items.append(f"{i}. {summary}")
        
        # Enhanced prompt emphasizing relevance over diversity
        prompt = f"""You are a {persona} working on: {task}

INSTRUCTION: Select the TOP {self.target_selections} MOST RELEVANT {content_type} from the {len(items_to_evaluate)} options below.

CRITERIA:
1. HIGHEST relevance to: "{task}"
2. Most useful for a {persona}
3. Practical, actionable information
4. Direct applicability to your work

OPTIONS:
{chr(10).join(numbered_items)}

RESPONSE FORMAT: Return EXACTLY {self.target_selections} numbers separated by commas (e.g., "1, 3, 7, 12, 15")

YOUR SELECTION:"""

        try:
            response = self.llm.invoke(prompt)
            numbers = re.findall(r'\d+', response.content.strip())
            
            # Convert to indices
            selected_indices = []
            for num_str in numbers[:self.target_selections]:
                idx = int(num_str) - 1  # Convert to 0-based
                if 0 <= idx < len(items_to_evaluate):
                    selected_indices.append(idx)
            
            # Simple fallback if not enough selections
            while len(selected_indices) < self.target_selections and len(selected_indices) < len(summaries):
                for i in range(len(summaries)):
                    if i not in selected_indices:
                        selected_indices.append(i)
                        break
            
            return selected_indices[:self.target_selections]
            
        except Exception as e:
            print(f"      ⚠️  Error in LLM selection: {e}")
            # Simple fallback - first N items
            return list(range(min(self.target_selections, len(summaries))))
    
    def _balanced_llm_selection(self, summaries: List[str], items: List[Dict], persona: str, task: str, content_type: str) -> List[int]:
        """
        Balanced selection that ensures representation from different documents
        """
        # Group items by document
        doc_groups = {}
        for i, item in enumerate(items):
            doc = item['document']
            if doc not in doc_groups:
                doc_groups[doc] = []
            doc_groups[doc].append(i)
        
        print(f"      📚 Found {len(doc_groups)} different documents")
        
        # Sample items from each document to ensure balanced representation
        balanced_indices = []
        items_per_doc = max(2, 60 // len(doc_groups))  # At least 2 items per doc, max 60 total
        
        for doc, indices in doc_groups.items():
            # Take top items from this document (up to items_per_doc)
            doc_sample = indices[:items_per_doc]
            balanced_indices.extend(doc_sample)
        
        # Limit total items for LLM evaluation
        balanced_indices = balanced_indices[:60]
        balanced_summaries = [summaries[i] for i in balanced_indices]
        
        print(f"      🎯 Evaluating {len(balanced_summaries)} items from {len(doc_groups)} documents")
        
        # Create numbered list
        numbered_items = []
        for i, summary in enumerate(balanced_summaries, 1):
            numbered_items.append(f"{i}. {summary}")
        
        # Enhanced prompt emphasizing relevance and document diversity
        prompt = f"""Role: {persona}
Task: {task}

Select the TOP {self.target_selections} MOST RELEVANT {content_type} from the {len(balanced_summaries)} options below.

IMPORTANT: These options come from different documents. Select the most useful content for your task, ensuring you consider options from various sources.

{chr(10).join(numbered_items)}

SELECTION CRITERIA (prioritize in this order):
1. HIGHEST relevance to: "{task}"
2. Most useful for a {persona}  
3. Practical, actionable information
4. Consider different document sources for comprehensive coverage

Return EXACTLY {self.target_selections} numbers separated by commas (e.g., "1, 7, 15, 23, 35"):"""

        try:
            response = self.llm.invoke(prompt)
            numbers = re.findall(r'\d+', response.content.strip())
            
            # Convert back to original indices
            selected_indices = []
            for num_str in numbers[:self.target_selections]:
                local_idx = int(num_str) - 1  # Convert to 0-based
                if 0 <= local_idx < len(balanced_indices):
                    original_idx = balanced_indices[local_idx]
                    selected_indices.append(original_idx)
            
            # Ensure we have enough selections with fallback
            if len(selected_indices) < self.target_selections:
                # Add missing selections from different documents
                used_docs = set()
                for idx in selected_indices:
                    used_docs.add(items[idx]['document'])
                
                # Try to add one from each unused document
                for doc, indices in doc_groups.items():
                    if doc not in used_docs and len(selected_indices) < self.target_selections:
                        if indices and indices[0] not in selected_indices:
                            selected_indices.append(indices[0])
                            used_docs.add(doc)
                
                # Fill remaining with best available
                while len(selected_indices) < self.target_selections and len(selected_indices) < len(items):
                    for idx in balanced_indices:
                        if idx not in selected_indices:
                            selected_indices.append(idx)
                            break
            
            print(f"      ✅ Selected {len(selected_indices)} items from multiple documents")
            return selected_indices[:self.target_selections]
            
        except Exception as e:
            print(f"      ⚠️  Error in balanced selection: {e}")
            # Fallback: one from each document first
            fallback_indices = []
            for doc, indices in doc_groups.items():
                if indices and len(fallback_indices) < self.target_selections:
                    fallback_indices.append(indices[0])
            
            # Fill remaining if needed
            while len(fallback_indices) < self.target_selections:
                for doc, indices in doc_groups.items():
                    for idx in indices[1:]:  # Skip first one already added
                        if idx not in fallback_indices and len(fallback_indices) < self.target_selections:
                            fallback_indices.append(idx)
            
            return fallback_indices[:self.target_selections]
    
    def _global_llm_selection(self, summaries: List[str], items: List[Dict], persona: str, task: str, content_type: str) -> List[int]:
        """
        Global selection that compares ALL content to find the absolute top 5 most relevant
        """
        print(f"      🌍 Evaluating ALL {len(summaries)} {content_type} globally for top {self.target_selections}")
        
        # For efficiency, if we have too many items, we'll do it in batches
        if len(summaries) <= 100:
            # Small enough - evaluate all at once
            return self._evaluate_all_content(summaries, items, persona, task, content_type)
        else:
            # Large collection - use batch scoring approach
            return self._batch_global_evaluation(summaries, items, persona, task, content_type)
    
    def _evaluate_all_content(self, summaries: List[str], items: List[Dict], persona: str, task: str, content_type: str) -> List[int]:
        """
        Evaluate all content at once to find global top 5
        """
        # Create numbered list of ALL items
        numbered_items = []
        for i, summary in enumerate(summaries, 1):
            numbered_items.append(f"{i}. {summary}")
        
        # Global comparison prompt
        prompt = f"""Role: {persona}
Task: {task}

You must compare ALL {len(summaries)} {content_type} below and select the TOP {self.target_selections} MOST RELEVANT ones for your specific task.

CRITICAL: Compare every option and choose only the {self.target_selections} most useful for: "{task}"

{chr(10).join(numbered_items)}

EVALUATION PROCESS:
1. Read through ALL options carefully
2. Identify which content is most directly useful for your task
3. Consider practical applicability for a {persona}
4. Rank by relevance - ignore document source, focus only on content value
5. Select the {self.target_selections} highest-scoring items

Return EXACTLY {self.target_selections} numbers separated by commas (most relevant first):"""

        try:
            response = self.llm.invoke(prompt)
            numbers = re.findall(r'\d+', response.content.strip())
            
            # Convert to indices
            selected_indices = []
            for num_str in numbers[:self.target_selections]:
                idx = int(num_str) - 1  # Convert to 0-based
                if 0 <= idx < len(summaries):
                    selected_indices.append(idx)
            
            # Fill remaining slots if needed
            while len(selected_indices) < self.target_selections and len(selected_indices) < len(summaries):
                for i in range(len(summaries)):
                    if i not in selected_indices:
                        selected_indices.append(i)
                        break
            
            print(f"      ✅ Global selection complete: {len(selected_indices)} most relevant items")
            return selected_indices[:self.target_selections]
            
        except Exception as e:
            print(f"      ⚠️  Error in global selection: {e}")
            # Fallback to first N items
            return list(range(min(self.target_selections, len(summaries))))
    
    def _batch_global_evaluation(self, summaries: List[str], items: List[Dict], persona: str, task: str, content_type: str) -> List[int]:
        """
        For large collections, use batch evaluation to find global top 5
        """
        batch_size = 50
        all_candidates = []
        
        # Process in batches to find top candidates from each batch
        for i in range(0, len(summaries), batch_size):
            batch_summaries = summaries[i:i+batch_size]
            batch_items = items[i:i+batch_size]
            
            # Get top candidates from this batch
            batch_top = self._evaluate_all_content(batch_summaries, batch_items, persona, task, content_type)
            
            # Convert batch indices to global indices
            global_indices = [i + idx for idx in batch_top]
            all_candidates.extend(global_indices)
        
        # Now evaluate all candidates globally
        if len(all_candidates) <= 20:
            # Small enough for final comparison
            candidate_summaries = [summaries[idx] for idx in all_candidates]
            final_selection = self._evaluate_all_content(candidate_summaries, [items[idx] for idx in all_candidates], persona, task, content_type)
            return [all_candidates[idx] for idx in final_selection]
        else:
            # Take best candidates proportionally
            return all_candidates[:self.target_selections]
    
    def _fast_top_selection(self, summaries: List[str], items: List[Dict], persona: str, task: str, content_type: str) -> List[int]:
        """
        Fast selection that finds top 5 most relevant without TLE risk
        Uses strategic sampling + multiple rounds for efficiency
        """
        print(f"      ⚡ Fast evaluation of {len(summaries)} {content_type} for top {self.target_selections}")
        
        if len(summaries) <= 30:
            # Small enough - evaluate all directly
            return self._evaluate_all_content(summaries, items, persona, task, content_type)
        
        # For large collections: Multi-round tournament approach
        return self._tournament_selection(summaries, items, persona, task, content_type)
    
    def _tournament_selection(self, summaries: List[str], items: List[Dict], persona: str, task: str, content_type: str) -> List[int]:
        """
        Tournament-style selection for fast but accurate top-5 finding
        """
        # Round 1: Sample strategically from all documents
        doc_groups = {}
        for i, item in enumerate(items):
            doc = item['document']
            if doc not in doc_groups:
                doc_groups[doc] = []
            doc_groups[doc].append(i)
        
        # Take top 2-3 candidates from each document (ensures diversity + quality)
        candidates = []
        max_per_doc = max(2, min(3, 30 // len(doc_groups)))
        
        for doc, indices in doc_groups.items():
            # Take first few from each doc (already somewhat filtered by extraction)
            doc_candidates = indices[:max_per_doc]
            candidates.extend(doc_candidates)
        
        # Limit to manageable size for LLM
        candidates = candidates[:30]
        candidate_summaries = [summaries[i] for i in candidates]
        
        print(f"      🏆 Tournament round: {len(candidate_summaries)} candidates from {len(doc_groups)} documents")
        
        # Evaluate candidates with focused prompt
        numbered_items = []
        for i, summary in enumerate(candidate_summaries, 1):
            numbered_items.append(f"{i}. {summary}")
        
        prompt = f"""You are a {persona} working on: {task}

INSTRUCTION: Select the TOP {self.target_selections} most relevant {content_type} from these {len(candidate_summaries)} pre-filtered options.

FOCUS: Choose the {self.target_selections} most useful for: "{task}"

OPTIONS:
{chr(10).join(numbered_items)}

RESPONSE FORMAT: Return exactly {self.target_selections} numbers (e.g., "1, 5, 12, 18, 25")

YOUR SELECTION:"""

        try:
            response = self.llm.invoke(prompt)
            numbers = re.findall(r'\d+', response.content.strip())
            
            # Convert back to original indices
            selected_indices = []
            for num_str in numbers[:self.target_selections]:
                local_idx = int(num_str) - 1
                if 0 <= local_idx < len(candidates):
                    original_idx = candidates[local_idx]
                    selected_indices.append(original_idx)
            
            # Fill remaining if needed
            while len(selected_indices) < self.target_selections and len(selected_indices) < len(candidates):
                for idx in candidates:
                    if idx not in selected_indices:
                        selected_indices.append(idx)
                        break
            
            print(f"      ✅ Tournament complete: {len(selected_indices)} winners selected")
            return selected_indices[:self.target_selections]
            
        except Exception as e:
            print(f"      ⚠️  Error in tournament: {e}")
            # Fast fallback - one from each doc + fill remaining
            fallback = []
            for doc, indices in doc_groups.items():
                if indices and len(fallback) < self.target_selections:
                    fallback.append(indices[0])
            
            # Fill remaining
            while len(fallback) < self.target_selections:
                for idx in candidates:
                    if idx not in fallback and len(fallback) < self.target_selections:
                        fallback.append(idx)
                        break
            
            return fallback[:self.target_selections]
    
    def enhance_section_scores_with_subsections(self, sections: List[Dict], subsections: List[Dict]) -> List[Dict]:
        """
        Simple enhancement of section scores based on subsections
        """
        # Group subsections by document
        subsection_by_doc = {}
        for subsection in subsections:
            doc = subsection['document']
            if doc not in subsection_by_doc:
                subsection_by_doc[doc] = []
            subsection_by_doc[doc].append(subsection)
        
        enhanced_sections = []
        for section in sections:
            section_copy = section.copy()
            doc = section['document']
            
            # Find related subsections in same document
            if doc in subsection_by_doc:
                related_subsections = subsection_by_doc[doc]
                if related_subsections:
                    # Simple boost based on number of related subsections
                    avg_subsection_score = sum(sub.get('relevance_score', 50) for sub in related_subsections) / len(related_subsections)
                    boost = min(len(related_subsections) * 2, 10)  # Max 10 point boost
                    
                    original_score = section_copy.get('relevance_score', 50)
                    enhanced_score = (original_score * 0.8) + (avg_subsection_score * 0.2) + boost
                    section_copy['relevance_score'] = min(enhanced_score, 100.0)
                    section_copy['related_subsections_count'] = len(related_subsections)
            
            enhanced_sections.append(section_copy)
        
        return enhanced_sections
    
    def _remove_duplicate_sections(self, sections: List[Dict]) -> List[Dict]:
        """
        Remove duplicate sections based on title and document with enhanced logic
        """
        seen = set()
        unique_sections = []
        
        for section in sections:
            # Create a key from title + document + page (more specific)
            title = section['section_title'].strip().lower()
            doc = section['document']
            page = section.get('page_number', 1)
            
            # Primary key: exact title + document + page
            primary_key = f"{title}|{doc}|{page}"
            # Secondary key: just title + document (for cases where page doesn't matter)
            secondary_key = f"{title}|{doc}"
            
            # Also check for very similar titles (first N chars from config)
            title_prefix = title[:self.dedup_title_prefix_chars]
            prefix_key = f"{title_prefix}|{doc}"
            
            if primary_key not in seen and secondary_key not in seen and prefix_key not in seen:
                seen.add(primary_key)
                seen.add(secondary_key)
                seen.add(prefix_key)
                unique_sections.append(section)
        
        print(f"      🔍 Section deduplication: {len(sections)} → {len(unique_sections)}")
        return unique_sections
    
    def _remove_duplicate_subsections(self, subsections: List[Dict]) -> List[Dict]:
        """
        Remove duplicate subsections based on content similarity and document+title combination
        """
        unique_subsections = []
        seen_content = set()
        seen_doc_title = set()
        
        for subsection in subsections:
            content = subsection['refined_text']
            doc = subsection['document']
            
            # Use configurable chars of content as primary deduplication key
            content_key = content[:self.dedup_content_chars].strip().lower()
            # Also check for doc + content start combination
            doc_content_key = f"{doc}|{content[:self.dedup_content_start_chars].strip().lower()}"
            
            # Skip if content is too short to be meaningful
            if len(content_key) < self.min_meaningful_content_chars:
                continue
                
            # Check for exact content match or very similar content from same doc
            is_duplicate = False
            for seen_key in seen_content:
                # If 80% of shorter content matches, consider duplicate
                shorter_len = min(len(content_key), len(seen_key))
                if shorter_len > 0:
                    overlap = sum(1 for i in range(shorter_len) if i < len(content_key) and i < len(seen_key) and content_key[i] == seen_key[i])
                    similarity = overlap / shorter_len
                    if similarity > 0.8:  # 80% similar = duplicate
                        is_duplicate = True
                        break
            
            if not is_duplicate and content_key not in seen_content and doc_content_key not in seen_doc_title:
                seen_content.add(content_key)
                seen_doc_title.add(doc_content_key)
                unique_subsections.append(subsection)
        
        print(f"      🔍 Subsection deduplication: {len(subsections)} → {len(unique_subsections)}")
        return unique_subsections

    def _final_deduplication_sections(self, sections: List[Dict]) -> List[Dict]:
        """
        Final aggressive deduplication for sections before output
        """
        if not sections:
            return sections
            
        unique_sections = []
        seen_combinations = set()
        
        for section in sections:
            # Create composite key for exact matching
            title = section['section_title'].strip().lower()
            doc = section['document']
            key = f"{title}|{doc}"
            
            if key not in seen_combinations:
                seen_combinations.add(key)
                unique_sections.append(section)
        
        # Re-rank the remaining sections
        for i, section in enumerate(unique_sections):
            section['importance_rank'] = i + 1
            
        print(f"      🔄 Final section cleanup: {len(sections)} → {len(unique_sections)}")
        return unique_sections
    
    def _final_deduplication_subsections(self, subsections: List[Dict]) -> List[Dict]:
        """
        Final aggressive deduplication for subsections before output
        """
        if not subsections:
            return subsections
            
        unique_subsections = []
        seen_content = set()
        
        for subsection in subsections:
            # Use first 100 characters as exact duplicate key
            content_key = subsection['refined_text'][:100].strip().lower()
            
            if content_key not in seen_content and len(content_key) > 20:
                seen_content.add(content_key)
                unique_subsections.append(subsection)
        
        print(f"      🔄 Final subsection cleanup: {len(subsections)} → {len(unique_subsections)}")
        return unique_subsections
