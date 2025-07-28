"""
Output Formatting - Takes our results and packages them up nicely
This module makes sure everything looks good in the final JSON output
"""

import json
from datetime import datetime
from typing import List, Dict


class OutputFormatter:
    """Takes all the extracted content and formats it the way the challenge expects"""
    
    def __init__(self):
        pass
        
    def format_challenge1b(self, sections: List[Dict], subsections: List[Dict], 
                          config: Dict, document_names: List[str]) -> Dict:
        """
        Formats our results into the exact format the challenge wants
        
        Args:
            sections: All the main sections we found and ranked
            subsections: All the detailed subsections we extracted
            config: Our system configuration settings
            document_names: Names of all the PDF files we processed
            
        Returns:
            A nicely formatted dictionary ready to save as JSON
        """
        
        # Figure out how many sections and subsections to include
        max_sections = config.get('output_settings', {}).get('max_sections', 5)
        max_subsections = config.get('output_settings', {}).get('max_subsections', 5)
        
        # Make sure all document names have .pdf extension
        formatted_doc_names = []
        for name in document_names:
            if not name.endswith('.pdf'):
                name += '.pdf'
            formatted_doc_names.append(name)
        
        # Format our sections with proper PDF naming and clean titles
        formatted_sections = []
        for section in sections[:max_sections]:
            doc_name = section['document']
            if not doc_name.endswith('.pdf'):
                doc_name += '.pdf'
                
            formatted_sections.append({
                "document": doc_name,
                "section_title": section['section_title'],  # Exact title only
                "importance_rank": section['importance_rank'],
                "page_number": section['page_number']
            })
        
        # Format subsections - ensure .pdf extension and include FULL exact content
        formatted_subsections = []
        for subsection in subsections[:max_subsections]:
            doc_name = subsection['document']
            if not doc_name.endswith('.pdf'):
                doc_name += '.pdf'
                
            formatted_subsections.append({
                "document": doc_name,
                "refined_text": subsection['refined_text'],  # Full exact content of subsection
                "page_number": subsection['page_number']
            })
        
        # Create final result in exact Challenge1b format
        result = {
            "challenge_info": config.get('challenge_info', {}),
            "metadata": {
                "input_documents": list(set(formatted_doc_names)),
                "persona": config['persona']['role'],
                "job_to_be_done": config['job_to_be_done']['task'],
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": formatted_sections,
            "subsection_analysis": formatted_subsections
        }
        
        return result
    
    def save_output(self, result: Dict, output_path: str) -> None:
        """
        Save result to JSON file
        
        Args:
            result: Result dictionary to save
            output_path: Path to save the file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            print(f"✅ Output saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error saving output: {e}")
    
    def print_summary(self, result: Dict) -> None:
        """Print a summary of the results"""
        
        sections = result.get('extracted_sections', [])
        subsections = result.get('subsection_analysis', [])
        
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"   📋 Sections extracted: {len(sections)}")
        print(f"   🔍 Subsections extracted: {len(subsections)}")
        
        print(f"\n🏆 TOP SECTIONS:")
        for i, section in enumerate(sections[:3], 1):
            title = section['section_title'][:60] + "..." if len(section['section_title']) > 60 else section['section_title']
            print(f"   {i}. {title}")
            print(f"      📄 {section['document']} (Page {section['page_number']})")
        
        print(f"\n🔍 TOP SUBSECTIONS:")
        for i, sub in enumerate(subsections[:3], 1):
            preview = sub['refined_text'][:80].replace('\n', ' ') + "..."
            print(f"   {i}. {sub['document']} (Page {sub['page_number']})")
            print(f"      Preview: {preview}")
