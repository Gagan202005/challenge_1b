#!/usr/bin/env python3
"""
Main RAG Processing Script
This handles the complete PDF processing pipeline with smart relevance scoring
"""

import os
import sys
import json
from pathlib import Path

# We need to add the src directory to our path so Python can find our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from langchain_ollama import ChatOllama
from src.pdf_processor import PDFProcessor
from src.section_extractor import SectionExtractor
from src.subsection_extractor import SubsectionExtractor
from src.output_formatter import OutputFormatter


def load_config(config_path: str = "config/challenge_config.json") -> dict:
    """Loads our main configuration file with all the settings"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Configuration loaded from {config_path}")
        return config
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)


def load_input_config(input_path: str) -> dict:
    """Loads the specific input configuration for each collection"""
    try:
        with open(input_path, 'r') as f:
            input_config = json.load(f)
        print(f"✅ Input configuration loaded from {input_path}")
        return input_config
    except Exception as e:
        print(f"❌ Error loading input config: {e}")
        sys.exit(1)


def initialize_llm(config: dict) -> ChatOllama:
    """Sets up our local Ollama language model with the right settings"""
    try:
        model_settings = config["model_settings"]
        llm = ChatOllama(
            model=model_settings["llm_model"],
            temperature=model_settings["temperature"],
            base_url="http://localhost:11434"
        )
        print(f"✅ LLM initialized: {model_settings['llm_model']}")
        return llm
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        sys.exit(1)


def main():
    """The main function that brings everything together and runs the whole process"""
    
    # Let's handle command line arguments so users can choose collections easily
    import argparse
    parser = argparse.ArgumentParser(description='Process PDF RAG with input configuration')
    parser.add_argument('--collection', default='Collection 1', 
                        help='Collection name to process (e.g., "Collection 1", "Collection 2", "Collection 3")')
    args = parser.parse_args()
    
    print("🚀 STARTING OLLAMA PDF RAG PROCESSING")
    print("=" * 50)
    
    # Build our file paths based on the collection the user wants
    collection_path = f"input/{args.collection}"
    input_config_path = os.path.join(collection_path, "challenge1b_input.json")
    
    # Load up all our configuration files
    input_config = load_input_config(input_config_path)
    
    # Load our base configuration for model settings
    base_config = load_config()
    
    # Merge everything together so we have all the settings we need
    config = {
        **base_config,
        "persona": input_config["persona"],
        "job_to_be_done": input_config["job_to_be_done"],
        "challenge_info": input_config["challenge_info"],
        "documents": input_config["documents"]
    }
    
    # Get our language model ready
    llm = initialize_llm(config)
    
    # Set up all our processing components with parallel processing
    pdf_directory = os.path.join(collection_path, "PDFs")
    processing_config = config.get('processing_settings', {})
    max_workers = processing_config.get('max_workers', 4)
    
    pdf_processor = PDFProcessor(pdf_directory, max_workers=max_workers, config=config)
    section_extractor = SectionExtractor(llm, {**config, 'max_workers': max_workers})
    subsection_extractor = SubsectionExtractor(llm, {**config, 'max_workers': max_workers})
    output_formatter = OutputFormatter()
    
    # STEP 1: Let's load all the PDF documents
    print(f"\n📄 STEP 1: LOADING PDF DOCUMENTS")
    print("-" * 30)
    print(f"   📁 Collection: {args.collection}")
    print(f"   📂 PDF Directory: {pdf_directory}")
    documents = pdf_processor.load_pdfs_from_input(config["documents"])
    
    if not documents:
        print("❌ No documents loaded. Please add PDF files to data/pdfs/")
        sys.exit(1)
    
    # Show the user what we found
    summary = pdf_processor.get_document_summary(documents)
    print(f"   📊 Summary: {summary['total_docs']} documents, {summary['total_pages']} pages")
    
    # STEP 2: Extract the main sections from all the content
    print(f"\n📋 STEP 2: EXTRACTING SECTIONS WITH RELEVANCE SCORING")
    print("-" * 50)
    # Get the persona and task from what the user specified in their input
    persona = config['persona']['role']
    task = config['job_to_be_done']['task']
    
    print(f"   👤 Persona: {persona}")
    print(f"   🎯 Task: {task}")
    
    sections = section_extractor.extract_sections(documents, persona, task)
    
    # STEP 3: Now let's break down each section into subsections
    print(f"\n🔍 STEP 3: EXTRACTING SUBSECTIONS WITH RELEVANCE SCORING")
    print("-" * 50)
    subsections = subsection_extractor.extract_subsections(documents, persona, task)
    
    # STEP 3.5: Make our section scores even better by looking at subsection quality
    print(f"\n⚡ STEP 3.5: ENHANCING SECTION SCORES WITH SUBSECTION ANALYSIS")
    print("-" * 60)
    from src.relevance_scorer import RelevanceScorer
    scorer_config = config.get('relevance_scorer_settings', {})
    scorer = RelevanceScorer(llm, scorer_config)
    enhanced_sections = scorer.enhance_section_scores_with_subsections(sections, subsections)
    
    # Sort everything by quality - best stuff first
    enhanced_sections.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    
    # Give each section a ranking and clean up temporary data
    for i, section in enumerate(enhanced_sections, 1):
        section['importance_rank'] = i
        # Remove temporary fields for final output
        section.pop('relevance_score', None)
        section.pop('related_subsections_count', None)
    
    print(f"✅ Section scores enhanced based on subsection relevance")
    
    # STEP 4: Format everything nicely for the output file
    print(f"\n📤 STEP 4: FORMATTING OUTPUT")
    print("-" * 30)
    
    document_names = summary['filenames']
    result = output_formatter.format_challenge1b(enhanced_sections, subsections, config, document_names)
    
    # Step 5: Save output
    print(f"\n💾 STEP 5: SAVING RESULTS")
    print("-" * 30)
    
    # Always save to outputs folder
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "challenge1b_output.json")
    
    output_formatter.save_output(result, output_path)
    
    # Print summary
    output_formatter.print_summary(result)
    
    print(f"\n🎉 PROCESSING COMPLETE!")
    print("=" * 50)
    print(f"✅ Results saved to: {output_path}")
    print(f"📁 Processed collection: {args.collection}")
    print(f"🆔 Challenge ID: {config['challenge_info']['challenge_id']}")
    print(f"🧪 Test case: {config['challenge_info']['test_case_name']}")
    

if __name__ == "__main__":
    main()
