import os
from dotenv import load_dotenv
from illusion_analyzer import IllusionAnalyzer
from llm_analyzer import LLMAnalyzer
from similarity_evaluator import SimilarityEvaluator
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("Please set your OPENAI_API_KEY environment variable")
    
    try:
        # Initialize analyzers
        logging.info("Initializing analyzers...")
        illusion_analyzer = IllusionAnalyzer()
        llm_analyzer = LLMAnalyzer()
        similarity_evaluator = SimilarityEvaluator()
        
        # Load and analyze an illusion
        logging.info("Loading illusion data...")
        illusion_data = illusion_analyzer.load_illusion_data('results/rotate_cw.village.horse')
        
        logging.info("Analyzing illusion...")
        analysis_results = illusion_analyzer.analyze_illusion(illusion_data)
        
        # Evaluate the analysis for 64x64 sample only
        logging.info("Evaluating analysis for 64x64 sample...")
        evaluation = similarity_evaluator.evaluate_analysis(
            analysis_results['llm_analysis']['64'],  # Changed from '1024' to '64'
            analysis_results['original_prompts']
        )
        
        # Analyze the reasoning process for 64x64 sample
        logging.info("Analyzing reasoning process for 64x64 sample...")
        reasoning_analysis = similarity_evaluator.analyze_reasoning_process(
            analysis_results['llm_analysis']['64']  # Changed from '1024' to '64'
        )
        
        # Save results
        logging.info("Saving results...")
        results = {
            'analysis_results': analysis_results,
            'evaluation': evaluation,
            'reasoning_analysis': reasoning_analysis
        }
        
        with open('analysis_output_64.json', 'w') as f:  # Changed filename to indicate 64x64 analysis
            json.dump(results, f, indent=2)
            
        logging.info("Analysis complete! Results saved to analysis_output_64.json")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 