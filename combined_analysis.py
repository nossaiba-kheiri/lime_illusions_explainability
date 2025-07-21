import os
from dotenv import load_dotenv
from illusion_analyzer import IllusionAnalyzer
from llm_analyzer import LLMAnalyzer
from similarity_evaluator import SimilarityEvaluator
from lime_analyzer import LIMEAnalyzer
import json
import logging
from pathlib import Path

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
        lime_analyzer = LIMEAnalyzer()
        
        # Load and analyze an illusion
        logging.info("Loading illusion data...")
        illusion_data = illusion_analyzer.load_illusion_data('results/rotate_cw.village.horse')
        
        # LLM Analysis
        logging.info("Performing LLM analysis...")
        llm_analysis = illusion_analyzer.analyze_illusion(illusion_data)
        
        # Evaluate LLM analysis
        logging.info("Evaluating LLM analysis...")
        llm_evaluation = similarity_evaluator.evaluate_analysis(
            llm_analysis['llm_analysis']['64'],
            llm_analysis['original_prompts']
        )
        
        # Analyze LLM reasoning process
        logging.info("Analyzing LLM reasoning process...")
        llm_reasoning = similarity_evaluator.analyze_reasoning_process(
            llm_analysis['llm_analysis']['64']
        )
        
        # LIME Analysis
        logging.info("Performing LIME analysis...")
        lime_analysis = lime_analyzer.analyze_illusion('results/rotate_cw.village.horse')
        
        # Combine results
        results = {
            'llm_analysis': {
                'analysis': llm_analysis,
                'evaluation': llm_evaluation,
                'reasoning': llm_reasoning
            },
            'lime_analysis': lime_analysis
        }
        
        # Save results
        logging.info("Saving results...")
        output_dir = Path('analysis_output')
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / 'combined_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        logging.info("Analysis complete! Results saved to analysis_output/combined_analysis.json")
        
        # Print summary
        print("\nAnalysis Summary:")
        print("----------------")
        print("LLM Analysis:")
        print(f"- Prompt matching score: {llm_evaluation['prompt_matching_score']:.2f}")
        print(f"- Overall confidence: {llm_evaluation['overall_confidence']:.2f}")
        print("\nLIME Analysis:")
        print(f"- Number of important features: {len(lime_analysis['feature_importance'])}")
        print(f"- Visualization saved to: {lime_analysis['visualization_path']}")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 