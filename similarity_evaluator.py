import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

class SimilarityEvaluator:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the similarity evaluator with a sentence transformer model
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        
    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Get embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            Tensor of embeddings
        """
        # Tokenize texts
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        
        # Get embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Use mean pooling
            embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            
        return embeddings
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on model output
        
        Args:
            model_output: Output from the transformer model
            attention_mask: Attention mask for the input
            
        Returns:
            Mean pooled embeddings
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 0) / torch.clamp(input_mask_expanded.sum(0), min=1e-9)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        embeddings = self._get_embeddings([text1, text2])
        similarity = cosine_similarity(embeddings[0].numpy().reshape(1, -1), 
                                    embeddings[1].numpy().reshape(1, -1))[0][0]
        return float(similarity)
    
    def evaluate_analysis(self, analysis: Dict[str, Any], original_prompts: List[str]) -> Dict[str, Any]:
        """
        Evaluate how well the LLM analysis matches the original prompts
        
        Args:
            analysis: Dictionary containing LLM analysis
            original_prompts: List of original prompts used to generate the illusion
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Get the description from the analysis
        description = analysis['description']
        
        # Compute similarities with each original prompt
        similarities = []
        for prompt in original_prompts:
            similarity = self.compute_similarity(description, prompt)
            similarities.append(similarity)
        
        # Get the maximum similarity as the matching score
        prompt_matching_score = max(similarities)
        
        # Log the evaluation
        logging.info(f"Prompt matching score: {prompt_matching_score}")
        logging.info(f"Original prompts: {original_prompts}")
        logging.info(f"LLM description: {description}")
        
        return {
            "prompt_matching_score": prompt_matching_score,
            "similarities": dict(zip(original_prompts, similarities)),
            "transformation_accuracy": analysis["confidence_scores"]["transformation_identification"],
            "description_accuracy": analysis["confidence_scores"]["description_accuracy"],
            "overall_confidence": analysis["confidence_scores"]["overall_confidence"]
        }
    
    def analyze_reasoning_process(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the LLM's reasoning process
        
        Args:
            analysis: Dictionary containing LLM analysis
            
        Returns:
            Dictionary containing reasoning analysis
        """
        reasoning_steps = analysis["reasoning_process"]
        
        # Analyze each reasoning step
        step_analysis = []
        for i, step in enumerate(reasoning_steps):
            # Compute similarity with the final description
            similarity = self.compute_similarity(step, analysis["description"])
            
            step_analysis.append({
                "step_number": i + 1,
                "content": step,
                "relevance_to_final_description": similarity
            })
        
        return {
            "number_of_steps": len(reasoning_steps),
            "step_analysis": step_analysis,
            "average_step_relevance": np.mean([step["relevance_to_final_description"] for step in step_analysis])
        }

def main():
    # Example usage
    evaluator = SimilarityEvaluator()
    
    # Example analysis and prompts
    analysis = {
        "description": "I see a snowy mountain village that transforms into a horse when rotated",
        "reasoning_process": [
            "First, I notice the overall composition of the image",
            "Then, I identify the mountain village elements",
            "Upon rotation, I see the horse shape emerge",
            "The transformation appears to be a 90-degree rotation"
        ],
        "confidence_scores": {
            "transformation_identification": 0.9,
            "description_accuracy": 0.85,
            "overall_confidence": 0.88
        }
    }
    
    original_prompts = ["a snowy mountain village", "a horse"]
    
    # Evaluate analysis
    evaluation = evaluator.evaluate_analysis(analysis, original_prompts)
    
    # Analyze reasoning process
    reasoning_analysis = evaluator.analyze_reasoning_process(analysis)
    
    # Print results
    print("Evaluation:", evaluation)
    print("Reasoning Analysis:", reasoning_analysis)

if __name__ == "__main__":
    main() 