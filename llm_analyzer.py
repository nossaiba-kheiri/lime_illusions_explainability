import os
import base64
from io import BytesIO
import torch
from PIL import Image
import openai
from typing import Dict, Any, List
import json
import logging
from datetime import datetime

class LLMAnalyzer:
    def __init__(self, api_key: str = None):
        """
        Initialize the LLM analyzer with OpenAI API
        
        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY environment variable)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        openai.api_key = self.api_key
        
    def _encode_image(self, image: torch.Tensor) -> str:
        """
        Convert image tensor to base64 string
        
        Args:
            image: Image tensor
            
        Returns:
            Base64 encoded string
        """
        # Convert tensor to PIL Image
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).astype('uint8')
        pil_image = Image.fromarray(image)
        
        # Convert to base64
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def analyze_image(self, image: torch.Tensor, views: List[str]) -> Dict[str, Any]:
        """
        Analyze an image using GPT-4 Vision
        
        Args:
            image: Image tensor
            views: List of view transformations
            
        Returns:
            Dictionary containing analysis results
        """
        # Encode image
        base64_image = self._encode_image(image)
        
        # Prepare system message
        system_message = """You are an expert at analyzing visual illusions and optical effects. 
        Your task is to:
        1. Describe what you see in the image
        2. Identify any transformations or illusions present
        3. Explain your reasoning process
        4. Provide confidence scores for your analysis
        
        Be detailed in your analysis and explain your thinking step by step."""
        
        # Prepare user message
        user_message = f"""Please analyze this image. The possible transformations are: {', '.join(views)}.
        
        Provide your analysis in the following format:
        1. Initial observation
        2. Transformation identification
        3. Detailed description of what you see
        4. Reasoning process
        5. Confidence scores
        
        Format your response as a JSON object with the following structure:
        {{
            "initial_observation": "string",
            "transformation_identified": "string",
            "description": "string",
            "reasoning_process": ["string"],
            "confidence_scores": {{
                "transformation_identification": float,
                "description_accuracy": float,
                "overall_confidence": float
            }}
        }}"""
        
        try:
            # Call GPT-4 Vision
            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=[
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_message},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            # Parse response
            analysis = json.loads(response.choices[0].message.content)
            
            # Log the analysis
            logging.info(f"Analysis completed: {json.dumps(analysis, indent=2)}")
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error in LLM analysis: {str(e)}")
            return {
                "error": str(e),
                "initial_observation": "Error occurred during analysis",
                "transformation_identified": "Unknown",
                "description": "Error occurred during analysis",
                "reasoning_process": ["Error occurred during analysis"],
                "confidence_scores": {
                    "transformation_identification": 0.0,
                    "description_accuracy": 0.0,
                    "overall_confidence": 0.0
                }
            }
    
    def evaluate_analysis(self, analysis: Dict[str, Any], original_prompts: List[str]) -> Dict[str, Any]:
        """
        Evaluate how well the LLM analysis matches the original prompts
        
        Args:
            analysis: Dictionary containing LLM analysis
            original_prompts: List of original prompts used to generate the illusion
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # TODO: Implement semantic similarity comparison
        # This would use a text similarity model to compare the LLM's description
        # with the original prompts
        
        return {
            "prompt_matching_score": 0.0,  # Placeholder
            "transformation_accuracy": analysis["confidence_scores"]["transformation_identification"],
            "description_accuracy": analysis["confidence_scores"]["description_accuracy"],
            "overall_confidence": analysis["confidence_scores"]["overall_confidence"]
        }

def main():
    # Example usage
    analyzer = LLMAnalyzer()
    
    # Load an image (placeholder)
    image = torch.rand(3, 224, 224)  # Example image tensor
    views = ["rotate_cw", "flip"]
    
    # Analyze image
    analysis = analyzer.analyze_image(image, views)
    
    # Evaluate analysis
    original_prompts = ["a snowy mountain village", "a horse"]
    evaluation = analyzer.evaluate_analysis(analysis, original_prompts)
    
    # Print results
    print("Analysis:", json.dumps(analysis, indent=2))
    print("Evaluation:", json.dumps(evaluation, indent=2))

if __name__ == "__main__":
    main() 