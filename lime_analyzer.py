import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for threading compatibility
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
from openai import OpenAI # Import the new OpenAI client
import httpx # Import httpx
from typing import List, Dict, Any, Optional
import base64
from io import BytesIO
from dotenv import load_dotenv
from skimage.segmentation import mark_boundaries
from sentence_transformers import SentenceTransformer, util

class LIMEAnalyzer:
    def __init__(
        self,
        mode: str = "auto",
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        verbose: bool = True,
        sim_model_name_or_path: str = "all-MiniLM-L6-v2",
        sim_model_local_dir: Optional[str] = None,
        hf_offline: Optional[bool] = None,
        shared_sim_model: Optional[Any] = None,  # Accept pre-loaded model
    ):
        """
        Initialize the LIME analyzer for visual illusions.
        
        Args:
            model_name: Name of the GPT-4 Vision model to use
            mode: 'auto' to use API, 'manual' to enter LLM answers manually
        """
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key and mode == "auto":
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.mode = mode
        self.model_name = model_name
        self.verbose = verbose
        self.openai_client = None # Will be initialized if mode is 'auto'

        # Set OpenAI API key and initialize client
        if mode == "auto":
            # Create an httpx client without proxies to prevent TypeError if environment proxies are detected.
            custom_httpx_client = httpx.Client()
            self.openai_client = OpenAI(api_key=api_key, http_client=custom_httpx_client)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # Initialize LIME explainer
        self.explainer = lime_image.LimeImageExplainer()
        
        # Initialize segmentation algorithm with quickshift
        # kernel_size: Size of the Gaussian kernel for smoothing
        # max_dist: Maximum distance between points in the same segment
        # ratio: Balance between color and space distances
        self.segmenter = SegmentationAlgorithm(
            'quickshift',
            kernel_size=4,    # Controls the size of the smoothing kernel
            max_dist=200,     # Maximum distance between points in the same segment
            ratio=0.2         # Balance between color and space distances
        )

        # Configure semantic similarity model (with offline/local fallback)
        if sim_model_local_dir is None:
            sim_model_local_dir = os.getenv("SIM_MODEL_LOCAL_DIR")
        if hf_offline is None:
            hf_offline = os.getenv("HF_OFFLINE", "").lower() in ("1", "true", "yes")

        self.sim_model = None
        last_error: Optional[Exception] = None

        # Use shared model if provided
        if shared_sim_model is not None:
            print(f"[INFO] Using shared sentence-transformers model")
            self.sim_model = shared_sim_model
        else:
            # Attempt to load from local directory first
            if sim_model_local_dir:
                local_path = Path(sim_model_local_dir)
                if local_path.is_dir():
                    print(f"[DEBUG] hf_offline status before local load attempt: {hf_offline}")
                    try:
                        print(f"[INFO] Attempting to load sentence-transformers model from local directory: {local_path}")
                        self.sim_model = SentenceTransformer(str(local_path), local_files_only=True)
                    except Exception as e:
                        last_error = e
                        self.sim_model = None
                        print(f"[ERROR] Failed to load local sentence-transformers model from {local_path}: {e}")
                else:
                    last_error = FileNotFoundError(f"SIM_MODEL_LOCAL_DIR points to a non-existent directory: {local_path}")

            # If local load failed or was not specified, and not in offline mode, try loading online
            if self.sim_model is None and not hf_offline:
                print(f"[DEBUG] hf_offline status before online load attempt: {hf_offline}")
                try:
                    print(f"[INFO] Attempting to load sentence-transformers model online: {sim_model_name_or_path}")
                    self.sim_model = SentenceTransformer(sim_model_name_or_path)
                except Exception as e:
                    last_error = e
                    self.sim_model = None
                    print(f"[ERROR] Failed to load online sentence-transformers model {sim_model_name_or_path}: {e}")

            # If still not available, raise a clear error with guidance
            if self.sim_model is None:
                hint = (
                    "Failed to load sentence-transformers model. "
                    "Ensure SIM_MODEL_LOCAL_DIR is correctly set if you downloaded it, or check internet connectivity. "
                    "Example to pre-download while online:\n"
                    "  python -c \"from sentence_transformers import SentenceTransformer; "
                    "SentenceTransformer('all-MiniLM-L6-v2').save('models/all-MiniLM-L6-v2')\"\n"
                    "Then run with SIM_MODEL_LOCAL_DIR=models/all-MiniLM-L6-v2."
                )
                raise RuntimeError(hint) from last_error

        # Move sentence transformer model to device
        # Explicitly set to CPU to avoid persistent 'meta tensor' errors on some setups.
        self.device = torch.device('cpu') # Force CPU for stability
        print(f"[INFO] Forcing sentence-transformers model to use CPU for stability.")
        self.sim_model = self.sim_model.to(self.device)
        print(f"[INFO] Moved sentence-transformers model to {self.device}")
        
        self.llm_responses_for_lime = [] # To store LLM responses during LIME runs

    def _encode_image(self, image: np.ndarray) -> str:
        """
        Encode image to base64 for GPT-4 Vision API.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Base64 encoded image string
        """
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image.astype('uint8'))
        
        # Convert to base64
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _get_llm_prediction(self, image: np.ndarray) -> str:
        """
        Get GPT-4 Vision prediction for an image.
        
        Args:
            image: Input image as numpy array
        Returns:
            LLM response as a string
        """
        base64_image = self._encode_image(image)
        prompt = """Analyze this visual illusion. Structure your response as follows:
        [DESCRIPTION]
        1. What do you see in the image?
        [ANALYSIS]
        2. What transformations or changes are present?
        3. How do different views affect what you see?
        4. What elements make this an effective illusion?
        
        Provide a detailed analysis for each section."""
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "low" # Specify detail level if needed
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        analysis = response.choices[0].message.content
        return analysis

    def predict_fn(self, images: np.ndarray) -> np.ndarray:
        """
        Prediction function for LIME.
        Args:
            images: Batch of images to predict on
        Returns:
            2D numpy array of semantic similarity scores (shape: [num_samples, 1])
        """
        predictions = []
        for idx, image in enumerate(images):
            if self.mode == "manual":
                plt.figure()
                plt.imshow(image.astype(np.uint8))
                plt.title(f"Perturbed Image {idx+1}")
                plt.axis('off')
                plt.show()
                print("Paste the LLM's answer for this image (press Enter when done):")
                answer = input()
                similarity = self._semantic_similarity(self.current_prompt, answer)
                predictions.append([similarity])
            else:
                llm_response = self._get_llm_prediction(image)
                similarity = self._semantic_similarity(self.current_prompt, llm_response)
                predictions.append([similarity])
                # Store LLM response for potential later use
                self.llm_responses_for_lime.append(llm_response)
        return np.array(predictions)

    def _create_perturbation_examples(self, image: np.ndarray, num_samples: int = 5) -> List[np.ndarray]:
        """
        Create and visualize perturbed versions of the image.
        
        Args:
            image: Original image as numpy array
            num_samples: Number of perturbed samples to generate
            
        Returns:
            List of perturbed images
        """
        # Get image segments using the segmentation algorithm
        segments = self.segmenter(image)
        
        # Create perturbed versions
        perturbed_images = []
        for _ in range(num_samples):
            # Create a random binary mask for segments
            mask = np.random.randint(0, 2, size=len(np.unique(segments)))
            
            # Create perturbed image
            perturbed = image.copy()
            for segment_id in range(len(mask)):
                if mask[segment_id] == 0:  # If segment is masked
                    # Set the segment to gray (128)
                    perturbed[segments == segment_id] = 128
            
            perturbed_images.append(perturbed)
        
        return perturbed_images

    def visualize_perturbations(self, image_path: str, num_samples: int = 5, save_path: str = None):
        """
        Visualize perturbed versions of an image.
        
        Args:
            image_path: Path to the original image
            num_samples: Number of perturbed samples to generate
            save_path: Optional path to save the visualization
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Get perturbed versions
        perturbed_images = self._create_perturbation_examples(image_np, num_samples)
        
        # Create visualization
        plt.figure(figsize=(15, 3 * (num_samples + 1)))
        
        # Show original image
        plt.subplot(num_samples + 1, 1, 1)
        plt.imshow(image_np)
        plt.title('Original Image')
        plt.axis('off')
        
        # Show perturbed versions
        for i, perturbed in enumerate(perturbed_images, 2):
            plt.subplot(num_samples + 1, 1, i)
            plt.imshow(perturbed)
            plt.title(f'Perturbed Version {i-1}')
            plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()
        else:
            plt.show()

    def _semantic_similarity(self, text1, text2):
        emb1 = self.sim_model.encode(text1, convert_to_tensor=True, device=self.device)
        emb2 = self.sim_model.encode(text2, convert_to_tensor=True, device=self.device)
        return float(util.pytorch_cos_sim(emb1, emb2).item())

    def analyze_image(self, image_path: str, prompt: str, num_samples: int = 3, top_labels: int = 3) -> Dict[str, Any]:
        """
        Analyze a single image using LIME to understand LLM's interpretation.
        
        Args:
            image_path: Path to the image
            prompt: Prompt for semantic similarity
            num_samples: Number of LIME samples to generate
            top_labels: Number of top features to analyze
        Returns:
            Dictionary containing analysis results
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Store the prompt for use in predict_fn
        self.current_prompt = prompt
        
        # Get initial LLM analysis
        initial_llm_response = self._get_llm_prediction(image_np)
        
        # Clear previous LLM responses before starting new LIME explanation
        self.llm_responses_for_lime = []

        # Generate LIME explanation
        explanation = self.explainer.explain_instance(
            image_np,
            self.predict_fn,  # Uses LLM predictions for each perturbed image
            top_labels=top_labels,
            num_samples=num_samples,
            segmentation_fn=self.segmenter  # Uses quickshift algorithm for segmentation
        )
        
        # Retrieve all LLM responses that were stored during the LIME explanation process
        all_llm_perturbed_responses = list(self.llm_responses_for_lime)
        # Clear the list for the next run
        self.llm_responses_for_lime = []
        
        # Get importance map
        importance_map = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,  # Only show regions that positively influenced the prediction
            num_features=10,     # Show top 10 most important regions
            hide_rest=False      # Show the full image with highlighted regions
        )
        
        # Calculate confidence scores based on the importance map
        confidence_scores = self._calculate_confidence_scores(explanation)
        
        # Get LLM's interpretation of important regions
        important_regions = self._analyze_important_regions(image_np, explanation, prompt)
        
        return {
            'original_image': image_np,
            'prompt': prompt, # Added for visualization
            'importance_map': importance_map[0],  # The original image
            'mask': importance_map[1],           # The importance mask
            'explanation': explanation,
            'confidence_scores': confidence_scores,
            'top_regions': self._get_top_regions(explanation),
            'llm_analysis': {
                'initial_llm_response': initial_llm_response,
                'important_regions': important_regions,
                'perturbed_image_llm_responses': all_llm_perturbed_responses
            }
        }

    def _analyze_important_regions(self, image: np.ndarray, explanation: Any, prompt: str) -> List[Dict[str, Any]]:
        """
        Get LLM's interpretation of important regions identified by LIME.
        
        Args:
            image: Original image
            explanation: LIME explanation object
            prompt: Prompt for semantic similarity
            
        Returns:
            List of region analyses
        """
        region_analyses = []
        
        # Get top regions
        for label in explanation.top_labels:
            regions = explanation.local_exp[label]
            sorted_regions = sorted(regions, key=lambda x: abs(x[1]), reverse=True)[:3]
            
            for region_id, importance in sorted_regions:
                # Create a masked version of the image highlighting this region
                mask = np.zeros_like(image)
                mask[explanation.segments == region_id] = image[explanation.segments == region_id]
                
                # Get LLM's interpretation of this region
                full_llm_response, llm_description, _ = self._get_llm_prediction_with_response(mask)
                
                similarity = self._semantic_similarity(prompt, llm_description)
                
                region_analyses.append({
                    'region_id': int(region_id),
                    'importance': float(importance),
                    'llm_response': full_llm_response,
                    'semantic_score': similarity
                })
        
        return region_analyses

    def analyze_illusion(self, illusion_dir, num_samples=3, segmentation_method='quickshift', top_labels=3):
        """
        Analyze a complete illusion with multiple views.
        
        Args:
            illusion_dir: Directory containing the illusion images
            num_samples: Number of LIME samples
            segmentation_method: Method for image segmentation
            top_labels: Number of top features to analyze
            
        Returns:
            Dictionary containing analysis results for all views
        """
        results = {}
        illusion_dir = Path(illusion_dir)
        
        # Analyze each view
        for image_path in illusion_dir.glob('sample_*.png'):
            view_name = image_path.stem
            results[view_name] = self.analyze_image(
                str(image_path),
                prompt="A village and a horse are visible in the image.",
                num_samples=num_samples,
                top_labels=top_labels
            )
        
        # Add comparative analysis
        results['comparative'] = self._compare_views(results)
        
        return results

    def _calculate_confidence_scores(self, explanation):
        """
        Calculate confidence scores for the explanation.
        
        Args:
            explanation: LIME explanation object
            
        Returns:
            Dictionary of confidence scores
        """
        # Get local prediction scores from LIME
        local_pred = explanation.local_pred
        
        # Calculate confidence metrics based on:
        # 1. Maximum prediction confidence
        # 2. Feature importance confidence
        # 3. Overall confidence combining importance and prediction
        feature_weights = [abs(score[1]) for score in explanation.local_exp[explanation.top_labels[0]]]
        confidence = {
            'prediction_confidence': float(np.max(local_pred)),
            'feature_importance_confidence': float(np.mean(feature_weights)),
            'overall_confidence': float(np.mean(feature_weights) * np.max(local_pred))
        }
        
        return confidence

    def _get_top_regions(self, explanation):
        """
        Get the top influential regions from the explanation.
        
        Args:
            explanation: LIME explanation object
            
        Returns:
            List of top regions with their importance scores
        """
        top_regions = []
        for label in explanation.top_labels:
            regions = explanation.local_exp[label]
            # Sort regions by absolute importance
            sorted_regions = sorted(regions, key=lambda x: abs(x[1]), reverse=True)
            top_regions.append({
                'label': int(label),
                'regions': [(int(r[0]), float(r[1])) for r in sorted_regions[:5]]
            })
        return top_regions

    def _compare_views(self, results):
        """
        Compare different views of the same illusion.
        
        Args:
            results: Dictionary of analysis results for different views
            
        Returns:
            Dictionary containing comparative analysis
        """
        comparison = {
            'common_regions': self._find_common_regions(results),
            'view_similarity': self._calculate_view_similarity(results),
            'confidence_comparison': self._compare_confidence_scores(results)
        }
        return comparison

    def _find_common_regions(self, results):
        """
        Find regions that are important across different views.
        
        Args:
            results: Dictionary of analysis results
            
        Returns:
            List of common important regions
        """
        common_regions = set()
        for view_result in results.values():
            if 'top_regions' in view_result:
                for region in view_result['top_regions']:
                    common_regions.update(r[0] for r in region['regions'])
        return list(common_regions)

    def _calculate_view_similarity(self, results):
        """
        Calculate similarity between different views.
        
        Args:
            results: Dictionary of analysis results
            
        Returns:
            Dictionary of similarity scores
        """
        similarities = {}
        views = list(results.keys())
        for i in range(len(views)):
            for j in range(i + 1, len(views)):
                view1, view2 = views[i], views[j]
                similarity = self._compute_similarity(
                    results[view1]['importance_map'],
                    results[view2]['importance_map']
                )
                similarities[f"{view1}_{view2}"] = float(similarity)
        return similarities

    def _compute_similarity(self, map1, map2):
        """
        Compute similarity between two importance maps.
        
        Args:
            map1: First importance map
            map2: Second importance map
            
        Returns:
            Similarity score
        """
        # Normalize maps
        map1_norm = (map1 - map1.min()) / (map1.max() - map1.min())
        map2_norm = (map2 - map2.min()) / (map2.max() - map2.min())
        
        # Compute cosine similarity
        similarity = np.sum(map1_norm * map2_norm) / (
            np.sqrt(np.sum(map1_norm ** 2)) * np.sqrt(np.sum(map2_norm ** 2))
        )
        return similarity

    def _compare_confidence_scores(self, results):
        """
        Compare confidence scores across different views.
        
        Args:
            results: Dictionary of analysis results
            
        Returns:
            Dictionary of confidence comparisons
        """
        confidence_comparison = {}
        for view, result in results.items():
            if 'confidence_scores' in result:
                confidence_comparison[view] = result['confidence_scores']
        return confidence_comparison

    def save_analysis(self, analysis_results, output_path):
        """
        Save analysis results to a JSON file.
        
        Args:
            analysis_results: Dictionary containing analysis results
            output_path: Path to save the results
        """
        # Remove non-serializable fields
        if 'explanation' in analysis_results:
            analysis_results = dict(analysis_results)  # Make a shallow copy
            analysis_results.pop('explanation')
        
        # Convert numpy arrays to lists
        serializable_results = self._make_serializable(analysis_results)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

    def load_analysis(self, input_path):
        """
        Load analysis results from a JSON file.
        
        Args:
            input_path: Path to the saved results
            
        Returns:
            Dictionary containing loaded analysis results
        """
        with open(input_path, 'r') as f:
            results = json.load(f)
        return results

    def _make_serializable(self, obj):
        """
        Convert numpy arrays and other non-serializable objects to serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            Serializable object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj

    def visualize_importance_map(self, results: Dict[str, Any], save_path: str = None):
        """
        Visualize the importance map and LLM's interpretation.
        
        Args:
            results: Analysis results from analyze_image
            save_path: Optional path to save the visualization
        """
        plt.figure(figsize=(15, 7))
        plt.subplots_adjust(bottom=0.2) # Adjust bottom to make space for xlabel
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(results['original_image'])
        plt.title('Original Image')
        # Add prompt under the original image
        plt.xlabel(results.get('prompt', 'N/A'), fontsize=10, wrap=True)
        plt.axis('off')
        
        # Importance map (original image with mask overlay)
        plt.subplot(1, 3, 2)
        temp_image = results['original_image'].copy()
        mask_indices = results['mask'] > 0  # Get indices where mask is active
        temp_image[mask_indices] = temp_image[mask_indices] * 0.5 + np.array([255, 0, 0]) * 0.5 # Blend with red
        plt.imshow(temp_image.astype(np.uint8))
        plt.title('Importance Map Overlay')
        plt.axis('off')
        
        # Mask showing important regions
        plt.subplot(1, 3, 3)
        plt.imshow(results['mask'], cmap='hot')
        plt.title('Important Regions')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()
        else:
            plt.show()

    def _visualize_segmentation(self, image: np.ndarray, save_path: str = None):
        """
        Visualize the image segmentation process.
        
        Args:
            image: Input image as numpy array
            save_path: Optional path to save the visualization
        """
        # Get segments using quickshift
        segments = self.segmenter(image)
        
        # Create visualization
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Segmentation boundaries
        plt.subplot(1, 3, 2)
        plt.imshow(mark_boundaries(image, segments))
        plt.title('Segmentation Boundaries')
        plt.axis('off')
        
        # Segmented regions
        plt.subplot(1, 3, 3)
        plt.imshow(segments, cmap='nipy_spectral')
        plt.title('Segmented Regions')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()
        else:
            plt.show()
        
        return segments

    def _get_llm_prediction_with_response(self, image: np.ndarray):
        """
        Gets the full LLM prediction and extracts the description for semantic analysis.
        """
        full_response = self._get_llm_prediction(image)
        
        # Extract the description part
        try:
            description = full_response.split("[DESCRIPTION]")[1].split("[ANALYSIS]")[0].strip()
        except IndexError:
            description = "" # Or handle this case as you see fit
        
        return full_response, description, None

def main():
    """
    Example usage of the LIMEAnalyzer.
    """
    # Initialize analyzer
    analyzer = LIMEAnalyzer()
    
    # Load and analyze an image
    image_path = "results/rotate_cw.village.horse/sample_1024.png"
    prompt = "A village and a horse are visible in the image."
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    # Visualize segmentation
    analyzer._visualize_segmentation(image_np, save_path='segmentation.png')
    
    # Analyze the image
    results = analyzer.analyze_image(image_path, prompt, num_samples=3)
    
    # Visualize perturbations
    analyzer.visualize_perturbations(image_path, num_samples=5, save_path='perturbations.png')
    
    # Visualize importance map
    analyzer.visualize_importance_map(results, save_path='importance_map.png')
    
    # Print Confidence Scores
    print("\n--- Confidence Scores ---")
    confidence = results.get('confidence_scores', {})
    print(f"Prediction Confidence: {confidence.get('prediction_confidence', 'N/A'):.3f}")
    print(f"Feature Importance Confidence: {confidence.get('feature_importance_confidence', 'N/A'):.3f}")
    print(f"Overall Confidence: {confidence.get('overall_confidence', 'N/A'):.3f}")
    
    # Print LLM's analysis of important regions
    print("\n--- LLM Analysis of Important Regions ---")
    for region in results['llm_analysis']['important_regions']:
        print(f"\nRegion {region['region_id']} (Importance: {region['importance']:.2f}):")
        print(f"Semantic similarity to prompt: {region['semantic_score']:.3f}")
        print(f"LLM response: {region['llm_response']}")

if __name__ == "__main__":
    main() 
