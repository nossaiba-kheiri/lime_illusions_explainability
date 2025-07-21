# Visual Illusion Analyzer

This project combines LLM analysis and LIME (Local Interpretable Model-agnostic Explanations) to analyze visual illusions and multi-view optical illusions.

## Prerequisites

- Python 3.10 or higher
- Hugging Face account with access to DeepFloyd models
- OpenAI API key (for LLM analysis)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/visual_illusion_analyzer.git
cd visual_illusion_analyzer
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the visual_anagrams package:
```bash
python setup_visual_anagrams.py
```

5. Log in to Hugging Face:
```bash
huggingface-cli login
```
You will need to provide your Hugging Face token. Get it from [Hugging Face Settings](https://huggingface.co/settings/tokens).

6. Set up your OpenAI API key:
Create a `.env` file in the project root and add:
```
OPENAI_API_KEY=your-api-key-here
```

## Usage

### 1. Generate Illusions

Use the `generate.py` script from the visual_anagrams package:

```bash
python visual_anagrams/generate.py \
  --name rotate_cw.village.horse \
  --save_dir results \
  --prompts "village" "horse" \
  --views rotate_cw identity \
  --num_samples 1 \
  --generate_1024
```

This will create:
- Sample images (64x64, 256x256, and 1024x1024)
- A metadata.pkl file with view information

### 2. Run Analysis

Run the combined analysis script:

```bash
python combined_analysis.py
```

This will:
- Perform LLM analysis on the illusion
- Generate LIME explanations
- Save results to `analysis_output/combined_analysis.json`

## Analysis Methods

### LLM Analysis
The system uses GPT-4 Vision to analyze visual illusions through several steps:

1. **Initial Observation**
   - The model first observes the image and identifies basic visual elements
   - Records initial impressions and patterns

2. **Transformation Identification**
   - Analyzes how the image changes under different views
   - Identifies geometric transformations (rotation, scaling, etc.)
   - Maps transformations to specific visual elements

3. **Detailed Description**
   - Provides comprehensive description of the illusion
   - Explains how different views reveal different images
   - Describes the relationship between views

4. **Reasoning Process**
   - Tracks step-by-step reasoning
   - Explains how conclusions were reached
   - Provides confidence scores for each aspect

### LIME Analysis
Local Interpretable Model-agnostic Explanations (LIME) is used to understand which parts of the image contribute to the model's decisions:

1. **Feature Importance**
   - Identifies key regions in the image
   - Calculates importance scores for different areas
   - Generates heatmaps showing influential regions

2. **Segmentation Analysis**
   - Divides image into interpretable segments
   - Analyzes contribution of each segment
   - Identifies critical features for each view

3. **Visualization**
   - Creates heatmaps showing important regions
   - Overlays importance scores on original image
   - Generates comparative visualizations

### Combined Analysis
The system combines both methods to provide comprehensive insights:

1. **Cross-Validation**
   - Compares LLM and LIME findings
   - Validates interpretations across methods
   - Identifies areas of agreement/disagreement

2. **Confidence Scoring**
   - Overall confidence score
   - Method-specific confidence metrics
   - Reliability indicators

3. **Output Format**
```json
{
    "llm_analysis": {
        "initial_observation": "string",
        "transformation_identified": "string",
        "description": "string",
        "reasoning_process": ["string"],
        "confidence_scores": {
            "transformation_identification": float,
            "description_accuracy": float,
            "overall_confidence": float
        }
    },
    "lime_analysis": {
        "feature_importance": {
            "scores": [[float]],
            "segments": [[int]],
            "critical_regions": [[int]]
        },
        "visualization_path": "string"
    },
    "combined_metrics": {
        "cross_validation_score": float,
        "overall_confidence": float,
        "reliability_score": float
    }
}
```

## Using LIME to Analyze LLM Decisions

### 1. Basic LIME Analysis

To analyze how the LLM identifies illusions using LIME:

```python
from lime_analyzer import LIMEAnalyzer
import matplotlib.pyplot as plt

# Initialize the analyzer
analyzer = LIMEAnalyzer()

# Analyze a specific illusion
image_path = "results/rotate_cw.village.horse/sample_1024.png"
explanation = analyzer.analyze_image(image_path)

# Visualize the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(explanation['original_image'])
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(explanation['importance_map'])
plt.title('LIME Importance Map')
plt.savefig('lime_analysis.png')
```

### 2. Understanding the LIME Output

The LIME analysis provides several insights:

1. **Feature Importance Map**
   - Red regions: Highly influential areas for the LLM's decision
   - Blue regions: Less important areas
   - Intensity indicates relative importance

2. **Segmentation Analysis**
   - Image is divided into interpretable segments
   - Each segment is assigned an importance score
   - Helps identify which parts of the illusion are most recognizable

3. **Confidence Scores**
   - Overall confidence in the analysis
   - Per-segment confidence levels
   - Reliability of the interpretation

### 3. Comparing Different Views

To analyze how the LLM perceives different views of the same illusion:

```python
# Analyze multiple views
views = ['rotate_cw', 'identity']
results = {}

for view in views:
    image_path = f"results/{view}.village.horse/sample_1024.png"
    results[view] = analyzer.analyze_image(image_path)

# Compare the importance maps
plt.figure(figsize=(15, 5))
for i, (view, result) in enumerate(results.items()):
    plt.subplot(1, len(views), i+1)
    plt.imshow(result['importance_map'])
    plt.title(f'{view} View')
plt.savefig('comparative_analysis.png')
```

### 4. Advanced Analysis

For deeper insights into the LLM's decision-making process:

```python
# Get detailed analysis
detailed_analysis = analyzer.analyze_illusion(
    "results/rotate_cw.village.horse",
    num_samples=1000,  # Number of LIME samples
    segmentation_method='quickshift',  # Segmentation algorithm
    top_labels=3  # Number of top features to analyze
)

# Access the results
print("Top influential regions:", detailed_analysis['top_regions'])
print("Confidence scores:", detailed_analysis['confidence_scores'])
print("Feature importance:", detailed_analysis['feature_importance'])
```

### 5. Interpreting the Results

The LIME analysis helps understand:

1. **Visual Cues**
   - Which parts of the image the LLM focuses on
   - How different views affect recognition
   - Critical features for each interpretation

2. **Transformation Effects**
   - How geometric transformations impact recognition
   - Which transformations are most significant
   - Relationship between views

3. **Confidence Patterns**
   - Areas of high/low confidence
   - Consistency across different views
   - Reliability of interpretations

   LIME : 

   the perturbation process works:
Image Segmentation:
First, the image is divided into segments using the quickshift algorithm
Each segment represents a meaningful region in the image (like objects, textures, or color regions)
This is done using self.segmenter(image)
Perturbation Creation:
For each sample:
A random binary mask is created for all segments
When a segment is masked (value = 0), it's replaced with gray (value = 128)
When a segment is not masked (value = 1), it keeps its original pixels
This creates variations of the image where different regions are masked out
Visualization:
The new visualize_perturbations method shows:
The original image
Multiple perturbed versions with different regions masked
This helps understand how LIME is testing different parts of the image
LLM Analysis:
Each perturbed version is sent to the LLM
The LLM's predictions for each version are recorded
This helps identify which regions are most important for the LLM's understanding

Here's how the code finds the importance map of the LLM's interpretation:
Image Segmentation:
The code uses the quickshift segmentation algorithm to divide the image into meaningful regions
This is done through self.segmenter = SegmentationAlgorithm('quickshift', kernel_size=4, max_dist=200, ratio=0.2)
LIME Explanation Generation:
The explain_instance method:
Takes the original image
Uses predict_fn (which calls the LLM) to get predictions
Generates perturbed versions of the image by masking different segments
Tracks how these perturbations affect the LLM's predictions
Importance Map Creation:
The get_image_and_mask method:
Creates a heatmap showing which regions most influenced the LLM's predictions
Uses positive_only=True to focus on regions that positively influenced the prediction
Shows the top 10 most important regions (num_features=10)
Returns both the original image and the importance mask
Confidence Scoring:
The _calculate_confidence_scores method computes:
Prediction confidence: How confident the LLM is in its predictions
Feature importance confidence: How strongly regions influenced the prediction
Overall confidence: Combined score of importance and prediction confidence
Visualization:
The new visualize_importance_map method shows:
Original image
Importance map (heatmap)
Important regions mask
The process works by:
Segmenting the image into meaningful regions
Systematically masking different regions
Getting LLM predictions for each masked version
Analyzing which regions had the biggest impact on the LLM's predictions
Creating a heatmap showing these important regions
Calculating confidence scores for the analysis
This helps understand:
Which parts of the image the LLM focuses on
How different regions influence the LLM's interpretation
The confidence level in the analysis
The relationship between image regions and the LLM's understanding of the illusion
### 6. Saving and Loading Analysis

To save and reuse your analysis:

```python
# Save analysis results
analyzer.save_analysis(
    detailed_analysis,
    output_path="analysis_output/lime_analysis.json"
)

# Load previous analysis
loaded_analysis = analyzer.load_analysis(
    "analysis_output/lime_analysis.json"
)
```

The saved analysis includes:
- Feature importance maps
- Segmentation results
- Confidence scores
- Visualization paths
- Metadata about the analysis

## Troubleshooting

### Hugging Face Authentication
If you see a 401 Unauthorized error:
1. Make sure you're logged in: `huggingface-cli login`
2. Check that you have access to the DeepFloyd models
3. Visit [DeepFloyd/IF-I-M-v1.0](https://huggingface.co/DeepFloyd/IF-I-M-v1.0) to accept terms if needed

### TensorFlow/Metal Plugin Issues
If you see TensorFlow-related errors:
1. Uninstall TensorFlow if you're only using PyTorch:
```bash
pip uninstall tensorflow tensorflow-macos tensorflow-metal
```
2. Set the environment variable to force PyTorch-only mode:
```bash
export TRANSFORMERS_NO_TF=1
```

## Project Structure

```
visual_illusion_analyzer/
├── combined_analysis.py    # Main analysis script
├── lime_analyzer.py        # LIME analysis implementation
├── llm_analyzer.py         # LLM analysis implementation
├── illusion_analyzer.py    # Core illusion analysis
├── similarity_evaluator.py # Evaluation metrics
├── requirements.txt        # Python dependencies
├── setup_visual_anagrams.py # Setup script
└── results/               # Generated illusions and analysis
```

## Dependencies

- torch>=2.0.0
- torchvision>=0.15.0
- Pillow>=9.0.0
- numpy>=1.21.0
- scikit-learn>=1.0.0
- transformers>=4.30.0
- openai>=1.0.0
- sentence-transformers>=2.2.0
- python-dotenv>=0.19.0
- lime>=0.2.0.1
- matplotlib>=3.5.0
- scikit-image>=0.19.0

## License

This project is licensed under the MIT License - see the LICENSE file for details. 