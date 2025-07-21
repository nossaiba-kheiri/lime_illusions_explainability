import os
from lime_analyzer import LIMEAnalyzer
import matplotlib.pyplot as plt
from dotenv import load_dotenv

def test_single_image():
    """Test LIME analysis on a single image"""
    # Initialize analyzer
    analyzer = LIMEAnalyzer()
    
    # Path to your illusion image
    image_path = "results/rotate_cw.village.horse/sample_1024.png"
    
    print(f"Analyzing image: {image_path}")
    results = analyzer.analyze_image(image_path)
    
    # Save results
    os.makedirs("analysis_output", exist_ok=True)
    analyzer.save_analysis(results, "analysis_output/single_image_analysis.json")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(results['original_image'])
    plt.title('Original Image')
    
    # Importance map
    plt.subplot(1, 3, 2)
    plt.imshow(results['importance_map'])
    plt.title('Importance Map')
    
    # Mask
    plt.subplot(1, 3, 3)
    plt.imshow(results['mask'])
    plt.title('Important Regions')
    
    plt.savefig('analysis_output/single_image_visualization.png')
    plt.close()
    
    # Print analysis results
    print("\nLLM Analysis Results:")
    print("Initial Analysis Scores:", results['llm_analysis']['initial_analysis'])
    print("\nImportant Regions Analysis:")
    for region in results['llm_analysis']['important_regions']:
        print(f"\nRegion {region['region_id']} (Importance: {region['importance']:.2f}):")
        print(f"Analysis scores: {region['analysis']}")

def test_illusion_directory():
    """Test LIME analysis on a complete illusion with multiple views"""
    # Initialize analyzer
    analyzer = LIMEAnalyzer()
    
    # Path to your illusion directory
    illusion_dir = "results/rotate_cw.village.horse"
    
    print(f"Analyzing illusion directory: {illusion_dir}")
    results = analyzer.analyze_illusion(illusion_dir)
    
    # Save results
    os.makedirs("analysis_output", exist_ok=True)
    analyzer.save_analysis(results, "analysis_output/illusion_analysis.json")
    
    # Print comparative analysis
    print("\nComparative Analysis:")
    print("Common Regions:", results['comparative']['common_regions'])
    print("\nView Similarities:")
    for view_pair, similarity in results['comparative']['view_similarity'].items():
        print(f"{view_pair}: {similarity:.2f}")
    
    print("\nConfidence Comparison:")
    for view, confidence in results['comparative']['confidence_comparison'].items():
        print(f"\n{view}:")
        for metric, score in confidence.items():
            print(f"  {metric}: {score:.2f}")

def main():
    """Main function to run tests"""
    # Load environment variables (for OpenAI API key)
    load_dotenv()
    
    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in a .env file or environment variables")
        return
    
    print("Starting LIME analysis tests...")
    
    # Test single image analysis
    print("\n=== Testing Single Image Analysis ===")
    test_single_image()
    
    # Test illusion directory analysis
    print("\n=== Testing Illusion Directory Analysis ===")
    test_illusion_directory()
    
    print("\nTests completed! Check the analysis_output directory for results.")

if __name__ == "__main__":
    main() 