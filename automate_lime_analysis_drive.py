import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import concurrent.futures # For parallel processing
import os 
from tqdm import tqdm 
from dotenv import load_dotenv

from lime_analyzer import LIMEAnalyzer


load_dotenv()

# --- Configuration for your specific run ---
# Path to the directory where you've downloaded the images and Excel file from Google Drive.
# This can be set via an environment variable 'LOCAL_CONTENT_DIR' or defaults to a local path.
LOCAL_DRIVE_CONTENT_DIR = Path(os.getenv(
    'LOCAL_CONTENT_DIR',
    "/app/local_content" # Fallback to Docker-internal path
))
EXCEL_FILE_NAME = "fourth_dataset.xlsx" # Adjust if your Excel file has a different name

# Output directory for importance maps and summary JSON
OUT_DIR = Path('batch_importance_maps_drive')

def extract_info_from_filename(filename: str) -> Dict[str, str]:
    """
    Extracts illusion ID and view from filenames like 'illusion_fourth_set_0278_view_1.png'.
    """
    match = re.search(r"(\d{3,})_view_(\d+)\.png$", filename)
    if match:
        illusion_id_str = match.group(1)
        # Convert to int and back to str to remove leading zeros
        clean_illusion_id = str(int(illusion_id_str))
        return {
            "illusion_id": clean_illusion_id,
            "view": match.group(2)
        }
    return {}

def _process_single_analysis(
    image_path: Path,
    illusion_id: str,
    view: str,
    prompt_type: str,
    prompt: str,
    openai_api_key: str,
    output_dir: Path,
    sim_model_local_dir: Optional[str] = None,
    hf_offline: Optional[bool] = None # Add hf_offline to the signature
) -> Dict[str, Any]:
    """
    Performs a single LIME analysis for an image-prompt pair and saves results.
    This function is designed to be run in parallel.
    """
    # Each parallel process needs its own analyzer instance to avoid race conditions
    # with shared state (like self.llm_responses_for_lime and self.current_prompt)
    # It also needs its own OpenAI API key to be set if not in environment.
    os.environ['OPENAI_API_KEY'] = openai_api_key

    # Temporarily remove proxy environment variables to prevent TypeError with OpenAI client
    # The openai library's underlying httpx client might automatically pick these up,
    # and the Client.__init__() doesn't accept a 'proxies' argument directly in newer versions.
    original_http_proxy = os.environ.pop('HTTP_PROXY', None)
    original_https_proxy = os.environ.pop('HTTPS_PROXY', None)

    try:
        print(f"[DEBUG PROCESS] sim_model_local_dir: {sim_model_local_dir}, hf_offline: {hf_offline}") # Debug print
        analyzer = LIMEAnalyzer(mode='auto', sim_model_local_dir=sim_model_local_dir, hf_offline=hf_offline)
    finally:
        # Restore proxy environment variables after LIMEAnalyzer initialization
        if original_http_proxy is not None:
            os.environ['HTTP_PROXY'] = original_http_proxy
        if original_https_proxy is not None:
            os.environ['HTTPS_PROXY'] = original_https_proxy

    result_entry: Dict[str, Any] = {
        'illusion_id': illusion_id,
        'view': view,
        'original_image_filename': image_path.name,
        'prompt_type': prompt_type,
        'prompt': prompt,
        'importance_map_filename': 'N/A',
        'confidence_scores': {},
        'perturbed_llm_responses': 'N/A',
        'error': 'None'
    }

    try:
        print(f"[run] Analyzing {image_path.name} with {prompt_type}: {prompt[:50]}...")
        results = analyzer.analyze_image(str(image_path), prompt, num_samples=5)

        map_out = output_dir / f"importance_map_{illusion_id}_view_{view}_{prompt_type}.png"
        analyzer.visualize_importance_map(results, save_path=str(map_out))

        result_entry['importance_map_filename'] = map_out.name
        result_entry['confidence_scores'] = results.get('confidence_scores', {})
        result_entry['perturbed_llm_responses'] = json.dumps(results['llm_analysis']['perturbed_image_llm_responses'])
        
    except Exception as e:
        print(f"[error] {image_path.name} ({prompt_type}): {e}")
        result_entry['error'] = str(e)

    return result_entry

def main():
    # Explicitly remove proxy environment variables from os.environ
    # to prevent TypeError with OpenAI client if they are implicitly set.
    os.environ.pop('HTTP_PROXY', None)
    os.environ.pop('HTTPS_PROXY', None)

    OUT_DIR.mkdir(exist_ok=True)

    if not LOCAL_DRIVE_CONTENT_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {LOCAL_DRIVE_CONTENT_DIR.resolve()}. Please download the Google Drive content here.")
    
    excel_path = LOCAL_DRIVE_CONTENT_DIR / EXCEL_FILE_NAME
    if not excel_path.exists():
        raise FileNotFoundError(f"Missing Excel file: {excel_path.resolve()}. Make sure it's in the LOCAL_DRIVE_CONTENT_DIR.")

    # Verify OpenAI key if using auto mode
    use_auto = True
    if os.getenv('OPENAI_API_KEY'):
        os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY').strip()
    else:
        raise RuntimeError("OPENAI_API_KEY is not set. Please ensure it's in your .env file or set it in your environment.")
    
    # Retrieve API key and SIM_MODEL_LOCAL_DIR once to pass to parallel processes
    openai_api_key = os.getenv('OPENAI_API_KEY')
    sim_model_local_dir = os.getenv('SIM_MODEL_LOCAL_DIR')
    hf_offline = os.getenv("HF_OFFLINE", "").lower() in ("1", "true", "yes") # Read HF_OFFLINE here

    # Initialize analyzer (only if needed for non-parallel initial setup, but not for core LIME runs)
    # analyzer = LIMEAnalyzer(mode='auto') # No longer needed here as each worker gets its own

    # Load prompts from Excel
    df = pd.read_excel(excel_path)
    df['illusion_id'] = df['identifier'].astype(str) # Ensure identifier is string for matching
    print(f"Loaded {len(df)} prompts from {excel_path}")

    tasks = []

    # Process images to create tasks
    image_files = list(LOCAL_DRIVE_CONTENT_DIR.glob('*.png')) # Assuming images are PNGs
    if not image_files:
        print(f"[warning] No PNG images found in {LOCAL_DRIVE_CONTENT_DIR.resolve()}")

    for img_path in image_files:
        filename_info = extract_info_from_filename(img_path.name)
        illusion_id = filename_info.get("illusion_id")
        view = filename_info.get("view")

        if not illusion_id or not view:
            print(f"[skip] Could not extract illusion_id or view from filename: {img_path.name}")
            continue

        # Find corresponding prompts in the DataFrame
        matching_rows = df[df['illusion_id'] == illusion_id]

        if matching_rows.empty:
            print(f"[skip] No matching entry found in Excel for illusion_id: {illusion_id} (image: {img_path.name})")
            continue
        
        # Assuming there's only one row per illusion_id or taking the first one
        prompt_data = matching_rows.iloc[0]
        prompt1 = prompt_data.get('prompt_1')
        prompt2 = prompt_data.get('prompt_2')

        if not prompt1 and not prompt2:
            print(f"[skip] No prompts (prompt_1, prompt_2) found for illusion_id: {illusion_id} (image: {img_path.name})")
            continue

        # Add tasks for parallel execution
        if prompt1:
            tasks.append((img_path, illusion_id, view, 'prompt_1', prompt1, openai_api_key, OUT_DIR, sim_model_local_dir, hf_offline))
        if prompt2:
            tasks.append((img_path, illusion_id, view, 'prompt_2', prompt2, openai_api_key, OUT_DIR, sim_model_local_dir, hf_offline))
    
    max_workers = os.cpu_count() * 2 if os.cpu_count() else 4 # Max workers based on CPU count
    print(f"Prepared {len(tasks)} analysis tasks for parallel execution.")
    print(f"Running with up to {max_workers} parallel workers.")

    summary = [] # Initialize the summary list

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map the _process_single_analysis function to all tasks and show progress
        for result_entry in tqdm(
            executor.map(lambda p: _process_single_analysis(*p), tasks),
            total=len(tasks),
            desc="Processing LIME analyses"
        ):
            summary.append(result_entry)

    # Save the aggregated summary to a CSV file
    if summary:
        summary_df = pd.DataFrame(summary)
        csv_path = OUT_DIR / 'metadata.csv'
        summary_df.to_csv(csv_path, index=False)
        print(f"Done. Saved metadata to {csv_path.resolve()}")
    else:
        print("No analysis results to save.")


if __name__ == '__main__':
    main() 