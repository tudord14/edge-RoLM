import os
import math
import csv
import glob
import random
from tqdm import tqdm
import torch
from llama_cpp import Llama
from typing import Tuple, List


def find_model_files(folder: str) -> List[str]:
    """
    Find all GGUF model files in the specified folder.

    Args:
        folder: Path to folder containing model files

    Returns:
        List of model filenames found
    """
    print(f"🔍 Looking for .gguf model files in {folder}...")

    # Find all .gguf files in the folder
    gguf_files = glob.glob(os.path.join(folder, "*.gguf"))
    model_names = [os.path.basename(f) for f in gguf_files]

    if not model_names:
        print(f"❌ No .gguf model files found in {folder}")
        return []

    print(f"📁 Found {len(model_names)} model files:")
    for model in sorted(model_names):
        print(f"  - {model}")

    return sorted(model_names)


def sample_dataset(dataset_path: str, sample_percentage: float = 0.01, seed: int = 42) -> str:
    """
    Randomly sample a percentage of the dataset text.

    Args:
        dataset_path: Path to the text dataset
        sample_percentage: Percentage of text to sample (0.1 = 10%)
        seed: Random seed for reproducible sampling

    Returns:
        Sampled text as string
    """
    random.seed(seed)

    print(f"📖 Reading dataset from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    original_size = len(full_text)
    print(f"📊 Original dataset: {original_size:,} characters ({original_size / (1024 * 1024):.2f} MB)")

    # Split into characters and randomly sample
    chars = list(full_text)
    sample_size = int(len(chars) * sample_percentage)

    # Randomly sample characters while maintaining some locality for coherence
    sampled_chars = []

    # Use chunk-based sampling to maintain some text coherence
    chunk_size = max(100, len(chars) // 1000)  # Reasonable chunk size
    num_chunks = len(chars) // chunk_size
    chunks_to_sample = int(num_chunks * sample_percentage)

    if chunks_to_sample > 0:
        # Randomly select chunk indices
        selected_chunk_indices = random.sample(range(num_chunks), min(chunks_to_sample, num_chunks))

        for chunk_idx in sorted(selected_chunk_indices):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(chars))
            sampled_chars.extend(chars[start_idx:end_idx])
    else:
        # Fallback to character-level sampling for very small datasets
        sampled_chars = random.sample(chars, sample_size)

    sampled_text = ''.join(sampled_chars)
    sampled_size = len(sampled_text)

    print(f"🎯 Sampled dataset: {sampled_size:,} characters ({sampled_size / (1024 * 1024):.2f} MB)")
    print(f"📉 Reduction: {((original_size - sampled_size) / original_size * 100):.1f}%")

    return sampled_text


def compute_perplexity(model_path: str, text: str,
                       block_size: int = 512, stride: int = 256) -> Tuple[float, int]:
    """
    Compute perplexity using sliding window approach to maintain context.

    Args:
        model_path: Path to the GGUF model file
        text: Text to evaluate (already sampled)
        block_size: Size of each evaluation window
        stride: Step size for sliding window (smaller = more overlap)

    Returns:
        Tuple of (perplexity, total_tokens_evaluated)
    """
    print(f"\n🔄 Evaluating {model_path}...")

    # Load model with larger context to accommodate sliding windows
    llm = Llama(
        model_path=model_path,
        n_ctx=block_size + stride,  # Extra room for context
        n_threads=4,
        verbose=False
    )

    total_nll = 0.0
    total_count = 0

    # Tokenize the sampled text
    tokens = llm.tokenize(text.encode("utf-8"))
    print(f"📊 Tokenized {len(tokens)} tokens")

    if len(tokens) < 2:
        print("⚠️ Dataset too small!")
        return float('inf'), 0

    # Use sliding window approach
    for start_idx in tqdm(range(0, len(tokens) - block_size, stride),
                          desc="Processing", unit="window"):

        end_idx = min(start_idx + block_size, len(tokens))
        window = tokens[start_idx:end_idx]

        if len(window) < 2:
            continue

        try:
            # Reset only at the beginning or when context gets too long
            if start_idx == 0 or start_idx % (block_size * 4) == 0:
                llm.reset()

            # Process tokens one by one to get proper logits alignment
            window_nll = 0.0
            window_count = 0

            # Feed context (all but last token)
            context = window[:-1]

            # Evaluate the sequence to get logits for each position
            llm.eval(context)

            # Get logits for the last token in context
            logits = torch.tensor(llm.get_logits(), dtype=torch.float32)
            log_probs = torch.log_softmax(logits, dim=-1)

            # The last token in context predicts the last token in window
            target_token = window[-1]
            window_nll -= log_probs[target_token].item()
            window_count += 1

            # For longer sequences, process incrementally
            if len(window) > 2:
                for i in range(len(context) - 1):
                    # Reset and eval up to position i
                    partial_context = context[:i + 1]
                    llm.reset()
                    llm.eval(partial_context)

                    logits = torch.tensor(llm.get_logits(), dtype=torch.float32)
                    log_probs = torch.log_softmax(logits, dim=-1)

                    target_token = context[i + 1] if i + 1 < len(context) else window[-1]
                    window_nll -= log_probs[target_token].item()
                    window_count += 1

            total_nll += window_nll
            total_count += window_count

        except Exception as e:
            print(f"⚠️ Error processing window starting at {start_idx}: {e}")
            continue

    if total_count == 0:
        print("⚠️ No tokens were successfully processed!")
        return float('inf'), 0

    perplexity = math.exp(total_nll / total_count)
    return perplexity, total_count


def compute_perplexity_simple(model_path: str, text: str, max_tokens: int = 512) -> Tuple[float, int]:
    """
    Compute perplexity using a simpler, more memory-efficient method.
    """
    print(f"\n🔄 Evaluating {model_path} (simple method)...")

    llm = Llama(model_path=model_path, n_ctx=max_tokens + 100, n_threads=4, verbose=False)

    # Tokenize text
    tokens = llm.tokenize(text.encode("utf-8"))

    # Limit tokens to avoid context overflow
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        print(f"📊 Using first {len(tokens)} tokens")
    else:
        print(f"📊 Processing all {len(tokens)} tokens")

    if len(tokens) < 2:
        print("⚠️ Text too small!")
        return float('inf'), 0

    total_nll = 0.0
    total_count = 0

    try:
        # Use a much smaller evaluation window to avoid memory issues
        eval_size = min(100, len(tokens) - 1)  # Very conservative size

        llm.reset()

        # Process a smaller subset of tokens
        for i in tqdm(range(1, eval_size + 1), desc="Evaluating", leave=False):
            try:
                # Reset and evaluate up to position i
                llm.reset()

                # Evaluate the sequence up to position i
                output = llm(tokens[:i], logits_all=True, echo=False)

                # Get the logits from the output
                if hasattr(output, 'logits') and output.logits is not None:
                    # Use the last logits
                    logits = torch.tensor(output.logits[-1], dtype=torch.float32)
                elif hasattr(llm, 'scores') and llm.scores is not None:
                    # Alternative: try to get scores
                    logits = torch.tensor(llm.scores[-1], dtype=torch.float32)
                else:
                    # Try the eval method approach
                    llm.eval(tokens[:i])
                    # Check different possible attribute names
                    if hasattr(llm, '_model') and hasattr(llm._model, 'logits'):
                        logits = torch.tensor(llm._model.logits, dtype=torch.float32)
                    elif hasattr(llm, 'ctx') and hasattr(llm.ctx, 'logits'):
                        logits = torch.tensor(llm.ctx.logits, dtype=torch.float32)
                    else:
                        # Last resort: use the call method to get probabilities
                        prob_output = llm.generate([tokens[i]], max_tokens=1, temperature=0, top_p=1)
                        # Skip this token if we can't get logits
                        continue

                log_probs = torch.log_softmax(logits, dim=-1)
                target_token = tokens[i]
                total_nll -= log_probs[target_token].item()
                total_count += 1

            except Exception as inner_e:
                print(f"⚠️ Skipping token {i}: {inner_e}")
                continue

        print(f"📊 Successfully evaluated {total_count} tokens")

    except Exception as e:
        print(f"⚠️ Error during evaluation: {e}")
        return float('inf'), 0

    if total_count == 0:
        print("⚠️ No tokens successfully evaluated!")
        return float('inf'), 0

    perplexity = math.exp(total_nll / total_count)
    return perplexity, total_count


def main():
    folder = ""
    dataset_file = os.path.join(folder, "dataset.txt")

    # Check if dataset exists
    if not os.path.exists(dataset_file):
        print(f"❌ Dataset not found: {dataset_file}")
        return

    # Sample 10% of the dataset
    print("🎯 Step 1: Sampling 10% of dataset")
    try:
        sampled_text = sample_dataset(dataset_file, sample_percentage=0.1, seed=42)
    except Exception as e:
        print(f"❌ Error sampling dataset: {e}")
        return

    print("\n🚀 Step 2: Finding model files")

    # Automatically find all .gguf model files in the folder
    models = find_model_files(folder)

    if not models:
        print("❌ No model files found. Exiting.")
        return

    print("\n🚀 Step 3: Evaluating models")

    results = []

    for model in models:
        model_path = os.path.join(folder, model)

        if not os.path.exists(model_path):
            print(f"⚠️ Missing model: {model}")
            results.append((model, "MISSING", 0))
            continue

        try:
            # Use the simpler method for better reliability with sampled data
            ppl, ntokens = compute_perplexity_simple(model_path, sampled_text, max_tokens=256)

        except Exception as e:
            print(f"⚠️ Error with {model}: {e}")
            results.append((model, "ERROR", 0))
            continue

        print(f"✅ {model}: Perplexity={ppl:.2f} (on {ntokens} tokens)")
        results.append((model, f"{ppl:.2f}", ntokens))

    # Save results
    csv_file = os.path.join(folder, "perplexity_results_10pct.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "perplexity", "num_tokens"])
        writer.writerows(results)

    print(f"\n📂 Results saved to {csv_file}")

    # Print summary
    print("\n📊 Summary:")
    for model, ppl, ntokens in results:
        if isinstance(ppl, str) and ppl not in ["MISSING", "ERROR"]:
            print(f"  {model}: {ppl}")
        elif ppl == "MISSING":
            print(f"  {model}: Missing file")
        elif ppl == "ERROR":
            print(f"  {model}: Processing error")


if __name__ == "__main__":
    main()