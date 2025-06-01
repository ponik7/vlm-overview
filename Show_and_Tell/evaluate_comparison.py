import json
import os
from collections import defaultdict
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def load_ground_truth_captions():
    print("Loading COCO ground truth captions...")
    with open('val/captions_val2014.json', 'r') as f:
        coco_data = json.load(f)
    
    gt_captions = defaultdict(list)
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption'].lower().strip()
        gt_captions[image_id].append(caption)
    
    return gt_captions

def load_comparison_results():
    try:
        with open('val/llava_specialist_comparison_50.json', 'r') as f:
            results = json.load(f)
        print("Using 50-image comparison results")
    except FileNotFoundError:
        with open('val/llava_specialist_comparison_test.json', 'r') as f:
            results = json.load(f)
        print("Using 20-image comparison results (50-image file not found)")
    
    return results

def calculate_bleu_score(candidate, references):
    try:
        candidate_tokens = word_tokenize(candidate.lower())
        reference_tokens = [word_tokenize(ref.lower()) for ref in references]
        
        smoothing = SmoothingFunction().method1
        score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing)
        return score
    except:
        return 0.0

def calculate_meteor_score(candidate, references):
    try:
        candidate_tokens = word_tokenize(candidate.lower())
        best_score = 0.0
        for ref in references:
            ref_tokens = word_tokenize(ref.lower())
            score = meteor_score([ref_tokens], candidate_tokens)
            best_score = max(best_score, score)
        return best_score
    except:
        return 0.0

def calculate_length_stats(captions):
    lengths = [len(word_tokenize(caption)) for caption in captions]
    return {
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths)
    }

def evaluate_model(model_name, predicted_captions, ground_truth_captions):
    print(f"\nEvaluating {model_name}...")
    
    bleu_scores = []
    meteor_scores = []
    valid_predictions = 0
    
    for item in predicted_captions:
        image_id = item['image_id']
        
        if model_name == "Show & Tell":
            prediction = item['specialist_caption']
        else:  # LLaVA
            prediction = item['llava_caption']
        
        if image_id in ground_truth_captions:
            references = ground_truth_captions[image_id]

            bleu = calculate_bleu_score(prediction, references)
            meteor = calculate_meteor_score(prediction, references)
            
            bleu_scores.append(bleu)
            meteor_scores.append(meteor)
            valid_predictions += 1
        else:
            print(f"Warning: No ground truth found for image {image_id}")
    
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
    avg_meteor = np.mean(meteor_scores) if meteor_scores else 0
    
    all_predictions = []
    for item in predicted_captions:
        if model_name == "Show & Tell":
            all_predictions.append(item['specialist_caption'])
        else:
            all_predictions.append(item['llava_caption'])
    
    length_stats = calculate_length_stats(all_predictions)
    
    results = {
        'model': model_name,
        'valid_predictions': valid_predictions,
        'total_images': len(predicted_captions),
        'avg_bleu': avg_bleu,
        'avg_meteor': avg_meteor,
        'bleu_std': np.std(bleu_scores) if bleu_scores else 0,
        'meteor_std': np.std(meteor_scores) if meteor_scores else 0,
        'length_stats': length_stats
    }
    
    return results, bleu_scores, meteor_scores

def print_detailed_results(results):
    print(f"\n{'='*60}")
    print(f"MODEL: {results['model']}")
    print(f"{'='*60}")
    print(f"Valid predictions: {results['valid_predictions']}/{results['total_images']}")
    print(f"BLEU Score:  {results['avg_bleu']:.4f} ± {results['bleu_std']:.4f}")
    print(f"METEOR Score: {results['avg_meteor']:.4f} ± {results['meteor_std']:.4f}")
    print(f"\nLength Statistics:")
    print(f"  Mean length: {results['length_stats']['mean_length']:.2f} words")
    print(f"  Std length:  {results['length_stats']['std_length']:.2f} words")
    print(f"  Min length:  {results['length_stats']['min_length']} words")
    print(f"  Max length:  {results['length_stats']['max_length']} words")

def print_comparison_examples(comparison_data, num_examples=5):
    print(f"\n{'='*80}")
    print(f"COMPARISON EXAMPLES (First {num_examples} images)")
    print(f"{'='*80}")
    
    for i, item in enumerate(comparison_data[:num_examples]):
        print(f"\nImage {i+1} (ID: {item['image_id']}):")
        print(f"  Show & Tell: {item['specialist_caption']}")
        print(f"  LLaVA:       {item['llava_caption']}")

def main():
    print("Loading data...")
    ground_truth = load_ground_truth_captions()
    comparison_data = load_comparison_results()
    
    print(f"Loaded {len(comparison_data)} comparison results")
    print(f"Loaded ground truth for {len(ground_truth)} images")
    
    specialist_results, specialist_bleu, specialist_meteor = evaluate_model(
        "Show & Tell", comparison_data, ground_truth
    )
    
    llava_results, llava_bleu, llava_meteor = evaluate_model(
        "LLaVA", comparison_data, ground_truth
    )
    
    print_detailed_results(specialist_results)
    print_detailed_results(llava_results)
    
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'Show & Tell':<15} {'LLaVA':<15} {'Difference':<15}")
    print(f"{'-'*60}")
    
    bleu_diff = llava_results['avg_bleu'] - specialist_results['avg_bleu']
    meteor_diff = llava_results['avg_meteor'] - specialist_results['avg_meteor']
    
    print(f"{'BLEU':<15} {specialist_results['avg_bleu']:<15.4f} {llava_results['avg_bleu']:<15.4f} {bleu_diff:+.4f}")
    print(f"{'METEOR':<15} {specialist_results['avg_meteor']:<15.4f} {llava_results['avg_meteor']:<15.4f} {meteor_diff:+.4f}")
    
    print_comparison_examples(comparison_data)
    
    final_results = {
        'specialist_model': specialist_results,
        'llava_model': llava_results,
        'comparison': {
            'bleu_difference': bleu_diff,
            'meteor_difference': meteor_diff,
            'winner_bleu': 'LLaVA' if bleu_diff > 0 else 'Show & Tell',
            'winner_meteor': 'LLaVA' if meteor_diff > 0 else 'Show & Tell'
        }
    }
    
    with open('val/evaluation_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nDetailed results saved to val/evaluation_results.json")

if __name__ == "__main__":
    main() 