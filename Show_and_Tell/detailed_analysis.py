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

def load_comparison_data():
    with open('val/llava_specialist_comparison_50.json', 'r') as f:
        return json.load(f)

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

def calculate_word_overlap(candidate, references):
    candidate_words = set(word_tokenize(candidate.lower()))
    
    max_overlap = 0.0
    for ref in references:
        ref_words = set(word_tokenize(ref.lower()))
        if len(candidate_words) > 0:
            overlap = len(candidate_words.intersection(ref_words)) / len(candidate_words.union(ref_words))
            max_overlap = max(max_overlap, overlap)
    
    return max_overlap

def analyze_individual_performance():
    ground_truth = load_ground_truth_captions()
    comparison_data = load_comparison_data()
    
    results = []
    
    for item in comparison_data:
        image_id = item['image_id']
        specialist_caption = item['specialist_caption']
        llava_caption = item['llava_caption']
        
        if image_id in ground_truth:
            references = ground_truth[image_id]
            
            specialist_bleu = calculate_bleu_score(specialist_caption, references)
            specialist_meteor = calculate_meteor_score(specialist_caption, references)
            specialist_overlap = calculate_word_overlap(specialist_caption, references)
            
            llava_bleu = calculate_bleu_score(llava_caption, references)
            llava_meteor = calculate_meteor_score(llava_caption, references)
            llava_overlap = calculate_word_overlap(llava_caption, references)
            
            bleu_diff = llava_bleu - specialist_bleu
            meteor_diff = llava_meteor - specialist_meteor
            overlap_diff = llava_overlap - specialist_overlap
            
            results.append({
                'image_id': image_id,
                'image_path': item['image_path'],
                'specialist_caption': specialist_caption,
                'llava_caption': llava_caption,
                'ground_truth': references,
                'specialist_scores': {
                    'bleu': specialist_bleu,
                    'meteor': specialist_meteor,
                    'overlap': specialist_overlap
                },
                'llava_scores': {
                    'bleu': llava_bleu,
                    'meteor': llava_meteor,
                    'overlap': llava_overlap
                },
                'differences': {
                    'bleu': bleu_diff,
                    'meteor': meteor_diff,
                    'overlap': overlap_diff
                },
                'combined_diff': (bleu_diff + meteor_diff + overlap_diff) / 3
            })
    
    return results

def find_cherry_pick_examples(results, top_n=5):
    sorted_results = sorted(results, key=lambda x: x['combined_diff'])
    
    specialist_wins = []
    for item in sorted_results[:top_n]:
        if item['combined_diff'] < -0.1:
            specialist_wins.append(item)
    
    llava_wins = []
    for item in sorted_results[-top_n:]:
        if item['combined_diff'] > 0.1:
            llava_wins.append(item)
    
    return specialist_wins, llava_wins

def analyze_failure_patterns(results):
    specialist_failures = []
    llava_failures = []
    
    for item in results:
        if item['specialist_scores']['bleu'] < 0.1 and item['specialist_scores']['meteor'] < 0.2:
            specialist_failures.append(item)
        
        if item['llava_scores']['bleu'] < 0.1 and item['llava_scores']['meteor'] < 0.2:
            llava_failures.append(item)
    
    return specialist_failures, llava_failures

def print_detailed_comparison():
    print("="*80)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("="*80)
    
    results = analyze_individual_performance()
    
    print("\nA. ОБЩЕЕ КАЧЕСТВО МОДЕЛЕЙ")
    print("-" * 40)
    
    avg_specialist_bleu = np.mean([r['specialist_scores']['bleu'] for r in results])
    avg_specialist_meteor = np.mean([r['specialist_scores']['meteor'] for r in results])
    avg_specialist_overlap = np.mean([r['specialist_scores']['overlap'] for r in results])
    
    avg_llava_bleu = np.mean([r['llava_scores']['bleu'] for r in results])
    avg_llava_meteor = np.mean([r['llava_scores']['meteor'] for r in results])
    avg_llava_overlap = np.mean([r['llava_scores']['overlap'] for r in results])
    
    print(f"Show & Tell - BLEU: {avg_specialist_bleu:.4f}, METEOR: {avg_specialist_meteor:.4f}, Overlap: {avg_specialist_overlap:.4f}")
    print(f"LLaVA       - BLEU: {avg_llava_bleu:.4f}, METEOR: {avg_llava_meteor:.4f}, Overlap: {avg_llava_overlap:.4f}")
    
    specialist_wins_count = sum(1 for r in results if r['combined_diff'] < 0)
    llava_wins_count = sum(1 for r in results if r['combined_diff'] > 0)
    ties_count = sum(1 for r in results if r['combined_diff'] == 0)
    
    print(f"\nПобеды по отдельным примерам:")
    print(f"Show & Tell: {specialist_wins_count} примеров")
    print(f"LLaVA: {llava_wins_count} примеров")
    print(f"Ничьи: {ties_count} примеров")
    
    print("\nB. CHERRY-PICKING ПРИМЕРОВ")
    print("-" * 40)
    
    specialist_best, llava_best = find_cherry_pick_examples(results)
    
    print(f"\nПРИМЕРЫ ГДЕ SHOW & TELL ПРЕВОСХОДИТ LLAVA:")
    if specialist_best:
        for i, item in enumerate(specialist_best):
            print(f"\nПример {i+1} (Image ID: {item['image_id']}):")
            print(f"Ground Truth: {item['ground_truth'][0]}")
            print(f"Show & Tell:  {item['specialist_caption']}")
            print(f"              BLEU: {item['specialist_scores']['bleu']:.3f}, METEOR: {item['specialist_scores']['meteor']:.3f}")
            print(f"LLaVA:        {item['llava_caption']}")
            print(f"              BLEU: {item['llava_scores']['bleu']:.3f}, METEOR: {item['llava_scores']['meteor']:.3f}")
            print(f"Разность:     {item['combined_diff']:.3f} (в пользу Show & Tell)")
    else:
        print("Не найдено примеров где Show & Tell значительно превосходит LLaVA")
    
    print(f"\nПРИМЕРЫ ГДЕ LLAVA ПРЕВОСХОДИТ SHOW & TELL:")
    if llava_best:
        for i, item in enumerate(llava_best[-3:]):
            print(f"\nПример {i+1} (Image ID: {item['image_id']}):")
            print(f"Ground Truth: {item['ground_truth'][0]}")
            print(f"Show & Tell:  {item['specialist_caption']}")
            print(f"              BLEU: {item['specialist_scores']['bleu']:.3f}, METEOR: {item['specialist_scores']['meteor']:.3f}")
            print(f"LLaVA:        {item['llava_caption']}")
            print(f"              BLEU: {item['llava_scores']['bleu']:.3f}, METEOR: {item['llava_scores']['meteor']:.3f}")
            print(f"Разность:     {item['combined_diff']:.3f} (в пользу LLaVA)")
    
    specialist_failures, llava_failures = analyze_failure_patterns(results)
    
    print(f"\nАНАЛИЗ НЕУДАЧ:")
    print(f"Show & Tell: {len(specialist_failures)} полных неудач из {len(results)}")
    print(f"LLaVA: {len(llava_failures)} полных неудач из {len(results)}")
    
    if specialist_failures:
        print(f"\nПример неудачи Show & Tell:")
        failure = specialist_failures[0]
        print(f"Ground Truth: {failure['ground_truth'][0]}")
        print(f"Show & Tell:  {failure['specialist_caption']} (BLEU: {failure['specialist_scores']['bleu']:.3f})")
    
    if llava_failures:
        print(f"\nПример неудачи LLaVA:")
        failure = llava_failures[0]
        print(f"Ground Truth: {failure['ground_truth'][0]}")
        print(f"LLaVA:        {failure['llava_caption']} (BLEU: {failure['llava_scores']['bleu']:.3f})")
    
    
    bleu_improvement = ((avg_llava_bleu - avg_specialist_bleu) / avg_specialist_bleu) * 100
    meteor_improvement = ((avg_llava_meteor - avg_specialist_meteor) / avg_specialist_meteor) * 100
    overlap_improvement = ((avg_llava_overlap - avg_specialist_overlap) / avg_specialist_overlap) * 100
    
    print(f"1. КОЛИЧЕСТВЕННЫЕ УЛУЧШЕНИЯ LLaVA:")
    print(f"   • BLEU: +{bleu_improvement:.1f}%")
    print(f"   • METEOR: +{meteor_improvement:.1f}%")
    print(f"   • Word Overlap: +{overlap_improvement:.1f}%")
    
    
    with open('val/detailed_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)

def main():
    print_detailed_comparison()

if __name__ == "__main__":
    main()