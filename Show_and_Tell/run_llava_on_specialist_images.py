import json
import os
import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

def load_specialist_results(limit=None):
    with open('val/results.json', 'r') as f:
        results = json.load(f)
    
    if limit:
        results = results[:limit]
    
    return results

def find_image_file(image_id):
    filename = f"COCO_val2014_{image_id:012d}.jpg"
    filepath = f"val/images/{filename}"
    
    if os.path.exists(filepath):
        return filepath
    
    for file in os.listdir("val/images"):
        if str(image_id) in file:
            return f"val/images/{file}"
    
    print(f"Warning: Image file for ID {image_id} not found")
    return None

def setup_llava():
    print("Loading LLaVA model...")
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf", 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    return processor, model

def generate_caption_with_llava(image_path, processor, model):
    try:
        image = Image.open(image_path).convert('RGB')
        
        prompt = "[INST] <image>\nDescribe this image in one sentence. [/INST]"
        
        inputs = processor(prompt, image, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        
        generated_text = processor.decode(output[0], skip_special_tokens=True)
        
        if "[/INST]" in generated_text:
            caption = generated_text.split("[/INST]")[-1].strip()
        else:
            caption = generated_text.strip()
            
        return caption
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return "Error generating caption"

def main():
    sample_limit = 50
    specialist_results = load_specialist_results(limit=sample_limit)
    print(f"Found {len(specialist_results)} images in specialist results (sample of {sample_limit})")
    
    processor, model = setup_llava()
    
    llava_results = []
    
    for i, item in enumerate(specialist_results):
        image_id = item['image_id']
        specialist_caption = item['caption']
        
        print(f"Processing {i+1}/{len(specialist_results)}: Image ID {image_id}")

        image_path = find_image_file(image_id)
        
        if image_path and os.path.exists(image_path):
            llava_caption = generate_caption_with_llava(image_path, processor, model)
            
            llava_results.append({
                'image_id': image_id,
                'image_path': image_path,
                'specialist_caption': specialist_caption,
                'llava_caption': llava_caption
            })
            
            print(f"  Specialist: {specialist_caption}")
            print(f"  LLaVA:      {llava_caption}")
            print()
        else:
            print(f"  Skipping - image file not found")
            
        if (i + 1) % 10 == 0:
            with open('val/llava_specialist_comparison_temp.json', 'w') as f:
                json.dump(llava_results, f, indent=2)
            print(f"Saved intermediate results after {i+1} images")
    
    with open('val/llava_specialist_comparison_50.json', 'w') as f:
        json.dump(llava_results, f, indent=2)
    
    print(f"Completed! Generated captions for {len(llava_results)} images")
    print("Results saved to val/llava_specialist_comparison_50.json")

if __name__ == "__main__":
    main() 