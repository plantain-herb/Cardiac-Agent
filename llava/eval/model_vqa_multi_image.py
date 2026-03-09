import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_questions(question_file):
    """Load questions from JSON or JSONL file, supporting both formats"""
    question_file = os.path.expanduser(question_file)
    
    # Try to load as JSON array first
    try:
        with open(question_file, "r") as f:
            content = f.read().strip()
            if content.startswith('['):
                return json.loads(content)
    except:
        pass
    
    # Fall back to JSONL format
    questions = []
    with open(question_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def extract_question_from_conversations(line):
    """Extract question text from conversations format"""
    if 'conversations' in line:
        # Find the first human message as the question
        for conv in line['conversations']:
            if conv.get('from', '').lower() == 'human':
                return conv.get('value', '')
    # Fall back to 'text' field
    return line.get('text', '')


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # If image_processor is None, try to get it from vision_tower
    if image_processor is None:
        vision_tower = model.get_vision_tower()
        if vision_tower is not None:
            if not vision_tower.is_loaded:
                vision_tower.load_model()
            vision_tower.to(device='cuda', dtype=torch.float16)
            image_processor = vision_tower.image_processor
            print("Loaded image_processor from vision_tower")

    questions = load_questions(args.question_file)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    if os.path.dirname(answers_file):
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    for line in tqdm(questions):
        # Support both 'question_id' and 'id' fields
        idx = line.get("question_id", line.get("id", "unknown"))
        
        # Get image file(s) - support both single image (string) and multiple images (list)
        if 'image' in line:
            image_file = line["image"]
            if isinstance(image_file, str):
                image_files = [image_file]
            else:
                image_files = image_file
        else:
            image_files = None
        
        num_images = len(image_files) if image_files else 0
        
        # Extract question text
        if 'pre_text' in line:
            qs_pre = line['pre_text']
            assist = line['assist']
        else:
            qs_pre = None
        
        # Get question from conversations or text field
        qs = extract_question_from_conversations(line)
        
        if "gpt4_answer" in line:
            gpt4_answer = line['gpt4_answer']
        else:
            gpt4_answer = ""
        if "in_text_mention" in line:
            in_text_mention = line["in_text_mention"]
        else:
            in_text_mention = ""
        if "domain" in line:
            domain = line["domain"]
        else:
            domain = ""
        if "fig_caption" in line:
            fig_caption = line["fig_caption"]
        else:
            fig_caption = ""
        conv_type = line['type'] if 'type' in line else 'conv'
        if 'pre_text' in line:
            qs = line["pre_text"]
        cur_prompt = qs
        
        # Remove existing <image> tokens and add correct number of image tokens
        qs_clean = qs.replace(DEFAULT_IMAGE_TOKEN, '').strip()
        if num_images > 0:
            if model.config.mm_use_im_start_end:
                image_tokens = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n') * num_images
            else:
                image_tokens = (DEFAULT_IMAGE_TOKEN + '\n') * num_images
            qs = image_tokens + qs_clean
        else:
            qs = qs_clean

        conv = conv_templates[args.conv_mode].copy()
        if 'pre_text' in line:
            conv.append_message(conv.roles[0], qs_pre)
            conv.append_message(conv.roles[1], assist)
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if image_files is not None and len(image_files) > 0:
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            
            # Load and process all images
            images = []
            for img_file in image_files:
                image_path = os.path.join(args.image_folder, img_file)
                image = Image.open(image_path).convert('RGB')
                images.append(image)
            
            # Process images
            image_tensors = []
            for image in images:
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                image_tensors.append(image_tensor)
            
            # Stack all images into a single tensor (N, C, H, W)
            image_tensor = torch.stack(image_tensors, dim=0).half().cuda()
        else:
            inputs = tokenizer([prompt])
            input_ids = torch.as_tensor(inputs.input_ids).cuda()
            image_tensor = None
            
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        if image_tensor is not None:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True,
                )
        else:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True,
                )
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
            
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "image": image_files if image_files else None,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "gpt4_answer": gpt4_answer,
                                   "in_text_mention": in_text_mention,
                                   "domain": domain,
                                   "fig_caption": fig_caption,
                                   "answer_id": ans_id,
                                   "type": conv_type,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
