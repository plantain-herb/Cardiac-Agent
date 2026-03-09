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


def extract_conversation_rounds(line):
    """Extract all conversation rounds from conversations format.
    Returns a list of tuples: [(human_msg1, gpt_msg1), (human_msg2, gpt_msg2), ...]
    """
    rounds = []
    if 'conversations' in line:
        conversations = line['conversations']
        i = 0
        while i < len(conversations):
            human_msg = None
            gpt_msg = None
            
            # Find human message
            if i < len(conversations) and conversations[i].get('from', '').lower() == 'human':
                human_msg = conversations[i].get('value', '')
                i += 1
            
            # Find corresponding gpt message (optional, may not exist for the last round)
            if i < len(conversations) and conversations[i].get('from', '').lower() == 'gpt':
                gpt_msg = conversations[i].get('value', '')
                i += 1
            
            if human_msg is not None:
                rounds.append((human_msg, gpt_msg))
    
    return rounds


def extract_question_from_conversations(line):
    """Extract question text from conversations format"""
    if 'conversations' in line:
        # Find the first human message as the question
        for conv in line['conversations']:
            if conv.get('from', '').lower() == 'human':
                return conv.get('value', '')
    # Fall back to 'text' field
    return line.get('text', '')


def generate_response(model, tokenizer, input_ids, image_tensor, args, stop_str):
    """Generate response from model"""
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
    
    return outputs


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
    
    # Create separate files for round 1 and round 2
    base_name = os.path.splitext(answers_file)[0]
    ext = os.path.splitext(answers_file)[1] or '.jsonl'
    ans_file_round1 = open(f"{base_name}_round1{ext}", "w")
    ans_file_round2 = open(f"{base_name}_round2{ext}", "w")
    
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
        
        # Extract metadata
        gpt4_answer = line.get("gpt4_answer", "")
        in_text_mention = line.get("in_text_mention", "")
        domain = line.get("domain", "")
        fig_caption = line.get("fig_caption", "")
        conv_type = line.get('type', 'conv')
        
        # Extract all conversation rounds
        conversation_rounds = extract_conversation_rounds(line)
        
        # If no conversations found, fall back to single round using text field
        if not conversation_rounds:
            qs = line.get('text', '')
            if qs:
                conversation_rounds = [(qs, None)]
        
        # Prepare image tensor (shared across rounds)
        image_tensor = None
        if image_files is not None and len(image_files) > 0:
            images = []
            for img_file in image_files:
                image_path = os.path.join(args.image_folder, img_file)
                image = Image.open(image_path).convert('RGB')
                images.append(image)
            
            image_tensors = []
            for image in images:
                img_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                image_tensors.append(img_tensor)
            
            image_tensor = torch.stack(image_tensors, dim=0).half().cuda()
        
        # Process conversation rounds
        conv = conv_templates[args.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        
        round_outputs = []
        
        for round_idx, (human_msg, gpt_ref_msg) in enumerate(conversation_rounds):
            # Clean and prepare the question
            qs_clean = human_msg.replace(DEFAULT_IMAGE_TOKEN, '').strip()
            
            # Only add image tokens for the first round
            if round_idx == 0 and num_images > 0:
                if model.config.mm_use_im_start_end:
                    image_tokens = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n') * num_images
                else:
                    image_tokens = (DEFAULT_IMAGE_TOKEN + '\n') * num_images
                qs = image_tokens + qs_clean
            else:
                qs = qs_clean
            
            # Add the current question to conversation
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            # Prepare input_ids
            if image_files is not None and len(image_files) > 0:
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            else:
                inputs = tokenizer([prompt])
                input_ids = torch.as_tensor(inputs.input_ids).cuda()
            
            # Generate response
            output = generate_response(model, tokenizer, input_ids, image_tensor, args, stop_str)
            round_outputs.append(output)
            
            # Update conversation with the model's response for next round
            # Remove the None placeholder and add actual response
            conv.messages[-1][-1] = output
            
            # Save round result
            ans_id = shortuuid.uuid()
            result = {
                "question_id": idx,
                "round": round_idx + 1,
                "image": image_files if image_files else None,
                "prompt": human_msg,
                "text": output,
                "gpt_reference": gpt_ref_msg,
                "gpt4_answer": gpt4_answer,
                "in_text_mention": in_text_mention,
                "domain": domain,
                "fig_caption": fig_caption,
                "answer_id": ans_id,
                "type": conv_type,
                "model_id": model_name,
                "metadata": {}
            }
            
            if round_idx == 0:
                ans_file_round1.write(json.dumps(result) + "\n")
                ans_file_round1.flush()
            elif round_idx == 1:
                # For round 2, also include round 1's output for reference
                result["round1_output"] = round_outputs[0] if round_outputs else None
                ans_file_round2.write(json.dumps(result) + "\n")
                ans_file_round2.flush()
    
    ans_file_round1.close()
    ans_file_round2.close()
    print(f"Results saved to:")
    print(f"  Round 1: {base_name}_round1{ext}")
    print(f"  Round 2: {base_name}_round2{ext}")

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
