"""LLaVA Agent 模型客户端"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import DEFAULT_IMAGE_TOKEN

from app.config import API_NAME_TO_WORKER, AGENT_URL, CONTROLLER_URL
from app.utils.dicom import encode_image_to_base64
from app.utils.http_client import get_http_session


class LLaVAAgentClient:
    """LLaVA Agent 模型客户端"""

    def __init__(self, controller_url: str = CONTROLLER_URL):
        self.controller_url = controller_url
        self.agent_url = None
        self.session = get_http_session()
        self._find_agent()

    def _find_agent(self):
        try:
            resp = self.session.post(f"{self.controller_url}/list_models", timeout=5)
            models = resp.json().get("models", [])

            for model in models:
                if "llava" in model.lower() or "agent" in model.lower():
                    resp = self.session.post(
                        f"{self.controller_url}/get_worker_address",
                        json={"model": model},
                        timeout=5
                    )
                    self.agent_url = resp.json().get("address")
                    print(f"找到Agent模型: {model} at {self.agent_url}")
                    return

            self.agent_url = AGENT_URL
            print(f"使用默认Agent地址: {self.agent_url}")

        except Exception as e:
            print(f"查找Agent失败: {e}")
            self.agent_url = AGENT_URL

    def chat(self, prompt: str, images: List[str] = None) -> Tuple[str, Optional[Dict]]:
        """与 Agent 进行单轮对话"""
        if not self.agent_url:
            return "Agent不可用", None

        conv = conv_templates["v1"].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        num_images = len(images) if images else 0
        if num_images > 0:
            prompt_clean = prompt.replace(DEFAULT_IMAGE_TOKEN, '').strip()
            image_tokens = (DEFAULT_IMAGE_TOKEN + '\n') * num_images
            prompt = image_tokens + prompt_clean

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        formatted_prompt = conv.get_prompt()

        pload = {
            "model": "agent",
            "prompt": formatted_prompt,
            "temperature": 0.2,
            "max_new_tokens": 1024,
            "stop": stop_str,
        }

        if images:
            encoded_images = []
            for img in images:
                if os.path.exists(img):
                    encoded_images.append(encode_image_to_base64(img))
                else:
                    encoded_images.append(img)
            pload["images"] = encoded_images

        try:
            resp = self.session.post(
                f"{self.agent_url}/worker_generate_stream",
                json=pload,
                stream=True,
                timeout=60
            )

            full_response = ""
            for chunk in resp.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    data = json.loads(chunk.decode())
                    if data.get("error_code", 0) == 0:
                        full_response = data.get("text", "")

            action = self._parse_action(full_response)
            return full_response, action

        except Exception as e:
            print(f"Agent请求失败: {e}")
            return str(e), None

    def chat_with_history(self, messages: List[Dict], images: List[str] = None) -> Tuple[str, Optional[Dict]]:
        """与 Agent 进行多轮对话"""
        if not self.agent_url:
            return "Agent不可用", None

        conv = conv_templates["v1"].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        num_images = len(images) if images else 0

        for i, msg in enumerate(messages):
            role = conv.roles[0] if msg["role"] == "human" else conv.roles[1]
            content = msg["content"]

            if i == 0 and msg["role"] == "human" and num_images > 0:
                content_clean = content.replace(DEFAULT_IMAGE_TOKEN, '').strip()
                image_tokens = (DEFAULT_IMAGE_TOKEN + '\n') * num_images
                content = image_tokens + content_clean

            conv.append_message(role, content)

        conv.append_message(conv.roles[1], None)
        formatted_prompt = conv.get_prompt()

        pload = {
            "model": "agent",
            "prompt": formatted_prompt,
            "temperature": 0.2,
            "max_new_tokens": 1024,
            "stop": stop_str,
        }

        if images:
            encoded_images = []
            for img in images:
                if os.path.exists(img):
                    encoded_images.append(encode_image_to_base64(img))
                else:
                    encoded_images.append(img)
            pload["images"] = encoded_images

        try:
            resp = self.session.post(
                f"{self.agent_url}/worker_generate_stream",
                json=pload,
                stream=True,
                timeout=60
            )

            full_response = ""
            for chunk in resp.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    data = json.loads(chunk.decode())
                    if data.get("error_code", 0) == 0:
                        full_response = data.get("text", "")

            if "ASSISTANT:" in full_response:
                last_assistant = full_response.rsplit("ASSISTANT:", 1)[-1]
                if last_assistant.strip():
                    full_response = last_assistant.strip()

            action = self._parse_action(full_response)
            return full_response, action

        except Exception as e:
            print(f"Agent请求失败: {e}")
            return str(e), None

    def _parse_action(self, response: str) -> Optional[Dict]:
        """解析 Agent 响应中的 action"""
        try:
            pattern = r'"thoughts🤔"(.*?)"actions🚀"(.*?)"value👉"'
            matches = re.findall(pattern, response, re.DOTALL)

            if matches:
                actions_str = matches[0][1].strip()
                try:
                    actions = json.loads(actions_str)
                    if actions and len(actions) > 0:
                        return actions[0]
                    else:
                        print(f"  [_parse_action] Agent返回空actions，进入Agent VQA模式")
                        return {"API_name": None, "API_params": {}, "no_api": True}
                except:
                    try:
                        actions = json.loads(actions_str.replace("'", '"'))
                        if actions and len(actions) > 0:
                            return actions[0]
                        else:
                            print(f"  [_parse_action] Agent返回空actions，进入Agent VQA模式")
                            return {"API_name": None, "API_params": {}, "no_api": True}
                    except:
                        pass

            for api_name in API_NAME_TO_WORKER.keys():
                if api_name in response:
                    return {"API_name": api_name, "API_params": {}}

            return None

        except Exception:
            return None

    def _parse_value(self, response: str) -> str:
        """解析 Agent 响应中的 value 部分（取最后一个匹配）"""
        try:
            pattern = r'"value👉"(.*?)(?="thoughts🤔"|"actions🚀"|"value👉"|$)'
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return matches[-1].strip().strip('"').strip()
            return response
        except:
            return response
