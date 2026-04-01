"""
ChatCAD_R 超精简版 - 仅保留医学知识问答，不支持报告分析
支持多种 LLM: OpenAI, DeepSeek, Qwen 等
兼容 openai>=1.0.0
"""
import sys
import os
import json
from openai import OpenAI
from text2vec import SentenceModel
from engine_LLM.api import answer_quest, query_range


class ChatBotRAG:
    """超精简版 - 只做医学知识问答，支持多种 LLM"""
    
    def __init__(self, engine: str = "gpt-4o", api_key: str = None, base_url: str = None):
        """
        初始化 ChatBotRAG
        
        Args:
            engine: 模型名称，如 "gpt-4o", "deepseek-chat", "qwen-plus" 等
            api_key: API key
            base_url: API base URL (可选)
                - OpenAI: 默认 https://api.openai.com/v1
                - DeepSeek: https://api.deepseek.com/v1
                - Qwen (DashScope): https://dashscope.aliyuncs.com/compatible-mode/v1
                - 其他兼容 OpenAI 的服务
        """
        self.engine = engine
        self.api_key = api_key
        self.base_url = base_url
        
        # 自动识别 base_url
        if base_url is None:
            if "deepseek" in engine.lower():
                self.base_url = "https://api.deepseek.com/v1"
            elif "qwen" in engine.lower():
                self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            # OpenAI 使用默认值 (None 会使用 openai 默认)
        
        # 初始化 OpenAI 客户端 (兼容 openai>=1.0.0)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        msd_path = os.path.join(os.path.dirname(__file__), 'engine_LLM', 'dataset', 'msd_dict.json')
        
        print("[ChatCAD_R] 加载语义模型...")
        self.sent_model = SentenceModel()
        print("[ChatCAD_R] 加载知识库映射...")
        self.msd_dict = json.load(open(msd_path, 'r', encoding='utf-8'))
        print("[ChatCAD_R] 初始化完成！")
    
    def ret_local(self, query: str, mode: int = 1) -> str:
        """返回默沙东医学手册链接"""
        topic_range, [_, _] = query_range(self.sent_model, query, k=1, bar=0.0)
        if len(topic_range) == 0:
            return ""
        
        if mode == 0:  # 中文
            return "https://" + self.msd_dict[topic_range[0]]
        else:  # 英文
            return "https://" + self.msd_dict[topic_range[0]].replace('www.msdmanuals.cn', 'www.merckmanuals.com')
    
    def chat_with_llm(self, prompt: str) -> str:
        """调用 LLM API (支持 OpenAI、DeepSeek、Qwen 等，兼容 openai>=1.0.0)"""
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.client.chat.completions.create(
                model=self.engine,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[ChatCAD_R] LLM API 调用失败: {e}")
            return f"Error: {str(e)}"
    
    def chat(self, message: str, ref_record: str = "") -> str:
        """
        医学知识问答（不支持报告分析）
        
        Args:
            message: 用户问题
            ref_record: 历史对话
            
        Returns:
            str: 回答文本
        """
        # 提炼问题
        refine_prompt = "请根据以下内容概括患者的提问并对所涉及的疾病指出其全称：\n"
        refined_message = self.chat_with_llm(ref_record + '\n' + refine_prompt + message)
        
        # 语义匹配检索
        topic_range, [raw_topic, cos_sim] = query_range(
            self.sent_model, refined_message, k=5, bar=0.6
        )
        
        if len(topic_range) == 0:
            response = self.chat_with_llm(f"{ref_record}\nuser:**Answer in English**\n" + message)
            response += "\nNote: No definitive evidence was found in the Merck Manual Professional Edition."
            return response
        
        # 从知识库检索
        refine_prompt = "请根据以下内容概括患者的提问：\n"
        refined_message = self.chat_with_llm(ref_record + '\n' + refine_prompt + message)
        ret = answer_quest(refined_message, api_key=self.api_key, topic_base_dict=topic_range,
                          model=self.engine, base_url=self.base_url)
        
        if ret == None:
            response = self.chat_with_llm(f"{ref_record}\nuser:**Answer in English**\n" + message)
            response += "\nNote: No definitive evidence was found in the Merck Manual Professional Edition."
            return response
        
        query, knowledge = ret
        knowledge = knowledge.replace("\n\n", "\n")
        needed_site = self.ret_local(query, 1)
        
        # 去除知识前缀
        try:
            index = knowledge.index("：")
        except ValueError:
            index = -1
        knowledge = knowledge[index + 1:]
        
        # 结合知识生成回答
        chat_message = (
            f"{ref_record}\nuser:**answer in English*\n"
            f"请参考以下知识来解答病人的问题\"{message}\"并给出分析，请注意保持语句通顺\n"
            f"[{knowledge}]"
        )
        response = self.chat_with_llm(chat_message)
        response += f"\nNote: Relevant information is sourced from the Merck Manual Professional Edition. ({needed_site})"
        
        return response


def initialize_chatbot(api_key: str = None, engine: str = "gpt-4o", base_url: str = None):
    """
    初始化 ChatBotRAG 实例
    
    Args:
        api_key: API key (支持 OpenAI, DeepSeek, Qwen 等)
        engine: 模型名称，如 "gpt-4o", "deepseek-chat", "qwen-plus" 等
        base_url: API base URL (可选，会自动识别)
        
    Returns:
        ChatBotRAG 实例
        
    示例:
        # OpenAI
        chatbot = initialize_chatbot(api_key="sk-...", engine="gpt-4o")
        
        # DeepSeek
        chatbot = initialize_chatbot(api_key="sk-...", engine="deepseek-chat")
        
        # Qwen
        chatbot = initialize_chatbot(api_key="sk-...", engine="qwen-plus")
    """
    if api_key is None:
        # 尝试从环境变量获取
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            api_key = input("Please enter your API key: ")
    
    return ChatBotRAG(engine=engine, api_key=api_key, base_url=base_url)


def concat_history(message_history: list) -> str:
    """
    将消息历史列表转换为字符串格式
    
    Args:
        message_history: 消息历史列表，格式为 [{"role": "user", "content": "..."}, ...]
        
    Returns:
        格式化后的历史对话字符串
    """
    history_str = ""
    for msg in message_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            history_str += f"user: {content}\n"
        elif role == "assistant":
            history_str += f"assistant: {content}\n"
    return history_str.strip()


def RAG(question: str, api_key: str = None, chatbot=None, engine: str = "gpt-4o", base_url: str = None) -> str:
    """
    简化版 RAG - 只支持问答，不支持报告分析
    
    Args:
        question: 用户问题
        api_key: API key (支持 OpenAI, DeepSeek, Qwen 等)
        chatbot: 预初始化的 chatbot
        engine: 模型名称 (如果 chatbot 为 None)
        base_url: API base URL (可选)
        
    Returns:
        str: 回答文本
        
    示例:
        # 使用 OpenAI
        answer = RAG("什么是高血压？", api_key="sk-...", engine="gpt-4o")
        
        # 使用 DeepSeek
        answer = RAG("什么是高血压？", api_key="sk-...", engine="deepseek-chat")
        
        # 使用 Qwen
        answer = RAG("什么是高血压？", api_key="sk-...", engine="qwen-plus")
    """
    if not chatbot:
        if api_key is None:
            api_key = input("Please enter your API key: ")
        chatbot = ChatBotRAG(engine=engine, api_key=api_key, base_url=base_url)
    
    return chatbot.chat(question)


if __name__ == "__main__":
    import os
    
    # 选择 LLM 引擎
    print("支持的 LLM 引擎:")
    print("1. OpenAI (gpt-4o, gpt-3.5-turbo)")
    print("2. DeepSeek (deepseek-chat)")
    print("3. Qwen (qwen-plus, qwen-turbo)")
    
    engine_choice = input("\n选择引擎 (1/2/3，默认 1): ").strip() or "1"
    
    if engine_choice == "2":
        engine = "deepseek-chat"
        api_key = os.getenv("DEEPSEEK_API_KEY") or input("请输入 DeepSeek API key: ")
    elif engine_choice == "3":
        engine = "qwen-plus"
        api_key = os.getenv("DASHSCOPE_API_KEY") or input("请输入 Qwen/DashScope API key: ")
    else:
        engine = "gpt-4o"
        api_key = os.getenv("OPENAI_API_KEY") or input("请输入 OpenAI API key: ")
    
    print(f"\n使用引擎: {engine}")
    
    # 测试
    questions = [
        "什么是高血压？",
        "糖尿病患者需要注意什么？"
    ]
    
    for q in questions:
        print(f"\n问题: {q}")
        print("-" * 60)
        answer = RAG(q, api_key=api_key, engine=engine)
        print(f"回答: {answer}")
        print("=" * 60)
