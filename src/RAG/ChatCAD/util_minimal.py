"""
ChatCAD_R 工具函数 - 精简版
只保留 RAG 功能，支持多种 LLM (OpenAI, DeepSeek, Qwen 等)
"""
from chatbot_minimal import ChatBotRAG, concat_history, initialize_chatbot


def RAG(report: str, api_key: str = None, chatbot=None, engine: str = "gpt-4o", base_url: str = None):
    """
    Retrieval Augmented Generation - 基于检索的问答
    支持多种 LLM: OpenAI, DeepSeek, Qwen 等
    
    Args:
        report: 医学报告或问题文本
        api_key: API key (支持 OpenAI, DeepSeek, Qwen 等)
        chatbot: 预初始化的 chatbot 实例（可选）
        engine: 模型名称，如 "gpt-4o", "deepseek-chat", "qwen-plus" 等
        base_url: API base URL (可选，会自动识别)
    
    Returns:
        生成的回答
        
    示例:
        # 使用 OpenAI
        answer = RAG("什么是高血压？", api_key="sk-...", engine="gpt-4o")
        
        # 使用 DeepSeek
        answer = RAG("什么是高血压？", api_key="sk-...", engine="deepseek-chat")
        
        # 使用 Qwen
        answer = RAG("什么是高血压？", api_key="sk-...", engine="qwen-plus")
    """
    if not chatbot:
        chatbot = initialize_chatbot(api_key, engine=engine, base_url=base_url)
    
    if chatbot is None:
        return "Error: Failed to initialize chatbot"
    
    message_history = [{"role": "user", "content": report}]
    ref_record = concat_history(message_history)
    ans = chatbot.chat("", ref_record)
    
    if type(ans) == str:    # 普通问题
        return ans
    
    # 医学报告问题
    response, check, query, abnormality_check, [raw_topic, cos_sim], knowledge = ans
    return response


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
    
    # 测试用例
    questions = [
        "什么是肺炎？",
        "高血压患者需要注意什么？"
    ]
    
    for q in questions:
        print(f"\n问题: {q}")
        print("-" * 60)
        answer = RAG(q, api_key=api_key, engine=engine)
        print(f"回答: {answer}")
        print("=" * 60)
