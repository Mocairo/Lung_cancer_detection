import os
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from flask import current_app

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def _call_deepseek_api(prompt):
    """
    Calls the DeepSeek API with retry logic.
    It is an internal function, prefixed with an underscore.
    """
    api_key = current_app.config.get('DEEPSEEK_API_KEY')
    if not api_key or 'YOUR_DEEPSEEK_API_KEY_HERE' in api_key:
        current_app.logger.error("DeepSeek API Key is not configured.")
        return "错误：DeepSeek API密钥未配置。请检查服务器设置。"

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一名专业的肺部影像学医生，擅长将复杂的AI分析数据解读成患者能够理解的、友好的中文报告。"},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        current_app.logger.error(f"An error occurred while calling DeepSeek API: {e}")
        # Re-raise the exception to trigger tenacity's retry mechanism
        raise

def generate_ai_summary(analysis_result):
    """
    Generates a human-readable AI summary report from structured analysis data.
    """
    nodules = analysis_result.get('nodules', [])
    nodules_found = analysis_result.get('nodules_found', 0)

    if nodules_found == 0:
        return "本次AI分析未在您的CT扫描中发现明显的可疑结节。这通常是一个积极的信号，但并不能完全排除所有肺部异常的可能性。我们强烈建议您定期进行健康检查，并遵循医生的专业指导。请将此报告分享给您的医生，以获得全面的医疗评估。"

    # Sort nodules: primarily by malignancy probability, secondarily by volume.
    sorted_nodules = sorted(nodules, key=lambda n: (n.get('tumor_malignancy_prob', 0), n.get('volume_mm3', 0)), reverse=True)
    
    # Select the top 3 most significant nodules for the report
    top_k = 3
    main_nodules = sorted_nodules[:top_k]
    
    prompt_details = ""
    for i, nodule in enumerate(main_nodules):
        # Safely get and format numerical values to prevent errors
        diameter = nodule.get('diameter_mm')
        volume = nodule.get('volume_mm3')
        prob = nodule.get('tumor_malignancy_prob', 0)

        diameter_str = f"{diameter:.2f}" if isinstance(diameter, (int, float)) else "未知"
        volume_str = f"{volume:.1f}" if isinstance(volume, (int, float)) else "未知"
        
        prompt_details += f"  - 结节{i+1}：直径约 {diameter_str} mm, 体积约 {volume_str} mm³, AI评估其为恶性肿瘤的概率为 {prob:.1%}。\n"

    remaining_count = len(nodules) - top_k
    summary_info = f"本次扫描共发现 {nodules_found} 个可疑结节。"
    if remaining_count > 0:
        summary_info += f"其中，我们为您列出了最重要的 {top_k} 个。其余 {remaining_count} 个结节因尺寸较小或AI评估的风险较低而未逐一列出。"

    final_prompt = f"""
    请根据以下提供的一次肺部CT扫描的AI分析数据，为患者撰写一份专业的、有同理心的分析摘要报告。

    **核心发现数据：**
    {summary_info}
    
    **重点关注的结节详情：**
    {prompt_details}

    **撰写要求：**
    1.  **开篇总结**：首先，用通俗易懂的语言对总体情况进行总结。
    2.  **指标解读**：简要解释“直径”、“体积”和“恶性概率”这几个指标的临床参考意义。例如，尺寸是评估结节的重要依据之一，而恶性概率是AI根据其学习到的海量数据给出的一个风险评估参考值。
    3.  **针对性建议**：根据数据显示的风险（特别是最高恶性概率），为患者提供清晰、明确的下一步行动建议。例如，如果存在高概率结节，应强调“强烈建议”或“尽快”咨询医生。
    4.  **重要声明**：在报告末尾，必须用加粗或显著方式强调：“**本报告由AI生成，仅供参考，不能替代执业医师的最终诊断。请务必携带此完整报告咨询您的医生，由医生结合您的具体情况给出专业的医疗意见。**”
    """
    
    try:
        return _call_deepseek_api(final_prompt)
    except Exception as e:
        current_app.logger.error(f"Failed to generate AI summary after multiple retries: {e}")
        return "AI智能报告生成失败。由于网络或服务暂时不可用，我们无法生成详细的文字解读。这不影响您本次CT分析的核心数据结果，请您以报告中的结构化数据为准，并咨询您的医生。" 