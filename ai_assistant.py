# ai_assistant.py
# AI预测助手模块 - 基于通义千问
import streamlit as st
from datetime import datetime
import pandas as pd
import streamlit as st
import pandas as pd
from datetime import datetime

def init_chat_state():
    """初始化对话状态"""
    if "ai_chat_messages" not in st.session_state:
        st.session_state.ai_chat_messages = []
    if "ai_last_pred_summary" not in st.session_state:
        st.session_state.ai_last_pred_summary = None
    if "show_export_options" not in st.session_state:
        st.session_state.show_export_options = False


def reset_chat_session():
    """强制重置对话"""
    st.session_state.ai_chat_messages = []
    st.session_state.ai_last_pred_summary = None
    st.session_state.show_export_options = False


def build_pred_summary(pred_df, predict_channel, predict_value, predict_mode):
    """构建预测数据摘要"""
    if pred_df is None or pred_df.empty:
        return None

    try:
        min_flow = pred_df['预测流量'].min()
        max_flow = pred_df['预测流量'].max()
        mean_flow = pred_df['预测流量'].mean()

        max_idx = pred_df['预测流量'].idxmax()
        min_idx = pred_df['预测流量'].idxmin()

        if '显示时间' in pred_df.columns:
            peak_time = pred_df.loc[max_idx, '显示时间']
            valley_time = pred_df.loc[min_idx, '显示时间']
        else:
            peak_time = str(pred_df.loc[max_idx, '时间'])
            valley_time = str(pred_df.loc[min_idx, '时间'])

        if '拥堵等级' in pred_df.columns:
            congestion_counts = pred_df['拥堵等级'].value_counts().to_dict()
            normal_count = congestion_counts.get('正常通行', 0)
            light_count = congestion_counts.get('轻度拥堵', 0)
            moderate_count = congestion_counts.get('中度拥堵', 0)
            heavy_count = congestion_counts.get('严重拥堵', 0)
        else:
            normal_count = light_count = moderate_count = heavy_count = 0

        pred_df['小时'] = pd.to_datetime(pred_df['时间']).dt.hour
        hour_avg = pred_df.groupby('小时')['预测流量'].mean().round(0).to_dict()

        summary = f"""
【预测数据摘要】
- 航道名称：{predict_channel}
- 预测时长：{predict_value}{'小时' if predict_mode == 'hour' else '天'}
- 预测时段数：{len(pred_df)} 个
- 流量范围：{min_flow:.0f} ~ {max_flow:.0f} 艘/小时
- 平均流量：{mean_flow:.0f} 艘/小时
- 高峰时段：{peak_time}（流量 {max_flow:.0f} 艘/小时）
- 低谷时段：{valley_time}（流量 {min_flow:.0f} 艘/小时）

【拥堵统计】
- 正常通行：{normal_count} 个时段
- 轻度拥堵：{light_count} 个时段
- 中度拥堵：{moderate_count} 个时段
- 严重拥堵：{heavy_count} 个时段

【各小时平均流量】
{hour_avg}
"""
        return summary
    except Exception as e:
        st.error(f"构建预测摘要失败: {str(e)}")
        return None


def _get_ai_answer(question, pred_summary, api_key_func, llm_model):
    """调用AI获取回答"""
    if pred_summary is None:
        return "⚠️ 暂无有效的预测数据，请先运行预测"

    try:
        import dashscope
        from dashscope import Generation

        api_key = api_key_func()
        if not api_key:
            return "❌ 请先在左侧边栏配置通义千问API Key"

        dashscope.api_key = api_key

        prompt = f"""{pred_summary}

请基于以上预测数据回答用户的问题。回答要简洁、专业。

用户问题：{question}
"""

        response = Generation.call(
            model=llm_model,
            prompt=prompt,
            temperature=0.7,
            max_tokens=800
        )

        if response.status_code == 200:
            answer = response.output.text.strip()
            if answer:
                return answer
            else:
                return "抱歉，我没有生成有效回答，请重试"
        else:
            return f"API调用失败：{response.message}"

    except Exception as e:
        return f"AI助手出错：{str(e)}"


def generate_full_report(pred_df, predict_channel, predict_value, predict_mode, chat_messages):
    """生成完整的预测报告"""
    if pred_df is None or pred_df.empty:
        return "暂无预测数据"

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    report = f"""# 基于多源数据的港口交通流预测与示警智能体预测报告

**生成时间：** {now}
**航道名称：** {predict_channel}
**预测时长：** {predict_value}{'小时' if predict_mode == 'hour' else '天'}

---

## 📊 预测统计摘要

| 指标 | 数值 |
|------|------|
| 预测时段数 | {len(pred_df)} 个 |
| 最小流量 | {pred_df['预测流量'].min():.0f} 艘/小时 |
| 最大流量 | {pred_df['预测流量'].max():.0f} 艘/小时 |
| 平均流量 | {pred_df['预测流量'].mean():.0f} 艘/小时 |

---

## 📈 拥堵分布

"""
    if '拥堵等级' in pred_df.columns:
        report += pred_df['拥堵等级'].value_counts().to_string()
    else:
        report += "无拥堵等级数据"

    report += f"""

---

## 📋 详细预测数据

| 时间 | 预测流量(艘/小时) | 拥堵等级 |
|------|-------------------|----------|
"""

    for _, row in pred_df.iterrows():
        time_str = row['显示时间'] if '显示时间' in row else str(row['时间'])
        flow = row['预测流量']
        level = row['拥堵等级'] if '拥堵等级' in row else '-'
        report += f"| {time_str} | {flow:.0f} | {level} |\n"

    if chat_messages:
        report += f"""

---

## 💬 AI 问答记录

"""
        for msg in chat_messages:
            role = "👤 用户" if msg["role"] == "user" else "🤖 AI助手"
            report += f"\n**{role}：**\n{msg['content']}\n"

    report += f"""

---

*报告由基于多源数据的港口交通流预测与示警智能体自动生成*
"""
    return report


def render_ai_assistant(pred_df, predict_channel, predict_value, predict_mode, api_key_func,
                        selected_model="qwen-turbo"):
    """渲染AI助手面板"""
    init_chat_state()

    current_summary = build_pred_summary(pred_df, predict_channel, predict_value, predict_mode)

    if current_summary != st.session_state.ai_last_pred_summary:
        reset_chat_session()
        st.session_state.ai_last_pred_summary = current_summary

    if current_summary is None:
        st.warning("⚠️ 预测数据异常，无法使用AI助手")
        return

    st.markdown("### 🤖 AI 预测助手")
    st.caption(f"基于通义千问大模型 | 当前模型: {selected_model}")

    # 快速提问按钮
    st.markdown("**💡 快速提问：**")
    quick_questions = [
        "什么时候最拥堵？",
        "什么时间适合通航？",
        "帮我总结拥堵情况",
        "哪个小时流量最高？",
        "平均流量是多少？"
    ]

    cols = st.columns(5)
    for i, q in enumerate(quick_questions):
        with cols[i]:
            if st.button(q, key=f"quick_q_{i}", use_container_width=True):
                st.session_state.ai_chat_messages.append({"role": "user", "content": q})
                with st.spinner("🤔 正在思考中..."):
                    answer = _get_ai_answer(q, current_summary, api_key_func, selected_model)
                st.session_state.ai_chat_messages.append({"role": "assistant", "content": answer})
                st.rerun()

    st.divider()

    # 用户输入框（放在上面）
    user_input = st.chat_input("💬 输入你的问题...")

    # 对话历史（放在下面）
    chat_container = st.container(height=300)
    with chat_container:
        for msg in st.session_state.ai_chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # 处理用户输入
    if user_input:
        st.session_state.ai_chat_messages.append({"role": "user", "content": user_input})
        with st.spinner("🤔 正在思考中..."):
            answer = _get_ai_answer(user_input, current_summary, api_key_func, selected_model)
        st.session_state.ai_chat_messages.append({"role": "assistant", "content": answer})
        st.rerun()

    st.divider()

    # 底部按钮 - 3个按钮
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🗑️ 清空对话", use_container_width=True):
            reset_chat_session()
            st.rerun()

    with col2:
        # 导出报告按钮 - 点击后显示两个选项
        if st.button("📥 导出报告", use_container_width=True):
            st.session_state.show_export_options = not st.session_state.show_export_options
            st.rerun()

        # 如果点击了导出报告，显示选项
        if st.session_state.show_export_options:
            report = generate_full_report(pred_df, predict_channel, predict_value, predict_mode,
                                          st.session_state.ai_chat_messages)

            col_export1, col_export2 = st.columns(2)
            with col_export1:
                st.download_button(
                    label="📄 TXT 格式",
                    data=report,
                    file_name=f"预测报告_{predict_channel}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    key="download_txt"
                )
            with col_export2:
                st.download_button(
                    label="📝 MD 格式",
                    data=report,
                    file_name=f"预测报告_{predict_channel}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    key="download_md"
                )
    with col3:
        if st.button("📄 查看报告", use_container_width=True):
            report = generate_full_report(pred_df, predict_channel, predict_value, predict_mode,
                                          st.session_state.ai_chat_messages)
            with st.expander("📊 完整预测报告", expanded=True):
                st.code(report, language="markdown")
            st.toast("✅ 报告已生成", icon="📄")

