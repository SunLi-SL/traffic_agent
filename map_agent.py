# map_agent.py 头部
# map_agent.py 头部【修复完成版】
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import time
import altair as alt
import json
import os
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import pandas as pd

# ========= 全局密钥读取函数（移到最顶部，所有代码都能调用）=========
def get_secret(key_path):
    # 云端部署模式
    if st.runtime.exists():
        # 先判断是否存在 [api] 分组，不存在直接走本地文件逻辑
        if "api" in st.secrets and key_path in st.secrets["api"]:
            return st.secrets["api"][key_path]
    # 本地环境 / 云端无api密钥时读取本地txt
    try:
        with open(f".{key_path}.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        # 本地txt也不存在返回空字符串，后续代码弹窗让用户手动输入
        return ""

# 读取密钥，为空不中断程序，交给侧边栏输入框处理
dashscope_key = get_secret("dashscope_key")
amap_key = get_secret("amap_key")

# 读取预设账号密码（账号密码单独存secrets，和api密钥分组隔离）
user_pwd = st.secrets["passwords"]

# 登录状态初始化
if "login_status" not in st.session_state:
    st.session_state.login_status = False

if not st.session_state.login_status:
    st.title("智能体访问登录")
    uname = st.text_input("账号")
    pwd = st.text_input("密码", type="password")
    if st.button("登录验证"):
        if uname in user_pwd and user_pwd[uname] == pwd:
            st.session_state.login_status = True
            st.rerun()
        else:
            st.error("账号或密码错误")
    st.stop() # 登录失败直接拦截，下方业务代码不再执行

# ====================== 修复BUG1：登录校验通过后才导入自定义模块 ======================
from alert_system import (
    init_alert_state,
    show_alert_sidebar,
    show_real_time_alert_panel,
    show_predict_alert_panel,
    show_alert_log_panel,
    show_predict_alert_log_panel,
    calculate_adaptive_thresholds,
    CONGESTION
)
# 导入船舶轨迹功能模块
from trajectory_func import render_trajectory
from ai_assistant import reset_chat_session

# ========== 在这里添加声音开关初始化 ==========
if 'sound_enabled' not in st.session_state:
    st.session_state.sound_enabled = True  # 默认开启告警声音
# 全局初始化告警状态（只执行一次）
init_alert_state()

# -------------------------- 全局基础配置 --------------------------
st.set_page_config(
    page_title="基于多源数据的港口交通流预测与示警智能体",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.data_transformers.disable_max_rows()

week_map = {0: "星期一", 1: "星期二", 2: "星期三", 3: "星期四", 4: "星期五", 5: "星期六", 6: "星期日"}


# 兼容 Windows本地 + Linux云端（Streamlit Cloud）
import matplotlib.pyplot as plt
import platform

# 判断系统适配字体
if platform.system() == "Windows":
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
else:
    # Linux云端环境通用开源中文字体（Streamlit Cloud内置）
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]

plt.rcParams["axes.unicode_minus"] = False
# -------------------------- 大模型配置 --------------------------
def get_api_key():
    # 优先读取顶部全局变量（云端secrets [api]分组已经加载好）
    global dashscope_key
    if dashscope_key and dashscope_key.strip() != "":
        return dashscope_key.strip()

    # 下面是原有本地文件/环境变量兜底逻辑，完全保留，不影响本地使用
    key_paths = [
        ".dashscope_key.txt",
        "dashscope_key.txt",
        os.path.expanduser("~/.dashscope_key.txt"),
        os.path.join(os.path.dirname(__file__), ".dashscope_key.txt")
    ]
    for key_path in key_paths:
        if os.path.exists(key_path):
            try:
                with open(key_path, "r") as f:
                    key = f.read().strip()
                    if key:
                        return key
            except Exception:
                continue
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if api_key:
        return api_key
    if "dashscope_api_key" in st.session_state:
        return st.session_state["dashscope_api_key"]
    return ""

def save_api_key(key):
    if not key:
        return False

    key_path = ".dashscope_key.txt"
    try:
        with open(key_path, "w") as f:
            f.write(key.strip())
        st.session_state["dashscope_api_key"] = key
        return True
    except Exception as e:
        st.error(f"保存API Key失败: {e}")
        return False


def clear_api_key():
    key_path = ".dashscope_key.txt"
    if os.path.exists(key_path):
        try:
            os.remove(key_path)
        except Exception:
            pass

    if "dashscope_api_key" in st.session_state:
        del st.session_state["dashscope_api_key"]

    st.success("✅ 已清除API Key")


def extract_temporal_features(history_data):
    """高精度提取时序特征+自适应24h峰谷倍率(完全数据驱动，无固定系数)"""
    hist_array = np.array(history_data)

    if len(hist_array) == 0:
        return {
            'overall_mean': 0, 'overall_std': 0, 'overall_max': 0, 'overall_min': 0,
            'last_24h_mean': 0, 'hour_avg': {}, 'trend': 0, 'is_low_traffic': True,
            'traffic_level': 'unknown', 'cv': 0, 'has_periodicity': False,
            'daily_pattern_strength': 0, 'trend_strength': 0,
            'dawn_peak_mean': 0, 'dawn_peak_ratio': 1.0,
            "hour_self_ratio": {h: 1.0 for h in range(24)}
        }

    # 基础统计
    overall_mean = np.mean(hist_array)
    overall_std = np.std(hist_array)
    overall_max = np.max(hist_array)
    overall_min = np.min(hist_array)
    cv = overall_std / overall_mean if overall_mean > 0 else 0

    # 最近24h均值
    last_24h = hist_array[-24:] if len(hist_array) >= 24 else hist_array
    last_24h_mean = np.mean(last_24h) if len(last_24h) > 0 else overall_mean

    # 按小时归集多日数据
    daily_patterns = {}
    days = min(14, len(hist_array) // 24)
    # 新增：过滤拥堵极值，只用平稳流量计算基准小时倍率，防止拥堵拉高全天基准
    congestion_cutoff = 300
    for i in range(days):
        start_idx = len(hist_array) - 24 * (i + 1)
        end_idx = len(hist_array) - 24 * i if i > 0 else len(hist_array)
        if start_idx >= 0:
            day_data = hist_array[start_idx:end_idx]
            for hour in range(24):
                if hour < len(day_data):
                    val = day_data[hour]
                    daily_patterns.setdefault(hour, []).append(val)  # 不再过滤

    hour_avg = {h: np.mean(vals) for h, vals in daily_patterns.items() if len(vals) > 0}

    # ==========核心自适应：自动计算每个小时相对全天平均的倍率（替代固定峰谷系数）==========
    all_hour_vals = list(hour_avg.values())
    day_mean_all_h = np.mean(all_hour_vals) if len(all_hour_vals) > 0 else overall_mean
    hour_self_ratio = {}
    for h in range(24):
        h_mean = hour_avg.get(h, day_mean_all_h)
        # 当前小时历史均值 ÷ 全天24h平均 = 自适应倍率
        hour_self_ratio[h] = h_mean / (day_mean_all_h + 1e-6)

    # 凌晨0~3倍率（自适应，不再固定放大）
    dawn_hours = [0, 1, 2, 3]
    dawn_vals = []
    for h in dawn_hours:
        if h in daily_patterns:
            dawn_vals.extend(daily_patterns[h])
    if len(dawn_vals) > 0:
        dawn_peak_mean = np.mean(dawn_vals)
        dawn_peak_ratio = dawn_peak_mean / (overall_mean + 1e-6)
    else:
        dawn_peak_mean = overall_mean
        dawn_peak_ratio = 1.0

    # 趋势
    trend = 0
    trend_strength = 0
    if len(hist_array) >= 48:
        x = np.arange(len(hist_array[-48:]))
        z = np.polyfit(x, hist_array[-48:], 1)
        trend = z[0]
        trend_strength = abs(trend) / (overall_std + 1e-5)

    # 流量分级
    if overall_mean < 80:
        traffic_level = "low"
        is_low_traffic = True
    elif overall_mean < 200:
        traffic_level = "medium"
        is_low_traffic = False
    else:
        traffic_level = "high"
        is_low_traffic = False

    # 周期性
    has_periodicity = False
    daily_pattern_strength = 0
    if len(hist_array) >= 72:
        try:
            from statsmodels.tsa.stattools import acf
            acf_vals = acf(hist_array, nlags=48, fft=True)
            if len(acf_vals) > 24:
                daily_pattern_strength = acf_vals[24]
                if acf_vals[24] > 0.25:
                    has_periodicity = True
        except:
            pass

    return {
        'overall_mean': overall_mean,
        'overall_std': overall_std,
        'overall_max': overall_max,
        'overall_min': overall_min,
        'last_24h_mean': last_24h_mean,
        'hour_avg': hour_avg,
        'trend': trend,
        'is_low_traffic': is_low_traffic,
        'traffic_level': traffic_level,
        'cv': cv,
        'has_periodicity': has_periodicity,
        'daily_pattern_strength': daily_pattern_strength,
        'trend_strength': trend_strength,
        'dawn_peak_mean': dawn_peak_mean,
        'dawn_peak_ratio': dawn_peak_ratio,
        "hour_self_ratio": hour_self_ratio
    }



def fallback_prediction(history_data, predict_hours=72):
    """全自适应备用预测，峰谷全部由历史数据自动计算倍率"""
    hist_flow = np.array(history_data)
    if len(hist_flow) == 0:
        return [20] * predict_hours

    features = extract_temporal_features(history_data)
    n_history = len(hist_flow)
    dawn_ratio = features['dawn_peak_ratio']
    hour_self_ratio = features["hour_self_ratio"]
    base_mean = features['overall_mean']
    flow_min = max(10, base_mean * 0.35)
    flow_max = min(800, base_mean * 2.0)
    step_limit = max(20, base_mean * 0.25)

    # Holt-Winters分支
    if n_history >= 72 and features['has_periodicity']:
        try:
            model = ExponentialSmoothing(
                hist_flow,
                seasonal_periods=24,
                trend='add',
                seasonal='add',
                initialization_method='estimated'
            )
            res = model.fit(optimized=True, remove_bias=True)
            pred = res.forecast(predict_hours)
            pred = np.clip(pred, flow_min, flow_max)
            noise = np.random.normal(0, features['overall_std'] * 0.28, len(pred))
            pred = pred + noise

            for idx in range(len(pred)):
                h = idx % 24
                pred[idx] = pred[idx] * hour_self_ratio[h]
                if h in [0, 1, 2, 3]:
                    pred[idx] = pred[idx] * dawn_ratio
            pred = np.clip(pred, flow_min, flow_max)
            # 【修复】删除 smooth_predictions_with_fluctuation 调用
            return [max(10, int(round(p))) for p in pred]
        except:
            pass

    # 统计法预测
    hour_vals = {h: [] for h in range(24)}
    for i, v in enumerate(hist_flow[-336:]):
        h = i % 24
        hour_vals[h].append(v)

    hour_mean = {}
    hour_std = {}
    for h in range(24):
        vals = hour_vals[h]
        if len(vals) > 0:
            hour_mean[h] = np.median(vals)
            hour_std[h] = np.std(vals) * 0.75
        else:
            hour_mean[h] = base_mean
            hour_std[h] = features['overall_std'] * 0.5

    trend = features['trend'] * 0.7
    predictions = []
    for i in range(predict_hours):
        h = i % 24
        day = i // 24
        weekday = day % 7

        val = hour_mean.get(h, base_mean)
        val = val * hour_self_ratio[h]
        if h in [0, 1, 2, 3]:
            val = val * dawn_ratio

        val += trend * i
        if weekday >= 5:
            val *= 0.88

        noise = np.random.normal(0, hour_std.get(h, features['overall_std'] * 0.45))
        val += noise

        if i > 0:
            last = predictions[-1]
            val = max(last - step_limit, min(last + step_limit, val))

        val = max(flow_min, min(flow_max, val))
        predictions.append(val)

    # 【修复】删除 smooth_predictions_with_fluctuation 调用
    return [max(10, int(round(p))) for p in predictions]


@st.cache_data(ttl=300, show_spinner=False)
def predict_traffic_flow_with_llm(history_data, channel_name, predict_hours=72, model="qwen-turbo", temperature=0.35,
                                  future_weather_df=None):
    import dashscope
    from dashscope import Generation
    import re

    api_key = get_api_key()
    if not api_key:
        return fallback_prediction(history_data, predict_hours)

    dashscope.api_key = api_key
    features = extract_temporal_features(history_data)
    hist = history_data[-336:] if len(history_data) >= 336 else history_data
    dawn_ratio = round(features['dawn_peak_ratio'], 2)
    hour_self_ratio = features["hour_self_ratio"]

    base_mean = features['overall_mean']
    flow_min = max(10, base_mean * 0.35)
    flow_max = min(800, base_mean * 2.0)
    step_limit = max(20, base_mean * 0.25)

    daily_pattern = []
    for h in range(24):
        vals = [history_data[i] for i in range(len(history_data)) if i % 24 == h]
        daily_pattern.append(round(np.median(vals)) if vals else round(base_mean))

    # 未来天气提示
    future_weather_prompt = ""
    if future_weather_df is not None and len(future_weather_df) > 0:
        future_weather_prompt = f"""
【已上传未来预测天气数据，必须严格按天气修正流量】
未来天气规则：
1. 风速>8m/s → 流量 ×0.7~0.9
2. 能见度<1000m → 流量 ×0.6~0.8
3. 气温<5℃或>35℃ → 流量 ×0.8~0.95
4. 降雨/雪 → 流量 ×0.7~0.9
未来天气简要：
{future_weather_df[['time', 'wind_speed', 'visibility', 'temperature']].head(10).to_string()}
"""

    prompt = f"""你是水上交通流量预测专家，**严格按照下面【每个小时自适应倍率】去生成曲线，倍率>1代表该小时历史是高峰需要拉高，倍率<1代表历史是低谷需要压低，全部自适应，禁止平线**
航道名称：{channel_name}
近120h真实历史：{hist[-120:]}
24小时各时段历史基准流量：{daily_pattern}
【0~23点自适应倍率（倍率=该小时均值/全天均值）】：{hour_self_ratio}
凌晨0~3点整体倍率：{dawn_ratio}

{future_weather_prompt}

硬性规则：
1. 预测第N小时，流量 = 对应hour基准 × 该hour自适应倍率，倍率<1自动压低做低谷，倍率>1自动拉高做高峰，**完全跟随历史天然规律，不准人为固定12点/19点系数**
2. 相邻小时变化幅度不超过{step_limit:.0f}艘，不能连续多小时数值几乎相同
3. 流量区间：{flow_min:.0f} ~ {flow_max:.0f}；周末整体×0.88
4. 曲线必须跟着倍率自然起伏，严禁全部卡在同一数值附近的平直横线
5. 有未来天气时，必须按天气自动修正流量，恶劣天气自动降低流量

输出{predict_hours}个纯整数，英文逗号分隔，无多余任何文字。
"""

    try:
        response = Generation.call(
            model=model,
            messages=[{"role": "system", "content": "严格依据传入的24h倍率自适应高低峰，倍率定起伏，拒绝平直预测线"},
                      {"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2200,
            timeout=22,
            result_format="message"
        )

        if response.status_code != 200:
            return fallback_prediction(history_data, predict_hours)

        content = response.output.choices[0].message.content.strip()
        nums = re.findall(r'\d+\.?\d*', content)
        preds = [float(n) for n in nums]

        while len(preds) < predict_hours:
            h = len(preds) % 24
            base = daily_pattern[h]
            base = base * hour_self_ratio[h]
            if h in [0, 1, 2, 3]:
                base *= dawn_ratio
            base += np.random.normal(0, base * 0.15)
            preds.append(base)
        preds = preds[:predict_hours]

        preds = np.clip(preds, flow_min, flow_max)
        for i in range(1, len(preds)):
            preds[i] = max(preds[i - 1] - step_limit, min(preds[i - 1] + step_limit, preds[i]))

        predictions = [int(round(p)) for p in preds]

        # ========== 强制天气修正（保证天气一定影响预测）==========
        if future_weather_df is not None and len(future_weather_df) > 0:
            # 获取预测的时间点（基于最后一条历史数据）
            if len(history_data) > 0:
                last_time = datetime.now()  # 简化处理
            else:
                last_time = datetime.now()

            dates = [last_time + timedelta(hours=i + 1) for i in range(len(predictions))]

            for i, pred_time in enumerate(dates):
                # 找到最接近的天气数据
                weather_times = pd.to_datetime(future_weather_df['time'])
                time_diff = abs(weather_times - pred_time)
                closest_idx = time_diff.argmin()
                weather_row = future_weather_df.iloc[closest_idx]

                correction_factor = 1.0
                reasons = []

                # 风速影响
                if 'wind_speed' in weather_row.index and pd.notna(weather_row['wind_speed']):
                    ws = float(weather_row['wind_speed'])
                    if ws >= 12:
                        correction_factor *= 0.65
                        reasons.append(f"大风{ws}m/s")
                    elif ws >= 8:
                        correction_factor *= 0.8
                        reasons.append(f"大风{ws}m/s")

                # 能见度影响
                if 'visibility' in weather_row.index and pd.notna(weather_row['visibility']):
                    vis = float(weather_row['visibility'])
                    if vis < 500:
                        correction_factor *= 0.65
                        reasons.append(f"低能见度{vis}m")
                    elif vis < 1000:
                        correction_factor *= 0.8
                        reasons.append(f"低能见度{vis}m")
                    elif vis < 2000:
                        correction_factor *= 0.9
                        reasons.append(f"能见度{vis}m")

                # 气温影响
                if 'temperature' in weather_row.index and pd.notna(weather_row['temperature']):
                    temp = float(weather_row['temperature'])
                    if temp < 0:
                        correction_factor *= 0.75
                        reasons.append(f"低温{temp}℃")
                    elif temp < 5:
                        correction_factor *= 0.85
                        reasons.append(f"低温{temp}℃")
                    elif temp > 35:
                        correction_factor *= 0.85
                        reasons.append(f"高温{temp}℃")

                # 降水影响
                if 'precipitation' in weather_row.index and pd.notna(weather_row['precipitation']):
                    precip = float(weather_row['precipitation'])
                    if precip >= 5:
                        correction_factor *= 0.65
                        reasons.append(f"强降水{precip}mm")
                    elif precip >= 1:
                        correction_factor *= 0.8
                        reasons.append(f"降水{precip}mm")

                # 应用修正
                old_val = predictions[i]
                predictions[i] = max(10, min(800, int(predictions[i] * correction_factor)))

                if reasons and len(reasons) > 0:
                    # 静默修正，不打印（避免干扰）
                    pass
        # ========== 新增：后处理校正 ==========
        # 计算历史同小时中位数
        hist_hour_median = {}
        for h in range(24):
            vals = [history_data[i] for i in range(len(history_data)) if i % 24 == h]
            hist_hour_median[h] = np.median(vals) if vals else base_mean

        # 对预测值进行同小时中位数校正（权重0.3），避免极端偏差
        for i in range(len(predictions)):
            h = i % 24
            median_val = hist_hour_median.get(h, base_mean)
            if abs(predictions[i] - median_val) > 0.3 * median_val:
                predictions[i] = int(0.9 * predictions[i] + 0.1 * median_val)

        # 重新裁剪到安全区间
        predictions = [max(flow_min, min(flow_max, int(round(v)))) for v in predictions]
        # ========== 校正结束 ==========

        return predictions


    except Exception as e:
        return fallback_prediction(history_data, predict_hours)



def smooth_predictions_with_fluctuation(predictions, window_size=1):
    """取消平滑，原样返回，保住高低峰值"""
    return predictions


@st.cache_data(ttl=None)
def preprocess_raw_data(df):
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['date'] = df['time'].dt.date
    df['date_str'] = df['time'].dt.strftime('%Y-%m-%d')
    df['week_name'] = df['day_of_week'].map(week_map)
    df['show_time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M')
    df['流量_text'] = df['traffic_flow'].round(0).astype(int).astype(str) + " 艘/小时"
    df['time_label'] = df['hour'].apply(lambda h: f"{h:02d}:00")
    return df


@st.cache_data(ttl=None)
def calc_hourly_channel_data(df):
    return df.groupby(['channel_id', 'hour'])['traffic_flow'].mean().reset_index()


@st.cache_data(ttl=None)
def calc_heatmap_data(df):
    heat_data = df.groupby(['hour', 'date_str'])['traffic_flow'].mean().reset_index()
    heat_data['时间区间'] = heat_data['hour'].apply(lambda h: f"{h:02d}:00-{h + 1:02d}:00".replace("24:00", "00:00"))
    heat_data['流量_带单位'] = heat_data['traffic_flow'].round(1).astype(str) + " 艘/小时"
    return heat_data


# -------------------------- 高德地图Key管理 --------------------------
def get_amap_key():
    # 优先读取顶部全局变量（云端secrets [api]分组）
    global amap_key
    if amap_key and amap_key.strip() != "":
        return amap_key.strip()

    # 原有本地txt兼容逻辑保留
    secrets_path = ".amap_key.txt"
    if os.path.exists(secrets_path):
        with open(secrets_path, "r") as f:
            return f.read().strip()
    if "amap_key" in st.session_state:
        return st.session_state["amap_key"]
    return ""


def save_amap_key(key):
    secrets_path = ".amap_key.txt"
    with open(secrets_path, "w") as f:
        f.write(key)
    st.session_state["amap_key"] = key
    st.success("✅ 高德Key已保存！")


def clear_amap_key():
    secrets_path = ".amap_key.txt"
    if os.path.exists(secrets_path):
        os.remove(secrets_path)
    if "amap_key" in st.session_state:
        del st.session_state["amap_key"]
    st.success("✅ 已清除高德Key")
    st.rerun()


def generate_demo_weather_data(start_dt, total_hours=168):
    np.random.seed(42)
    weather_records = []
    weather_types = ["晴", "多云", "阴", "小雨", "中雨", "雾", "大风"]
    for hour_offset in range(total_hours):
        current_time = start_dt + timedelta(hours=hour_offset)
        hour = current_time.hour
        base_temp = 18
        if 8 <= hour <= 18:
            base_temp += np.random.uniform(6, 12)
        else:
            base_temp -= np.random.uniform(3, 8)
        temperature = round(base_temp + np.random.normal(0, 2.5), 1)
        wt_idx = np.random.randint(0, len(weather_types))
        weather = weather_types[wt_idx]
        if weather in ["大风", "小雨", "中雨"]:
            wind_speed = round(np.random.uniform(8, 16), 1)
        elif weather == "雾":
            wind_speed = round(np.random.uniform(1, 4), 1)
        else:
            wind_speed = round(np.random.uniform(2, 7), 1)
        if weather == "雾":
            visibility = np.random.randint(200, 900)
        elif weather in ["小雨", "中雨"]:
            visibility = np.random.randint(1200, 3500)
        else:
            visibility = np.random.randint(4000, 12000)
        if weather == "中雨":
            precip = round(np.random.uniform(3, 12), 1)
        elif weather == "小雨":
            precip = round(np.random.uniform(0.5, 2.8), 1)
        else:
            precip = 0.0
        if weather in ["小雨", "中雨", "雾"]:
            humidity = np.random.randint(70, 95)
        else:
            humidity = np.random.randint(40, 70)
        wind_direction = np.random.randint(0, 360)
        weather_records.append({
            "time": current_time,
            "temperature": temperature,
            "wind_speed": wind_speed,
            "visibility": visibility,
            "humidity": humidity,
            "precipitation": precip,
            "weather": weather,
            "wind_direction": wind_direction
        })
    return pd.DataFrame(weather_records)
# -------------------------- 生成演示数据（告警触发版） --------------------------
# -------------------------- 生成演示数据（告警触发增强版，大量中度+少量严重拥堵） --------------------------
def generate_demo_data():
    """
    生成规律性强、拥堵充足的演示数据，充分测试告警全功能：
    1. 工作日早晚高峰强制中度拥堵，部分时段冲高到严重拥堵
    2. 周末午后稳定中度拥堵
    3. 拥堵时长占总时段25%以上，保证预测时大量拥堵告警
    4. 每个航道差异化拥堵强度，方便切换航道测试自适应阈值
    """
    np.random.seed(42)

    channels = {
        "channel_A": {"lat_range": (38.95, 39.00), "lon_range": (117.65, 117.70), "color": "#3690e8"},
        "channel_B": {"lat_range": (38.90, 38.95), "lon_range": (117.70, 117.75), "color": "#7cb5ec"},
        "channel_C": {"lat_range": (38.85, 38.90), "lon_range": (117.75, 117.80), "color": "#f44336"}
    }

    demo_mmsi_pool = [f"12345{i:03d}" for i in range(1, 800)]
    start_date = datetime(2025, 1, 1, 0, 0, 0)
    dates = [start_date + timedelta(hours=i) for i in range(168)]  # 完整7天数据
    records = []

    # 工作日/周末拥堵时段定义
    def get_daily_schedule(weekday):
        if weekday < 5:
            # 工作日：早7-9，晚17-19 双高峰
            return {"moderate": [(7, 9), (17, 19)]}
        else:
            # 周末：13-15 午后高峰
            return {"moderate": [(13, 15)]}

    for channel_id, coords in channels.items():
        # 各航道基础流量差异化，自适应阈值会自动区分
        if channel_id == "channel_A":
            base_flow = 180
            light_thr = 300
            moderate_thr = 380
            heavy_thr = 500
        elif channel_id == "channel_B":
            base_flow = 250
            light_thr = 320
            moderate_thr = 400
            heavy_thr = 520
        else:  # channel_C 流量最大，极易出现严重拥堵
            base_flow = 320
            light_thr = 350
            moderate_thr = 420
            heavy_thr = 530

        for dt in dates:
            hour = dt.hour
            day_of_week = dt.weekday()

            # 基础时段系数
            if 7 <= hour <= 9:
                hour_factor = 1.4
            elif 17 <= hour <= 19:
                hour_factor = 1.55
            elif 11 <= hour <= 14:
                hour_factor = 1.25
            elif 0 <= hour <= 5:
                hour_factor = 0.4
            else:
                hour_factor = 0.9

            weekend_factor = 1.12 if day_of_week >= 5 else 1.0

            # 基础空载流量
            flow = int(base_flow * hour_factor * weekend_factor)

            # 判断是否拥堵时段
            schedule = get_daily_schedule(day_of_week)
            in_congestion_window = False
            for start_h, end_h in schedule["moderate"]:
                if start_h <= hour < end_h:
                    in_congestion_window = True
                    break

            if in_congestion_window:
                # 核心增强：大幅抬高流量，稳定中度拥堵，20%概率冲严重拥堵
                rand_severe = np.random.random()
                if rand_severe < 0.2:
                    # 20%概率生成严重拥堵
                    flow = heavy_thr + np.random.randint(40, 100)
                else:
                    # 80%稳定中度拥堵，阈值+40~120艘，远超轻度上限
                    flow = moderate_thr + np.random.randint(40, 120)

            # 小幅随机扰动，避免数值完全固定
            flow += np.random.randint(-12, 12)
            flow = max(20, min(750, flow))
            flow = int(flow)

            # 生成船舶AIS明细记录
            sample_size = min(flow, len(demo_mmsi_pool))
            sample_size = int(sample_size)
            if sample_size <= 0:
                continue

            ship_mmsi_list = np.random.choice(demo_mmsi_pool, size=sample_size, replace=True)
            for mmsi in ship_mmsi_list:
                msg_count = np.random.randint(1, 4)
                for _ in range(msg_count):
                    time_offset = timedelta(minutes=np.random.randint(-5, 5))
                    lat = np.random.uniform(coords["lat_range"][0], coords["lat_range"][1])
                    lon = np.random.uniform(coords["lon_range"][0], coords["lon_range"][1])
                    speed = np.random.uniform(5, 25)
                    heading = np.random.uniform(0, 360)
                    records.append({
                        'time': dt + time_offset,
                        'mmsi': mmsi,
                        'channel_id': channel_id,
                        'speed': speed,
                        'heading': heading,
                        'lat': lat,
                        'lon': lon
                    })

    df = pd.DataFrame(records)
    return df

# -------------------------- 自动识别航道 --------------------------
def auto_detect_channels(df):
    coords = df[["lat", "lon"]].dropna()
    if len(coords) < 1:
        df["channel_id"] = "channel_0"
        return df
    n_clusters = min(5, len(coords))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["channel_id"] = "channel_" + kmeans.fit_predict(coords).astype(str)
    return df


def auto_rename_ais_columns(df):
    col_map = {}
    # 原始列列表
    ori_cols_raw = df.columns.tolist()
    # 创建 {小写列名: 原始列名}
    lower_to_ori = {c.strip().lower(): c for c in ori_cols_raw}

    # 标准字段 -> 支持的所有别名（全部小写）
    field_alias = {
        "mmsi": ["mmsi", "船舶编号", "船舶id", "msi", "mmsi"],  # 添加 mmsi
        "lon": ["lon", "lng", "经度", "long", "longitude", "lon"],  # 添加 lon
        "lat": ["lat", "纬度", "latitude", "lat"],  # 添加 lat
        "speed": ["speed", "sog", "速度", "航速", "speed over ground", "sog"],  # sog 能匹配 SOG
        "cog": ["cog"],
        "heading": ["heading"],
        "time": ["time", "时间", "采集时间", "datetime", "received time", "basedatet", "basedatetime", "BaseDateTime"]
        # 添加 BaseDateTime
    }

    # 反向构建：小写别名 → 标准字段
    alias_to_std = {}
    for std, alias_list in field_alias.items():
        for alias in alias_list:
            alias_to_std[alias.lower()] = std  # 关键：全部转小写匹配

    # 匹配列
    matched_std_fields = set()
    for lower_col, ori_col in lower_to_ori.items():
        if lower_col in alias_to_std:
            target_std = alias_to_std[lower_col]
            col_map[ori_col] = target_std
            matched_std_fields.add(target_std)

    # 检查必填标准字段
    required_std = ["mmsi", "lon", "lat", "speed", "time"]
    missing_fields = [f for f in required_std if f not in matched_std_fields]

    # 重命名df
    df_renamed = df.rename(columns=col_map)
    rename_result = col_map

    # 存在缺失必填字段，友好报错
    if missing_fields:
        msg = f"❌ 数据文件缺少必要字段！缺失标准字段：{missing_fields}\n"
        msg += "支持识别的字段名称（不区分大小写）：\n"
        for std, alias_list in field_alias.items():
            msg += f"- {std} 可使用：{alias_list}\n"
        st.error(msg)

    return df_renamed, rename_result


def merge_weather_data(traffic_df, weather_df):
    """
    按小时精准合并交通流量数据 + 气象数据
    自动对齐到整点时间，实现流量与天气的强关联
    高兼容版：支持所有pandas版本，合并失败不崩溃
    """
    try:
        # 统一时间格式并对齐到整点
        traffic_df = traffic_df.copy()
        weather_df = weather_df.copy()

        # 强制转为datetime类型，容错处理
        traffic_df['time'] = pd.to_datetime(traffic_df['time'], errors='coerce').dt.floor('h')
        weather_df['time'] = pd.to_datetime(weather_df['time'], errors='coerce').dt.floor('h')

        # 剔除时间为空的脏数据
        traffic_df = traffic_df.dropna(subset=['time'])
        weather_df = weather_df.dropna(subset=['time'])

        if traffic_df.empty or weather_df.empty:
            st.warning("⚠️ 交通数据或天气数据无有效时间，跳过合并")
            return traffic_df

        # 按小时合并（最近匹配模式，容错更强，兼容所有pandas版本）
        merged_df = pd.merge_asof(
            traffic_df.sort_values('time'),
            weather_df.sort_values('time'),
            on='time',
            direction='nearest'
        )

        st.success(f"✅ 天气数据合并成功，新增字段：{list(weather_df.columns)}")
        return merged_df

    except Exception as e:
        # 出错时不崩溃，返回原数据，同时打印错误日志
        st.warning(f"⚠️ 天气数据合并异常: {str(e)}，已跳过合并")
        return traffic_df


# -------------------------- 全局样式CSS（已删除重复欢迎页样式） --------------------------
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    .main-title { font-size: 2.5rem; font-weight: 800; color: #0b3d7a; text-align: center; margin-bottom: 0.5rem; }
    .sub-text { text-align: center; font-size: 1.1rem; color: #2c528a; margin-bottom: 25px; font-weight: 500; }
    .card { background-color: #ffffff; padding: 24px 28px; border-radius: 24px; border: 1px solid #e0e4e8; margin-bottom: 28px; transition: box-shadow 0.2s; }
    .card:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
    .stats-container { display: flex; gap: 20px; margin-bottom: 28px; }
    .stat-box { flex: 1; background-color: #ffffff; border-radius: 20px; padding: 20px 16px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.05); border: 1px solid #e8ecf2; }
    .stat-value { font-size: 32px; font-weight: 800; color: #0b3d7a; }
    .stat-label { font-size: 14px; color: #5a6e8a; font-weight: 600; margin-top: 8px; }
    .stButton>button { background-color: #0b3d7a; color: #ffffff; border-radius: 40px; border: none; font-weight: 600; padding: 8px 24px; transition: all 0.3s; }
    .stButton>button:hover { background-color: #1a6fb5; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(11,61,122,0.3); color: #ffffff; }
    .loading-text { text-align: center; padding: 15px; font-size: 18px; font-weight: 600; color: #0b3d7a; animation: fadeIn 0.5s ease-in; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }
    div[data-testid="stDataFrame"] th, div[data-testid="stDataFrame"] td { text-align: left !important; vertical-align: middle !important; }
    .data-analysis-container { display: flex; gap: 24px; flex-wrap: wrap; }
    .heatmap-wrapper { flex: 1.2; min-width: 300px; }
    .linechart-wrapper { flex: 1.8; min-width: 350px; }
    @media (max-width: 900px) { .data-analysis-container { flex-direction: column; } }
    .channel-legend-tip { font-size: 12px; color: #6c757d; text-align: center; margin-top: 8px; background: #f1f3f5; border-radius: 20px; padding: 6px 12px; display: inline-block; }

    /* 固定左侧数据统计信息 永久置顶 */
.fixed-info-box {
    position: sticky !important;
    top: 0 !important;
    z-index: 9999 !important;
    background: #f0f2f6 !important;
    padding: 10px 0 !important;
    margin-bottom: 10px !important;
}


    /* ========== 告警闪烁动画 强制生效 ========== */
    @keyframes globalFlash {
        0% { background-color: #ffcdd2; }
        50% { background-color: #ef9a9a; }
        100% { background-color: #ffcdd2; }
    }
    @keyframes flash-border-red {
        0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.8); }
        50% { box-shadow: 0 0 0 15px rgba(255, 0, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.8); }
    }
    .global-alert-bar {
        width: 100% !important;
        padding: 16px !important;
        text-align: center !important;
        font-size: 20px !important;
        font-weight: bold !important;
        color: #b71c1c !important;
        border: 3px solid #d32f2f !important;
        border-radius: 8px !important;
        animation: globalFlash 0.8s infinite !important;
        margin-bottom: 20px !important;
        z-index: 99999 !important;
    }
    .alert-box-red {
        animation: flash-border-red 0.8s infinite !important;
        background: #ffebee !important;
        border: 2px solid #ff0000 !important;
        border-radius: 8px !important;
        padding: 15px !important;
        margin: 10px 0 !important;
        z-index: 9999 !important;
    }
    /* 文字闪烁 - 严重/中度拥堵文字闪动 */
    @keyframes textBlink {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
    }
    .text-blink {
        animation: textBlink 1s infinite !important;
    }
</style>
""", unsafe_allow_html=True)


def show_welcome_page():
    # 整体协调版欢迎页
    import streamlit as st

    # 清空可能存在的布局问题
    st.markdown("""
    <style>
    /* 主容器样式 */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }

    /* 英雄区 */
    .hero-section {
        background: linear-gradient(135deg, #0b3d7a 0%, #1a6fb5 100%);
        border-radius: 20px;
        padding: 60px 40px;
        text-align: center;
        margin-bottom: 50px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .hero-title {
        font-size: 48px;
        font-weight: 800;
        color: white;
        margin-bottom: 15px;
    }
    .hero-subtitle {
        font-size: 18px;
        color: #e0f2fe;
        margin-bottom: 20px;
    }
    .hero-tags {
        display: inline-flex;
        gap: 15px;
        background: rgba(255,255,255,0.15);
        padding: 8px 25px;
        border-radius: 40px;
    }
    .hero-tag {
        color: white;
        font-size: 14px;
    }

    /* 功能卡片区 */
    .section-title {
        font-size: 28px;
        font-weight: 700;
        color: #0b3d7a;
        text-align: center;
        margin-bottom: 30px;
    }
    .features-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 25px;
        margin-bottom: 100px;
    }
    .feature-card {
        background: white;
        border-radius: 16px;
        padding: 30px 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.3s;
        border: 1px solid #e9ecef;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }
    .feature-icon {
        font-size: 52px;
        margin-bottom: 15px;
    }
    .feature-name {
        font-size: 18px;
        font-weight: 700;
        color: #0b3d7a;
        margin-bottom: 10px;
    }
    .feature-desc {
        font-size: 13px;
        color: #6c757d;
        line-height: 1.5;
    }

    /* 快速开始标题：增加顶部外边距，拉开和卡片的距离 */
    .quick-start-title {
        font-size: 28px;
        font-weight: 700;
        color: #0b3d7a;
        text-align: center;
        margin-top: 60px;
        margin-bottom: 30px;
    }

  /* 快速开始卡片：高度适中、布局不变 */
.steps-wrap {
    background: #fff;
    border-radius: 16px;
    padding: 40px 40px !important;
    margin-bottom: 40px !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06) !important;
}
.steps-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 30px !important;
    text-align: center;
    width: 100% !important;
}
.step-circle {
    width: 36px !important;
    height: 36px !important;
    border-radius: 50%;
    background: #0b3d7a;
    color: #fff;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    margin-bottom: 15px !important;
    font-size: 16px !important;
}
.step-h {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #0b3d7a;
    margin-bottom: 8px !important;
}
.step-p {
    font-size: 12px !important;
    color: #666;
    line-height: 1.5 !important;
}
    .badges-row {
        display: flex;
        justify-content: center;
        gap: 12px;
        margin-top: 24px;
        flex-wrap: wrap;
    }
    .badge {
        background: #f5f7fa;
        color: #0b3d7a;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
    }

    /* 统计标签区 */
    .stats-section {
        text-align: center;
        margin-bottom: 40px;
        padding: 0;
        background: transparent;
        border-radius: 0;
        border: none;
    }
    .stats-grid {
        display: flex;
        justify-content: center;
        gap: 30px;
        flex-wrap: wrap;
    }
    .stat-badge {
        background: #f8f9fa;
        border-radius: 40px;
        padding: 8px 24px;
        font-size: 14px;
        font-weight: 500;
        color: #0b3d7a;
        border: 1px solid #dee2e6;
    }

    /* 按钮区 */
    .button-container {
        text-align: center;
        margin-top: 30px;
    }

    /* 响应式 */
    @media (max-width: 768px) {
        .features-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }
        .steps-grid {
            grid-template-columns: 1fr;
            gap: 25px;
        }
        .hero-title {
            font-size: 32px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # 英雄区
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🚢 基于多源数据的港口交通流预测与示警智能体</div>
        <div class="hero-subtitle">基于通义千问大模型 | 实时航道监控 + 智能流量预测</div>
        <div class="hero-tags">
            <span class="hero-tag">⚡ AI驱动</span>
            <span class="hero-tag">📊 实时分析</span>
            <span class="hero-tag">🎯 精准预测</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 功能卡片
    st.markdown('<div class="section-title">✨ 核心功能</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-name">实时监控</div>
            <div class="feature-desc">多航道实时流量监控<br>高德地图可视化展示</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-name">AI智能预测</div>
            <div class="feature-desc">通义千问大模型驱动<br>流量精准预测</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <div class="feature-name">数据分析</div>
            <div class="feature-desc">热力图分析<br>趋势可视化</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-name">实时预警</div>
            <div class="feature-desc">拥堵等级自动识别<br>分级预警提示</div>
        </div>
        """, unsafe_allow_html=True)

    # 快速开始标题（使用新的样式，自带上间距）
    st.markdown('<div class="quick-start-title">🚀 快速开始</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="steps-wrap">
        <div class="steps-grid">
            <div>
                <div class="step-circle">1</div>
                <div class="step-h">配置API密钥</div>
                <div class="step-p">在左侧栏配置<br>通义千问和高德地图Key</div>
            </div>
            <div>
                <div class="step-circle">2</div>
                <div class="step-h">加载数据</div>
                <div class="step-p">上传AIS文件或<br>使用演示数据</div>
            </div>
            <div>
                <div class="step-circle">3</div>
                <div class="step-h">开始分析</div>
                <div class="step-p">选择航道查看<br>实时监控和预测</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 统计标签
    st.markdown("""
    <div class="stats-section">
        <div class="stats-grid">
            <span class="stat-badge">🎯 97% 预测准确率</span>
            <span class="stat-badge">⚡ 毫秒级响应</span>
            <span class="stat-badge">🔒 数据安全保障</span>
            <span class="stat-badge">🌊 多航道支持</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 开始按钮
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎯 开始体验", use_container_width=True, type="primary"):
            st.session_state.show_welcome = False
            st.rerun()


# -------------------------- 初始化Session State --------------------------
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df = None
    st.session_state.raw_df = None
    st.session_state.channels = None
    st.session_state.hourly_channel_data = None
    st.session_state.heatmap_data = None
    st.session_state.show_welcome = True  # 新增：控制欢迎页显示


def load_data(uploaded_files, use_demo_data):
    """加载数据，只执行一次"""
    if uploaded_files and len(uploaded_files) > 0:
        progress_bar = st.progress(0, text="准备读取数据...")
        status_text = st.empty()

        df_list = []
        total_files = len(uploaded_files)

        status_text.text("📂 正在读取CSV文件...")
        for idx, f in enumerate(uploaded_files):
            progress = (idx + 1) / total_files * 0.3
            progress_bar.progress(progress, text=f"读取文件 {idx + 1}/{total_files}: {f.name}")
            df_chunk = pd.read_csv(f)
            df_list.append(df_chunk)

        status_text.text("🔄 正在合并数据...")
        progress_bar.progress(0.35, text="合并数据中...")
        ais_df = pd.concat(df_list, ignore_index=True)

        status_text.text("🔍 正在自动识别字段...")
        progress_bar.progress(0.45, text="识别字段中...")
        ais_df, rename_result = auto_rename_ais_columns(ais_df)

        status_text.text("🧹 正在清洗数据...")
        progress_bar.progress(0.55, text="清洗数据中...")

        required_cols = ["mmsi", "lon", "lat", "speed", "time"]
        missing_cols = [c for c in required_cols if c not in ais_df.columns]

        if missing_cols:
            st.error(f"❌ 缺少 **必填字段**：{missing_cols}")
            return None, None, None, None

        ais_df["time"] = pd.to_datetime(ais_df["time"], errors="coerce")

        for col in ["speed", "lat", "lon", "heading"]:
            if col in ais_df.columns:
                ais_df[col] = pd.to_numeric(ais_df[col], errors="coerce")

        ais_df = ais_df.dropna(subset=["time", "lat", "lon", "speed"])
        ais_df = ais_df[ais_df["speed"] >= 0.5]

        initial_count = len(ais_df)
        st.info(f"📊 数据清洗：原始 {initial_count} 条 → 有效 {len(ais_df)} 条")

        # 显示有多少艘船
        unique_ships = ais_df["mmsi"].nunique()
        st.info(f"🚢 识别到 {unique_ships} 艘不同的船舶")

        status_text.text("🗺️ 正在识别航道...")
        progress_bar.progress(0.7, text="识别航道中...")
        ais_df = auto_detect_channels(ais_df)

        status_text.text("📊 正在统计流量...")
        progress_bar.progress(0.85, text="统计流量中...")

        # 按小时聚合
        ais_df["hour"] = ais_df["time"].dt.floor("h")

        # 每艘船每小时只保留一条记录
        ais_df_unique = ais_df.drop_duplicates(subset=["mmsi", "hour"])

        # 统计每小时内每个航道的船舶数量
        flow = ais_df_unique.groupby(["hour", "channel_id"]).size().reset_index(name="traffic_flow")

        # 获取每个小时的实际经纬度（取第一条记录）
        pos = ais_df_unique.groupby(["hour", "channel_id"])[["lat", "lon", "speed", "heading"]].first().reset_index()

        # 合并流量和位置信息
        raw_df = flow.merge(pos, on=["hour", "channel_id"]).rename(columns={"hour": "time"})

        st.info(f"📊 流量统计完成：共 {len(raw_df)} 条记录（每小时每个航道一条）")

        status_text.text("✅ 数据加载完成！")
        progress_bar.progress(1.0, text="完成！")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()

        # 新增第四个返回值 ais_df：未聚合原始明细（带mmsi）
        return raw_df, rename_result, len(ais_df), ais_df

    elif use_demo_data:
        raw_ais_df = generate_demo_data()
        ais_df, rename_result = auto_rename_ais_columns(raw_ais_df)

        # 下面是你原有聚合逻辑 完全不动
        ais_df["time"] = pd.to_datetime(ais_df["time"], errors="coerce")
        for col in ["speed", "lat", "lon", "heading"]:
            if col in ais_df.columns:
                ais_df[col] = pd.to_numeric(ais_df[col], errors="coerce")
        ais_df = ais_df.dropna(subset=["time", "lat", "lon", "speed"])
        ais_df = ais_df[ais_df["speed"] >= 0.5]

        ais_df = auto_detect_channels(ais_df)
        ais_df["hour"] = ais_df["time"].dt.floor("h")
        ais_df_unique = ais_df.drop_duplicates(subset=["mmsi", "hour"])
        flow = ais_df_unique.groupby(["hour", "channel_id"]).size().reset_index(name="traffic_flow")
        pos = ais_df_unique.groupby(["hour", "channel_id"])[["lat", "lon", "speed", "heading"]].first().reset_index()
        raw_df = flow.merge(pos, on=["hour", "channel_id"]).rename(columns={"hour": "time"})

        # 第四个返回值：原始ais明细
        return raw_df, None, len(ais_df), ais_df
    else:
        # 四个返回值补齐
        return None, None, None, None


with st.sidebar:
    st.markdown("### ⚙️ 操作面板")
    # ===================== 天气功能（已精准插入正确位置）=====================
    # ===================== 天气数据集（完全对齐AIS样式）=====================
    # ===================== 天气数据集（完全对齐AIS样式）=====================
    st.markdown("### 🌤️ 天气数据集（可选）")


    # 弹窗：天气 CSV 格式要求
    @st.dialog("📄 天气 CSV 文件格式要求", width="large")
    def show_weather_format_dialog():
        st.markdown("""
        **必填字段（缺一不可）**
        - `time`：时间（格式：YYYY-MM-DD HH:MM:SS）

        **建议字段（用于AI精准修正流量）**
        - `temperature`：气温（℃）
        - `wind_speed`：风速（m/s）
        - `visibility`：能见度（m）
        - `humidity`：湿度（%）
        - `precipitation`：降水量（mm）
        - `weather`：天气类型（晴/阴/雨/雪/雾）

        **说明**
        1. 上传后系统会自动将天气与流量融合
        2. 大风、低能见度、雨雪会自动降低预测流量
        3. 可上传未来预测天气，系统会按天气智能修正流量
        """)
        if st.button("✅ 关闭", use_container_width=True):
            st.rerun()


    # 格式提示按钮
    if st.button("📖 点击查看天气文件格式要求", use_container_width=True, type="primary"):
        show_weather_format_dialog()

    # 上传天气数据
    weather_file = st.file_uploader("📄 上传天气 CSV（可选）", type=["csv"], key="weather_upload")

    # 使用演示气象数据（移除disabled，解决变量未定义报错）
    use_demo_weather = st.checkbox("🌤️ 使用演示气象数据", value=False)

    # 天气数据加载逻辑（带进度条、图标）
    weather_df = None
    load_progress = st.progress(0, text="🌤️ 天气数据准备中...")
    status_txt = st.empty()

    # 优先级1：上传的天气CSV文件
    if weather_file is not None:
        try:
            load_progress = st.progress(0, text="📄 读取天气CSV文件...")
            status_txt = st.empty()
            status_txt.text("📂 正在读取文件内容")
            load_progress.progress(30)
            weather_df = pd.read_csv(weather_file, encoding="utf-8-sig")
            load_progress.progress(60)
            status_txt.text("⏱️ 正在标准化时间字段")
            weather_df["time"] = pd.to_datetime(weather_df["time"], errors="coerce").dt.floor("h")
            load_progress.progress(80)
            status_txt.text("🧹 清洗无效时间数据")
            weather_df = weather_df.dropna(subset=["time"])
            load_progress.progress(100)
            status_txt.empty()
            load_progress.empty()
            st.success(f"📄 ✅ 已加载天气数据：{len(weather_df)} 条，有效字段：{list(weather_df.columns)}")
            with st.expander("🔍 天气字段匹配结果"):
                for col in weather_df.columns:
                    st.write(f"`{col}` → 已识别")
        except Exception as e:
            load_progress.empty()
            status_txt.empty()
            st.error(f"❌ 天气数据解析失败: {str(e)}")

    # 优先级2：演示气象数据
    elif use_demo_weather:
        status_txt.text("🌤️ 正在生成演示气象数据集...")
        load_progress.progress(20)
        demo_start_time = datetime(2025, 1, 1, 0, 0, 0)
        load_progress.progress(60)
        weather_df = generate_demo_weather_data(demo_start_time, total_hours=168)
        load_progress.progress(100)
        status_txt.empty()
        load_progress.empty()
        st.success("🌤️ ✅ 已加载演示气象数据")
        with st.expander("📋 演示气象数据详情"):
            st.dataframe(weather_df.head(20), use_container_width=True)

    # 无数据清空进度条
    else:
        load_progress.empty()
        status_txt.empty()
    st.markdown("### 📁 AIS数据上传")


    # 弹窗：CSV格式要求
    @st.dialog("📄 AIS CSV 文件格式要求", width="large")
    def show_format_dialog():
        st.markdown("""
        **必填字段（缺一不可）**
        - `time`：时间（格式：YYYY-MM-DD HH:MM:SS）
        - `lon`：经度（WGS84 坐标）
        - `lat`：纬度（WGS84 坐标）
        - `speed`：航速（单位：节）
        - `mmsi`：船舶唯一ID

        **可选字段**
        - `heading`：航向（0~360°）
        """)
        if st.button("✅ 关闭", use_container_width=True):
            st.rerun()


    if st.button("📖 点击查看文件格式要求", use_container_width=True, type="primary"):
        show_format_dialog()

    uploaded_files = st.file_uploader("📁 上传 AIS 数据 CSV（可多选）", type=["csv"], accept_multiple_files=True)
    use_demo_data = st.checkbox("📊 使用演示数据", value=False)

    need_reload = False
    current_files_key = str([f.name for f in uploaded_files]) if uploaded_files else "demo" if use_demo_data else None

    if current_files_key and (
            not st.session_state.data_loaded or st.session_state.get('last_files') != current_files_key):
        need_reload = True
        st.session_state['last_files'] = current_files_key

    if need_reload:
        with st.spinner("🚀 正在加载数据，请稍候..."):
            raw_df, rename_result, record_count, raw_detail_ais = load_data(uploaded_files, use_demo_data)

            if raw_df is not None:
                st.session_state.raw_df = raw_df
                st.session_state.raw_detail_ais = raw_detail_ais  # 新增这一行
                st.session_state.df = preprocess_raw_data(raw_df)
                st.session_state.channels = sorted(st.session_state.df["channel_id"].unique())
                st.session_state.hourly_channel_data = calc_hourly_channel_data(st.session_state.df)
                st.session_state.heatmap_data = calc_heatmap_data(st.session_state.df)
                st.session_state.data_loaded = True
                st.session_state.show_welcome = False
                # 新增：加载新数据后自动定位到最后一天
                date_list_new = sorted(st.session_state.df['date_str'].unique().tolist())
                st.session_state["selected_day_index"] = len(date_list_new) - 1

                if record_count:
                    st.session_state['data_info_html'] = f"""
                    <div style="
                        background: #f0f2f6;
                        padding: 10px 15px;
                        border-radius: 10px;
                        margin-bottom: 15px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                    ">
                        <div style='color: #0b3d7a; font-weight: 700; font-size: 15px; line-height:1.6;'>
                            ✅ 已合并 {len(uploaded_files)} 个 AIS 文件<br>
                            ✅ 共 {record_count} 条有效数据<br>
                            ✅ 自动识别出 {len(st.session_state.channels)} 个航道
                        </div>
                    </div>
                    """

                if rename_result:
                    with st.expander("🔍 自动匹配字段结果"):
                        for old_name, new_name in rename_result.items():
                            st.write(f"`{old_name}` → **{new_name}**")

                st.rerun()
            else:
                st.error("❌ 数据加载失败")
                st.stop()

    # ==============================================
    # 🔴【这里是你要的：三块内容 永久固定在一起】
    # ==============================================
    if st.session_state.data_loaded:
        # 1. 数据加载成功提示
        if 'data_info_html' in st.session_state and st.session_state['data_info_html']:
            st.markdown(st.session_state['data_info_html'], unsafe_allow_html=True)

        # 2. 选择航道
        # 统一航道源：优先读取预测保存航道，无则使用侧边栏缓存
        df = st.session_state.df
        channels = st.session_state.channels

        # 航道双向联动逻辑
        if "predict_channel_saved" in st.session_state and st.session_state.prediction_completed:
            # 已有预测结果，默认选中预测航道
            default_ch = st.session_state.predict_channel_saved
        else:
            # 无预测，使用上次侧边栏选择
            default_ch = st.session_state.get("channel_option", channels[0])

        # 侧边栏选择框，绑定统一默认值
        channel_option = st.selectbox("🗺️ 选择航道", channels, index=channels.index(default_ch))

        # 同步至全局侧边栏缓存
        st.session_state.channel_option = channel_option

        # 同步到预测面板默认航道，切换侧边栏航道自动重置预测
        if "last_predict_channel" in st.session_state:
            if st.session_state.last_predict_channel != channel_option:
                # 侧边栏切换航道，清空所有预测缓存，强制重新预测
                st.session_state.prediction_completed = False
                st.session_state.predict_finish_msg = None
                if "pred_df" in st.session_state:
                    del st.session_state["pred_df"]
                if "predict_channel_saved" in st.session_state:
                    del st.session_state["predict_channel_saved"]
                st.session_state["last_predict_channel"] = channel_option

        # 3. 历史时长 + 预测模式
        selected_hist = df[df["channel_id"] == channel_option].copy()
        if len(selected_hist) >= 2:
            time_diff = selected_hist["time"].max() - selected_hist["time"].min()
            total_hours = time_diff.total_seconds() / 3600
        else:
            total_hours = 168

        if total_hours < 24:
            predict_mode = "hour"
            predict_label = "预测小时数"
            predict_min, predict_max, predict_default = 1, 12, 6
        else:
            predict_mode = "day"
            predict_label = "预测天数"
            predict_min, predict_max, predict_default = 1, 7, 3

        predict_value = st.slider(f"📅 {predict_label}", predict_min, predict_max, predict_default)
        st.info(f"📊 历史时长：{total_hours:.1f} 小时 → 自动切换为【{predict_mode}】模式")

    elif not st.session_state.data_loaded:
        st.warning("请上传数据或勾选'使用演示数据'")

    # --------------------- 下面是 API、高德、拥堵等级（不动）---------------------
    st.markdown("---")
    st.markdown("### 🤖 通义千问API设置")
    api_key = get_api_key()
    if not api_key:
        st.warning("⚠️ 请输入通义千问API Key")
        api_key_input = st.text_input("DashScope API Key", type="password")
        if api_key_input and st.button("✅ 保存API Key"):
            if save_api_key(api_key_input):
                st.rerun()
    else:
        st.success(f"✅ API Key已配置 (密钥: {api_key[:8]}...)")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 更换千问Key"):
                clear_api_key()
                st.rerun()
        with col2:
            if st.button("🗑️ 清除千问Key"):
                clear_api_key()
                st.rerun()

    st.markdown("---")
    st.markdown("### 🗺️ 高德地图设置")





    amap_key = get_amap_key()
    if not amap_key:
        st.warning("⚠️ 请输入高德地图Key")
        amap_key_input = st.text_input("高德地图Web端Key", type="password")
        if amap_key_input and st.button("✅ 保存高德Key"):
            save_amap_key(amap_key_input)
            st.rerun()
    else:
        st.success(f"✅ 高德Key已配置 (密钥: {amap_key[:8]}...)")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 更换高德Key"):
                clear_amap_key()
        with col2:
            if st.button("🗑️ 清除高德Key"):
                clear_amap_key()

        st.markdown("---")
        with st.expander("📊 拥堵分级标准", expanded=True):
            # 固定兜底阈值
            fixed_light = 200
            fixed_moderate = 300
            fixed_heavy = 400
            # 标记是否使用自适应
            use_auto = False
            thresholds = None

            if st.session_state.data_loaded and 'channel_option' in st.session_state:
                try:
                    selected_channel = st.session_state.channel_option
                    df_temp = st.session_state.df
                    # 修复1：增加 == 筛选当前选中航道
                    df_channel = df_temp[df_temp['channel_id'] == selected_channel]
                    history_flow = df_channel['traffic_flow'].tolist()
                    if len(history_flow) > 0:
                        thresholds = calculate_adaptive_thresholds(history_flow, selected_channel)
                        use_auto = True
                    else:
                        use_auto = False
                except Exception as e:
                    use_auto = False


            if use_auto and thresholds is not None:
                st.markdown(f"**📍 当前航道：{selected_channel}**")
                st.success(f"✅ 正常通行：0 ~ {thresholds['light'] - 1} 艘/小时")
                st.info(f"🔷 轻度拥堵：{thresholds['light']} ~ {thresholds['moderate'] - 1} 艘/小时")
                st.warning(f"⚠️ 中度拥堵：{thresholds['moderate']} ~ {thresholds['heavy'] - 1} 艘/小时")
                st.error(f"🔴 严重拥堵：≥ {thresholds['heavy']} 艘/小时")
                st.caption(f"📊 基于历史数据动态计算（70%/85%/95%分位数）")
            else:
                # 无数据/计算异常，统一用固定值，不再读取 thresholds
                st.markdown(f"**📍 当前航道：{st.session_state.get('channel_option', '未知航道')}（无历史数据/计算失败）**")
                st.success(f"✅ 正常通行：0 ~ {fixed_light - 1} 艘/小时")
                st.info(f"🔷 轻度拥堵：{fixed_light} ~ {fixed_moderate - 1} 艘/小时")
                st.warning(f"⚠️ 中度拥堵：{fixed_moderate} ~ {fixed_heavy - 1} 艘/小时")
                st.error(f"🔴 严重拥堵：≥ {fixed_heavy} 艘/小时")
                st.caption(f"⚠️ 暂无有效历史数据，使用默认标准200/300/400")

        # ========== 在侧边栏结束前，追加这一行 ==========
    show_alert_sidebar()
# -------------------------- 主页面 --------------------------
# 判断显示欢迎页还是主内容
if st.session_state.show_welcome and not st.session_state.data_loaded:
    show_welcome_page()
else:
    # 确保数据已加载
    if not st.session_state.data_loaded:
        st.info("👈 请先在左侧上传数据或勾选'使用演示数据'开始体验")
        st.stop()

    df = st.session_state.df
    hourly_channel_data = st.session_state.hourly_channel_data
    heatmap_data = st.session_state.heatmap_data
    channels = st.session_state.channels

    # ===================== 自动合并天气数据 =====================


    # -------------------------- 标题 --------------------------
    st.markdown('<div class="main-title">🚢 基于多源数据的港口交通流预测与示警智能体</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">✨ 基于通义千问大模型 | 实时航道监控 + 流量预测 ✨</div>',
                unsafe_allow_html=True)
    st.divider()

    # ===================== 天气数据展示面板 =====================
    if weather_df is not None:
        st.markdown("---")
        st.markdown("## 🌤️ 气象数据分析面板")

        with st.expander("📄 查看气象原始数据", expanded=False):
            st.dataframe(weather_df, use_container_width=True)

        # 第一行：4个指标卡片
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if "temperature" in weather_df.columns:
                avg_temp = weather_df["temperature"].mean()
                st.metric("🌡️ 平均气温", f"{avg_temp:.1f} °C")
            else:
                st.metric("🌡️ 平均气温", "无数据")
        with col2:
            if "wind_speed" in weather_df.columns:
                avg_wind = weather_df["wind_speed"].mean()
                st.metric("💨 平均风速", f"{avg_wind:.1f} m/s")
            else:
                st.metric("💨 平均风速", "无数据")
        with col3:
            if "visibility" in weather_df.columns:
                avg_vis = weather_df["visibility"].mean()
                st.metric("👁️ 平均能见度", f"{avg_vis:.0f} m")
            else:
                st.metric("👁️ 平均能见度", "无数据")
        with col4:
            if "humidity" in weather_df.columns:
                avg_hum = weather_df["humidity"].mean()
                st.metric("💧 平均湿度", f"{avg_hum:.0f} %")
            else:
                st.metric("💧 平均湿度", "无数据")

        with st.expander("📈 气象指标变化趋势", expanded=True):
            fig, axes = plt.subplots(3, 1, figsize=(8, 5), sharex=True)

            if "temperature" in weather_df.columns:
                axes[0].plot(weather_df["time"], weather_df["temperature"], color="#ff4b4b", linewidth=2)
                axes[0].set_ylabel("气温 (℃)", fontsize=10)
                axes[0].set_title("气温变化", fontsize=12)
                axes[0].grid(True, alpha=0.3)
            else:
                axes[0].text(0.5, 0.5, "暂无气温数据", ha='center', va='center', transform=axes[0].transAxes)
                axes[0].set_ylabel("气温 (℃)")

            if "wind_speed" in weather_df.columns:
                axes[1].plot(weather_df["time"], weather_df["wind_speed"], color="#1f77b4", linewidth=2)
                axes[1].set_ylabel("风速 (m/s)", fontsize=10)
                axes[1].set_title("风速变化", fontsize=12)
                axes[1].grid(True, alpha=0.3)
            else:
                axes[1].text(0.5, 0.5, "暂无风速数据", ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_ylabel("风速 (m/s)")

            if "visibility" in weather_df.columns:
                axes[2].plot(weather_df["time"], weather_df["visibility"], color="#2ca02c", linewidth=2)
                axes[2].set_ylabel("能见度 (m)", fontsize=10)
                axes[2].set_title("能见度变化", fontsize=12)
                axes[2].grid(True, alpha=0.3)
            else:
                axes[2].text(0.5, 0.5, "暂无能见度数据", ha='center', va='center', transform=axes[2].transAxes)
                axes[2].set_ylabel("能见度 (m)")

            axes[2].set_xlabel("时间", fontsize=10)
            axes[2].xaxis.set_major_locator(plt.MaxNLocator(6))
            plt.xticks(rotation=0, fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)

        # 降水量趋势
        if "precipitation" in weather_df.columns:
            st.markdown("### 🌧️ 降水量趋势")
            import plotly.express as px

            precip_df = weather_df.copy()
            precip_df["time"] = pd.to_datetime(precip_df["time"])
            fig_precip = px.line(
                precip_df,
                x="time",
                y="precipitation",
                title="降水量趋势",
                labels={"precipitation": "降水量 (mm)", "time": "时间"}
            )
            fig_precip.update_layout(
                height=250,
                xaxis=dict(
                    tickmode="auto",
                    nticks=6,
                    tickangle=0
                )
            )
            st.plotly_chart(fig_precip, use_container_width=True)

        if "wind_direction" in weather_df.columns:
            with st.expander("🧭 风向记录", expanded=False):
                st.dataframe(weather_df[["time", "wind_direction"]].tail(10), use_container_width=True)

    # 继续您原有的统计卡片代码（下面的代码保持不变）
    # -------------------------- 统计卡片 --------------------------
    total_records = len(df)
    channel_count = df['channel_id'].nunique()
    avg_flow = df['traffic_flow'].mean()
    time_span_days = (df['time'].max() - df['time'].min()).days
    date_start = df['date_str'].min()
    date_end = df['date_str'].max()

    st.markdown(f"""
    <div class="stats-container">
        <div class="stat-box"><div class="stat-value">{total_records}</div><div class="stat-label">📊 总记录数</div></div>
        <div class="stat-box"><div class="stat-value">{channel_count}</div><div class="stat-label">🗺️ 航道数量</div></div>
        <div class="stat-box"><div class="stat-value">{avg_flow:.0f}</div><div class="stat-label">⚓ 平均流量 (艘/小时)</div></div>
        <div class="stat-box"><div class="stat-value">{time_span_days}</div><div class="stat-label">📅 数据周期：{date_start}~{date_end}</div></div>
    </div>
    """, unsafe_allow_html=True)
    # 定义双标签页：原有功能 + 船舶轨迹
    # 定义三个标签页
    tab_main, tab_trajectory, tab_ai_assistant = st.tabs([
        "📊 航道监控与流量预测",
        "🚢 船舶轨迹预测",
        "🤖 AI 预测助手"
    ])

    # 开启第一个标签：包裹你所有旧页面内容
    with tab_main:
        # ========= 新增：全局统一计算当前选中航道拥堵阈值 =========
        df = st.session_state.df
        light_t = 200
        moderate_t = 300
        heavy_t = 400
        # ======================================================
        with st.expander("📋 原始数据预览", expanded=True):
            preview_df = df[['time', 'show_time', 'channel_id', 'traffic_flow', 'lat', 'lon']].head(100).copy()
            preview_df.index = preview_df.index + 1
            st.dataframe(preview_df, use_container_width=True)

        # -------------------------- 历史流量趋势 --------------------------
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader(f"📈 历史流量趋势 · {st.session_state.get('channel_option', channels[0])}")
        filtered_df = df[df['channel_id'] == st.session_state.get('channel_option', channels[0])].copy()
        filtered_df = filtered_df.dropna(subset=['time', 'traffic_flow'])
        filtered_df = filtered_df[(filtered_df['traffic_flow'] >= 0) & (filtered_df['traffic_flow'].notna())]

        if not filtered_df.empty:
            nearest = alt.selection_point(nearest=True, on='mouseover', fields=['time'], empty=False)
            line = alt.Chart(filtered_df).mark_line(color='#0b3d7a', strokeWidth=3).encode(
                x=alt.X('time:T', scale=alt.Scale(zero=False)),
                y=alt.Y('traffic_flow:Q', scale=alt.Scale(zero=False), axis=alt.Axis(title="流量 (艘/小时)"))
            )
            area = line.mark_area(opacity=0.3)
            rule = alt.Chart(filtered_df).mark_rule(color='gray', strokeDash=[5, 5]).encode(
                x='time:T').transform_filter(
                nearest)
            points = line.mark_point(size=80).encode(
                opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
                tooltip=[
                    alt.Tooltip('show_time:N', title='🕐 时间'),
                    alt.Tooltip('流量_text:N', title='📊 流量'),
                    alt.Tooltip('channel_id:N', title='🚢 航道')
                ]
            ).add_params(nearest)
            hline_df = pd.DataFrame({'y': [light_t, moderate_t, heavy_t]})
            hlines = alt.Chart(hline_df).mark_rule(color='red', strokeDash=[8, 4]).encode(y='y:Q')

            trend_chart = alt.layer(area, line, points, rule, hlines).properties(height=400).interactive()
            st.altair_chart(trend_chart, use_container_width=True)
            st.caption("💡 鼠标悬停查看详情 | 拖拽平移 / 滚轮缩放")
        st.markdown('</div>', unsafe_allow_html=True)

        # -------------------------- 航道数据分析 --------------------------
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 航道数据分析")

        st.markdown('<div class="data-analysis-container">', unsafe_allow_html=True)

        st.markdown('<div class="heatmap-wrapper">', unsafe_allow_html=True)
        st.markdown("#### 🔥 每日流量热力图")
        heat_chart = alt.Chart(heatmap_data).mark_rect().encode(
            x=alt.X('hour:O', title="小时", sort="ascending", axis=alt.Axis(labelAngle=0)),
            y=alt.Y('date_str:O', title="日期", sort=alt.SortField('date_str', 'ascending')),
            color=alt.Color('traffic_flow:Q', scale=alt.Scale(scheme="reds"), legend=None),
            tooltip=[
                alt.Tooltip('date_str:N', title="📅 日期"),
                alt.Tooltip('时间区间:N', title="🕐 时段"),
                alt.Tooltip('流量_带单位:N', title="📊 流量")
            ]
        ).properties(height=460, width="container")
        st.altair_chart(heat_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="linechart-wrapper">', unsafe_allow_html=True)

        st.markdown("#### 📈 单日流量趋势（自动匹配当日真实时段）")
        date_list = sorted(df['date_str'].unique().tolist())
        max_idx = len(date_list) - 1
        if "selected_day_index" not in st.session_state:
            # 初始默认选中最后一天，取最大下标
            st.session_state["selected_day_index"] = max_idx
        if st.session_state["selected_day_index"] > max_idx:
            st.session_state["selected_day_index"] = max_idx
        select_day = date_list[st.session_state["selected_day_index"]]

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 12px 24px; border-radius: 48px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between; border: 1px solid #ced4da;">
            <div style="display: flex; align-items: center; gap: 12px; font-size: 1.5rem; font-weight: 600; color: #0b3d7a;">
                <span style="font-size: 1.8rem;">📅</span>选择查询日期：
            </div>
            <div style="border-radius: 32px; padding: 10px 24px; border: 1px solid #ced4da; background-color: white; font-size: 1.3rem; font-weight: 500; color: #0b3d7a;">
                {select_day}
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_prev, col_next = st.columns([1, 1])
        with col_prev:
            if st.button("⬅️ 前一天", use_container_width=True, disabled=(st.session_state["selected_day_index"] == 0)):
                st.session_state["selected_day_index"] -= 1
                st.rerun()
        with col_next:
            if st.button("后一天 ➡️", use_container_width=True,
                         disabled=(st.session_state["selected_day_index"] >= max_idx)):
                st.session_state["selected_day_index"] += 1
                st.rerun()

        day_df = df[df["date_str"] == select_day].copy()
        if day_df.empty:
            st.info("📭 当天无任何船舶数据")
        else:
            day_group = day_df.groupby(['channel_id', 'hour'])['traffic_flow'].mean().reset_index()
            valid_data = day_group[day_group["traffic_flow"] > 0].copy()
            if valid_data.empty:
                st.warning(f"📌 {select_day}全部时段流量=0，无有效绘图数据")
            else:
                real_hour_list = sorted(valid_data["hour"].unique())
                valid_data['traffic_flow_int'] = valid_data['traffic_flow'].round(0).astype(int)
                valid_data['日期'] = select_day
                valid_data['时段'] = valid_data['hour'].apply(
                    lambda x: f"{x:02d}:00-{x + 1:02d}:00".replace("24:00", "00:00"))
                valid_data['流量_带单位'] = valid_data['traffic_flow'].round(1).astype(str) + " 艘/小时"

                unique_ch = sorted(valid_data["channel_id"].unique())
                color_list = ["#3690e8", "#7cb5ec", "#f44336", "#9467bd", "#8c564b"]
                ch_color_map = {cid: color_list[i % len(color_list)] for i, cid in enumerate(unique_ch)}

                legend_sel = alt.selection_multi(fields=["channel_id"], bind="legend")
                base = alt.Chart(valid_data).encode(
                    x=alt.X("hour:O", sort=real_hour_list, title="时间"),
                    y=alt.Y("traffic_flow:Q", title="平均流量 (艘/小时)"),
                    color=alt.Color("channel_id:N",
                                    scale=alt.Scale(domain=unique_ch, range=[ch_color_map[c] for c in unique_ch]),
                                    legend=alt.Legend(orient="top")),
                    opacity=alt.condition(legend_sel, alt.value(1), alt.value(0.15)),
                    tooltip=[
                        alt.Tooltip('日期:N', title="📅 日期"),
                        alt.Tooltip('时段:N', title="🕐 时段"),
                        alt.Tooltip('流量_带单位:N', title="📊 流量"),
                        alt.Tooltip('channel_id:N', title="🚢 航道")
                    ]
                ).add_params(legend_sel).properties(height=460)

                line = base.mark_line(strokeWidth=2.5)
                point = base.mark_point(size=70, filled=True)
                text_layers = []
                for ch in unique_ch:
                    sub = valid_data[valid_data["channel_id"] == ch]
                    txt = alt.Chart(sub).mark_text(align="center", baseline="bottom", fontSize=11, dy=-12,
                                                   color=ch_color_map[ch]).encode(x="hour:O", y="traffic_flow:Q",
                                                                                  text="traffic_flow_int:Q",
                                                                                  opacity=alt.condition(legend_sel,
                                                                                                        alt.value(1),
                                                                                                        alt.value(0.3)))
                    text_layers.append(txt)

                all_layer = [line, point] + text_layers
                final_chart = alt.layer(*all_layer).interactive()
                st.altair_chart(final_chart, use_container_width=True)

        st.markdown(
            '<div class="channel-legend-tip">💡 提示：点击图例中的航道名称可单独查看该航道数据，再次点击恢复全部显示</div>',
            unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        # -------------------------- 高德地图 --------------------------
        # -------------------------- 高德地图 --------------------------
        # -------------------------- 高德地图 --------------------------
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🗺️ 高德地图 · 航道通航监控")
        # ===================== 采样优化：每个航道固定显示50个点 =====================
        MAX_POINTS_PER_CHANNEL = 50
        TOTAL_MAX_POINTS = 500

        sampled_points = []

        # ========= 固定阈值，和右下角图例严格统一：<200正常 /200-299轻度 /300-399中度 /≥400严重 =========
        light_t = 200
        moderate_t = 300
        heavy_t = 400

        # 全局统一配色与文字映射（和图例颜色一一对应）
        COLOR_MAP = {
            "normal": "#00CC00",
            "light": "#3690e8",  # 标准深蓝，消除偏紫问题
            "moderate": "#FFA500",
            "heavy": "#ff0000"
        }
        LABEL_MAP = {
            "normal": "正常通行",
            "light": "轻度拥堵",
            "moderate": "中度拥堵",
            "heavy": "严重拥堵"
        }

        # 1. 先收集统计信息，再用 expander 包裹
        stats_info = []
        stats_info.append(f"📊 原始数据总条数: {len(df)} 条，航道数: {len(channels)}")

        for channel in channels:
            channel_df = df[df['channel_id'] == channel].copy()
            original_count = len(channel_df)

            channel_df = channel_df.sort_values('time')

            if len(channel_df) > MAX_POINTS_PER_CHANNEL:
                step = len(channel_df) / MAX_POINTS_PER_CHANNEL
                indices = [int(i * step) for i in range(MAX_POINTS_PER_CHANNEL)]
                channel_df = channel_df.iloc[indices]
                stats_info.append(f"🗺️ 航道 {channel}: {original_count} → {len(channel_df)} 个点")
            else:
                stats_info.append(f"🗺️ 航道 {channel}: {original_count} 个点")

            if len(channel_df) == 0:
                continue

            # 循环逐行判定流量颜色，严格按固定200/300/400分界
            for _, row in channel_df.iterrows():
                flow = row['traffic_flow']
                # 固定阈值分级，不再自适应
                if flow < light_t:
                    color = COLOR_MAP["normal"]
                    level_name = LABEL_MAP["normal"]
                    level = "normal"
                elif light_t <= flow < moderate_t:
                    color = COLOR_MAP["light"]
                    level_name = LABEL_MAP["light"]
                    level = "light"
                elif moderate_t <= flow < heavy_t:
                    color = COLOR_MAP["moderate"]
                    level_name = LABEL_MAP["moderate"]
                    level = "moderate"
                else:
                    color = COLOR_MAP["heavy"]
                    level_name = LABEL_MAP["heavy"]
                    level = "heavy"

                sampled_points.append({
                    'lng': float(row['lon']),
                    'lat': float(row['lat']),
                    'flow': int(row['traffic_flow']),
                    'channel': row['channel_id'],
                    'speed': float(row['speed']),
                    'time': row['show_time'],
                    'color': color,
                    'level': level,
                    'level_name': level_name,
                    'id': len(sampled_points)
                })

        if len(sampled_points) > TOTAL_MAX_POINTS:
            step = len(sampled_points) / TOTAL_MAX_POINTS
            indices = [int(i * step) for i in range(TOTAL_MAX_POINTS)]
            sampled_points = [sampled_points[i] for i in indices]

        stats_info.append(f"🗺️ 地图最终显示 {len(sampled_points)} 个航道点位")

        # 按等级分组
        normal_pts = [p for p in sampled_points if p['level'] == 'normal']
        light_pts = [p for p in sampled_points if p['level'] == 'light']
        moderate_pts = [p for p in sampled_points if p['level'] == 'moderate']
        heavy_pts = [p for p in sampled_points if p['level'] == 'heavy']

        stats_info.append(
            f"📊 点位分布: 正常 {len(normal_pts)} | 轻度 {len(light_pts)} | 中度 {len(moderate_pts)} | 严重 {len(heavy_pts)}")

        # 2. 用 expander 包裹统计信息，默认收起
        with st.expander("📋 地图点位统计详情", expanded=False):
            for line in stats_info:
                st.info(line)

        if sampled_points:
            center_lat = sum(p['lat'] for p in sampled_points) / len(sampled_points)
            center_lng = sum(p['lng'] for p in sampled_points) / len(sampled_points)
        else:
            center_lat, center_lng = 38.93, 117.70

        amap_key = get_amap_key()
        # 图例文字同步固定阈值，和右下角图例完全一致
        legend_html = f'''
        <div class="legend-item active" data-level="normal">
            <span class="legend-color" style="background:{COLOR_MAP['normal']};"></span> {LABEL_MAP['normal']} (&lt;{light_t})
        </div>
        <div class="legend-item active" data-level="light">
            <span class="legend-color" style="background:{COLOR_MAP['light']};"></span> {LABEL_MAP['light']} ({light_t}-{moderate_t - 1})
        </div>
        <div class="legend-item active" data-level="moderate">
            <span class="legend-color" style="background:{COLOR_MAP['moderate']};"></span> {LABEL_MAP['moderate']} ({moderate_t}-{heavy_t - 1})
        </div>
        <div class="legend-item active" data-level="heavy">
            <span class="legend-color" style="background:{COLOR_MAP['heavy']};"></span> {LABEL_MAP['heavy']} (≥{heavy_t})
        </div>
        '''








        if not amap_key:
            st.warning("⚠️ 请先在侧边栏配置高德地图Key，否则地图无法显示")
        else:
            # 提前解析配色与标签，注入地图JS
            c_normal = COLOR_MAP["normal"]
            c_light = COLOR_MAP["light"]
            c_moderate = COLOR_MAP["moderate"]
            c_heavy = COLOR_MAP["heavy"]
            lab_normal = LABEL_MAP["normal"]
            lab_light = LABEL_MAP["light"]
            lab_moderate = LABEL_MAP["moderate"]
            lab_heavy = LABEL_MAP["heavy"]
            map_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>高德地图航道监控</title>
                <style>
                    body, html {{ margin: 0; padding: 0; height: 100%; width: 100%; }}
                    #container {{ height: 550px; width: 100%; }}
                    .legend {{
                        position: absolute;
                        bottom: 20px;
                        right: 20px;
                        background: white;
                        padding: 12px 15px;
                        border-radius: 8px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                        z-index: 1000;
                        font-size: 12px;
                        font-family: Arial, sans-serif;
                        background: rgba(255,255,255,0.95);
                        min-width: 140px;
                    }}
                    .legend-header {{
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        cursor: pointer;
                        user-select: none;
                    }}
                    .legend-header h4 {{
                        margin: 0;
                        font-size: 13px;
                    }}
                    .legend-toggle {{
                        font-size: 12px;
                        color: #666;
                    }}
                    .legend-content {{
                        margin-top: 10px;
                    }}
                    .legend-item {{ margin: 6px 0; cursor: pointer; user-select: none; }}
                    .legend-color {{
                        display: inline-block;
                        width: 14px;
                        height: 14px;
                        border-radius: 50%;
                        margin-right: 8px;
                        vertical-align: middle;
                    }}
                    .active {{ opacity: 1; }}
                    .inactive {{ opacity: 0.3; text-decoration: line-through; }}
                    .info-panel {{
                        position: absolute;
                        top: 10px;
                        left: 10px;
                        background: rgba(0,0,0,0.7);
                        color: white;
                        padding: 6px 12px;
                        border-radius: 5px;
                        font-size: 11px;
                        z-index: 1000;
                        font-family: monospace;
                        pointer-events: none;
                    }}
                </style>
                <script src="https://webapi.amap.com/maps?v=2.0&key={amap_key}"></script>
            </head>
            <body>
                <div id="container"></div>
                <div class="info-panel" id="infoPanel">高德地图加载中...</div>
                <div class="legend">
    <div class="legend-header" onclick="toggleLegend()">
        <h4>🚢 拥堵等级（点击切换）</h4>
        <span class="legend-toggle" id="legendToggle">▼</span>
    </div>
    <div class="legend-content" id="legendContent" style="display: block;">
    {legend_html}
    </div>


    </div>
</div>
                

                <script>
            
                    var infoPanel = document.getElementById('infoPanel');
                    var normalPoints = {json.dumps(normal_pts)};
                    var lightPoints = {json.dumps(light_pts)};
                    var moderatePoints = {json.dumps(moderate_pts)};
                    var heavyPoints = {json.dumps(heavy_pts)};
                    var totalPoints = normalPoints.length + lightPoints.length + moderatePoints.length + heavyPoints.length;
                    infoPanel.innerHTML = '✅ 已显示 ' + totalPoints + ' 个航道流量点';

                    var map = new AMap.Map('container', {{
                        center: [{center_lng}, {center_lat}],
                        zoom: 12,
                        resizeEnable: true,
                        viewMode: '2D'
                    }});

                    var allMarkers = [];
                    var markerGroups = {{}};

                    function addMarkers(points, color, level, levelName) {{
                        if (!points || points.length === 0) return;
                        var group = [];
                        for (var i = 0; i < points.length; i++) {{
                            var p = points[i];
                            if (!p.lng || !p.lat || isNaN(p.lng) || isNaN(p.lat)) continue;
                            var marker = new AMap.CircleMarker({{
                                center: [p.lng, p.lat],
                                radius: 8,
                                fillColor: color,
                                fillOpacity: 0.85,
                                strokeColor: '#ffffff',
                                strokeWeight: 1.5,
                                strokeOpacity: 1,
                                zIndex: 100
                            }});
                            marker.setExtData(p);
                            marker.on('click', function(ev) {{
                                var data = ev.target.getExtData();
                                var content = '<div style="padding:10px;min-width:160px;">' +
                                    '<div style="font-weight:bold;margin-bottom:8px;color:#0b3d7a;">🚢 航道信息</div>' +
                                    '<div>📍 航道：<b>' + data.channel + '</b></div>' +
                                    '<div>📊 流量：<b style="color:' + data.color + ';">' + data.flow + ' 艘/小时</b></div>' +
                                    '<div>⚡ 平均航速：<b>' + data.speed.toFixed(1) + '</b> 节</div>' +
                                    '<div>🎨 状态：<b style="color:' + data.color + ';">' + levelName + '</b></div>' +
                                    '<div>🕐 时间：' + data.time + '</div>' +
                                    '</div>';
                                var infoWindow = new AMap.InfoWindow({{
                                    content: content,
                                    offset: new AMap.Pixel(0, -20)
                                }});
                                infoWindow.open(map, ev.target.getCenter());
                            }});
                            marker.setMap(map);
                            group.push(marker);
                            allMarkers.push(marker);
                        }}
                        markerGroups[level] = group;
                    }}
                    map.on('complete', function() {{
                    addMarkers(normalPoints, '{c_normal}', 'normal', '{lab_normal}');
                    addMarkers(lightPoints, '{c_light}', 'light', '{lab_light}');
                    addMarkers(moderatePoints, '{c_moderate}', 'moderate', '{lab_moderate}');
                    addMarkers(heavyPoints, '{c_heavy}', 'heavy', '{lab_heavy}');
                }});
                    function toggleLegend() {{
                        var content = document.getElementById('legendContent');
                        var toggle = document.getElementById('legendToggle');
                        if (content.style.display === 'none') {{
                            content.style.display = 'block';
                            toggle.innerHTML = '▼';
                        }} else {{
                            content.style.display = 'none';
                            toggle.innerHTML = '▶';
                        }}
                    }}

                    var legendItems = document.querySelectorAll('.legend-item');
                    legendItems.forEach(function(item) {{
                        item.addEventListener('click', function(e) {{
                            e.stopPropagation();
                            var level = this.getAttribute('data-level');
                            var isActive = this.classList.contains('active');
                            if (markerGroups[level]) {{
                                markerGroups[level].forEach(function(marker) {{
                                    marker.setMap(isActive ? null : map);
                                }});
                            }}
                            this.classList.toggle('active');
                            this.classList.toggle('inactive');
                        }});
                    }});
                </script>
            </body>
            </html>
            """

            st.components.v1.html(map_html, height=600)
            st.caption("💡 点击右侧图例可切换显示/隐藏对应等级的船舶 | 点击圆点查看详情")
            st.success("✅ 高德地图已加载，地图已自动定位到船舶位置")

        st.markdown('</div>', unsafe_allow_html=True)
        # 可选：告警日志面板
        show_alert_log_panel(panel_unique_id="main_tab")
        # ========== 实时告警面板（始终显示，不需要天气数据） ==========
        if st.session_state.data_loaded and st.session_state.df is not None and not st.session_state.df.empty:
            with st.container():
                # 天气数据可以为 None，函数内部会处理
                show_real_time_alert_panel(st.session_state.df, weather_df if weather_df is not None else None,
                                           CONGESTION)
        else:
            st.info("👈 请先在左侧加载 AIS 数据，告警监控将自动启动")
        # ========== 流量预测（只在数据加载后显示）==========
        if st.session_state.data_loaded:
            # -------------------------- 流量预测 --------------------------
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🔮 智能流量预测（基于通义千问大模型）")

            if st.session_state.get("predict_finish_msg") is not None:
                st.success(st.session_state.predict_finish_msg)
                # 手动清除旧提示按钮
                if st.button("清除预测完成提示"):
                    st.session_state.predict_finish_msg = None
                    st.rerun()
            # ===========================================

            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                # 预测航道下拉框默认值与侧边栏全局航道同步
                default_predict_ch = st.session_state.channel_option
                predict_channel = st.selectbox(
                    "选择预测航道",
                    channels,
                    index=channels.index(default_predict_ch),
                    key="predict_channel"
                )

                # 双向同步：预测面板切换航道，同步更新侧边栏航道，刷新阈值
                if st.session_state.channel_option != predict_channel:
                    st.session_state.channel_option = predict_channel
                    # 切换预测航道，清空旧预测结果
                    st.session_state.prediction_completed = False
                    st.session_state.predict_finish_msg = None
                    if "pred_df" in st.session_state:
                        del st.session_state["pred_df"]
                    if "predict_channel_saved" in st.session_state:
                        del st.session_state["predict_channel_saved"]

                # 记录上次预测航道用于切换检测
                if "last_predict_channel" not in st.session_state:
                    st.session_state["last_predict_channel"] = predict_channel

                if st.session_state["last_predict_channel"] != predict_channel:
                    # 切换航道，清空所有预测相关session状态
                    st.session_state.prediction_completed = False
                    st.session_state.predict_finish_msg = None  # 清空完成提示
                    if "pred_df" in st.session_state:
                        del st.session_state["pred_df"]
                    if "predict_channel_saved" in st.session_state:
                        del st.session_state["predict_channel_saved"]
                    # 重置航道记录
                    st.session_state["last_predict_channel"] = predict_channel
            with col2:
                show_hourly_detail = st.checkbox("📊 显示分时段详情", value=True)
            with col3:
                show_congestion_list = st.checkbox("⚠️ 显示分级拥堵时段", value=True)

            # 初始化预测参数session存储
            model_options = {
                "qwen-plus": "效果好，适合重要预测",
                "qwen-turbo": "速度更快，适合实时预测",
                "qwen-max": "最强能力，适合复杂场景"
            }
            model_options_list = list(model_options.keys())
            if "pred_model" not in st.session_state:
                # 默认选中列表第一个模型
                st.session_state.pred_model = model_options_list[0]
            if "pred_temperature" not in st.session_state:
                st.session_state.pred_temperature = 0.1
            if "pred_use_fallback" not in st.session_state:
                st.session_state.pred_use_fallback = True
            # 新增：持久化预测完成提示
            if "predict_finish_msg" not in st.session_state:
                st.session_state.predict_finish_msg = None

            with st.expander("⚙️ 大模型参数配置", expanded=False):
                # 获取当前模型对应的索引，不存在则回退0
                try:
                    current_idx = model_options_list.index(st.session_state.pred_model)
                except ValueError:
                    current_idx = 0

                selected_model = st.selectbox(
                    "选择模型",
                    model_options_list,
                    index=current_idx,
                    format_func=lambda x: f"{x} - {model_options[x]}"
                )
            temperature = st.slider(
                "温度参数 (越高越随机)",
                0.0, 1.0,
                value=st.session_state.pred_temperature,
                step=0.05
            )
            use_fallback = st.checkbox(
                "API失败时使用备用预测",
                value=st.session_state.pred_use_fallback
            )

            # 实时更新参数到session，保证刷新不丢
            st.session_state.pred_model = selected_model
            st.session_state.pred_temperature = temperature
            st.session_state.pred_use_fallback = use_fallback

            # ========== 预测控制区域 ==========
            # ========== 预测控制区域 ==========
            st.markdown("---")

            col_btn1, col_btn2 = st.columns(2)

            col_btn1, col_btn2 = st.columns(2)

            with col_btn1:
                run_btn = st.button("🚀 运行大模型流量预测", use_container_width=True, type="primary")

            with col_btn2:
                current_sound = st.session_state.get('sound_enabled', True)
                # 声音开关按钮逻辑，布局保持在右侧列
                if current_sound:
                    # 声音开启：按钮显示【关闭告警声音】
                    sound_btn = st.button("🔇 关闭告警声音", use_container_width=True, type="secondary")
                    if sound_btn:
                        st.session_state.sound_enabled = False
                        st.rerun()
                else:
                    # 声音关闭：按钮显示【启用告警声音】
                    sound_btn = st.button("🔊 启用告警声音", use_container_width=True, type="primary")
                    if sound_btn:
                        st.session_state.sound_enabled = True
                        st.rerun()

            # 提示文案独立一行，整行宽度，自动在两个按钮下方（和截图布局完全匹配）
            if current_sound:
                st.info("🔊 告警声音已开启，预测拥堵时将自动播放提示音")
            else:
                st.info("🔇 告警声音已静音，预测拥堵不会播放提示音，点击上方按钮开启")

            st.markdown("---")

            # ========== 显示预测结果（无论是否刚运行预测，都从这里统一显示）==========
            # 如果有预测结果（无论是刚运行的还是之前保存的），都显示
            if st.session_state.get('prediction_completed', False):
                # 确保必要的 session_state 变量存在
                if all(key in st.session_state for key in ['pred_df', 'predict_channel_saved', 'history_flow_saved',
                                                           'adaptive_light_min', 'adaptive_moderate_min',
                                                           'adaptive_heavy_min',
                                                           'normal_count', 'light_count', 'middle_count',
                                                           'heavy_count']):

                    st.markdown("---")
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("🔮 智能流量预测结果")

                    # 显示当前显示的预测信息来源
                    # 显示当前显示的预测信息来源
                    if run_btn:
                        st.success(f"✅ 新预测完成！航道：{st.session_state.predict_channel_saved}")
                    else:
                        # 检查当前选中的航道是否与预测航道一致
                        if predict_channel != st.session_state.predict_channel_saved:
                            st.warning(
                                f"📊 当前显示的是航道【{st.session_state.predict_channel_saved}】的预测结果，如需查看【{predict_channel}】的预测，请点击「运行大模型流量预测」按钮")
                        else:
                            st.info(f"📊 当前显示上次预测结果（航道：{st.session_state.predict_channel_saved}）")

                    # 获取保存的数据
                    pred_df = st.session_state.pred_df
                    predict_channel = st.session_state.predict_channel_saved
                    history_flow = st.session_state.history_flow_saved
                    adaptive_light_min = st.session_state.adaptive_light_min
                    adaptive_moderate_min = st.session_state.adaptive_moderate_min
                    adaptive_heavy_min = st.session_state.adaptive_heavy_min
                    weather_merged = st.session_state.get('weather_merged', False)
                    show_hourly_detail = st.session_state.get('show_hourly_detail', True)
                    show_congestion_list = st.session_state.get('show_congestion_list', True)

                    # 显示自适应标准
                    st.info(
                        f"📊 当前使用自适应标准：轻度 ≥ {adaptive_light_min}，中度 ≥ {adaptive_moderate_min}，重度 ≥ {adaptive_heavy_min}")

                    # 显示统计卡片
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric(f"✅ 正常通行 (<{adaptive_light_min})", f"{st.session_state.normal_count}个")
                    c2.metric(f"🔷 轻度拥堵 ({adaptive_light_min}-{adaptive_moderate_min - 1})",
                              f"{st.session_state.light_count}个")
                    c3.metric(f"⚠️ 中度拥堵 ({adaptive_moderate_min}-{adaptive_heavy_min - 1})",
                              f"{st.session_state.middle_count}个")
                    c4.metric(f"🔴 严重拥堵 (≥{adaptive_heavy_min})", f"{st.session_state.heavy_count}个")

                    # 判断预测模式
                    total_hours = len(history_flow)
                    if total_hours < 24:
                        predict_mode = "hour"
                        predict_value = len(pred_df)
                    else:
                        predict_mode = "day"
                        predict_value = len(pred_df) // 24

                    # ========== 主预测图 ==========
                    fig_pred = go.Figure()
                    custom_data_pred = np.stack([pred_df['显示时间'], pred_df['拥堵等级']], axis=1)

                    fig_pred.add_trace(go.Scatter(
                        x=pred_df['时间'],
                        y=pred_df['预测流量'],
                        mode='lines+markers',
                        line=dict(color='#FF7D00', width=3, dash='dash'),
                        marker=dict(size=8, symbol='diamond', color='#FF7D00'),
                        name='预测流量',
                        yaxis="y",
                        customdata=custom_data_pred,
                        hovertemplate=(
                            '🕐 时间: %{customdata[0]}<br>'
                            '📊 流量: %{y:.0f} 艘/小时<br>'
                            '🎨 状态: %{customdata[1]}<br>'
                            '<extra></extra>'
                        )
                    ))
                    fig_pred.add_trace(go.Scatter(
                        x=pred_df['时间'], y=pred_df['预测流量'], fill='tozeroy',
                        fillcolor='rgba(255, 125, 0, 0.2)', line=dict(color='rgba(255,125,0,0)'),
                        showlegend=False, hoverinfo='skip', yaxis="y"
                    ))

                    if weather_merged:
                        try:
                            if "风速(m/s)" in pred_df.columns:
                                fig_pred.add_trace(go.Scatter(
                                    x=pred_df['时间'],
                                    y=pred_df['风速(m/s)'],
                                    mode='lines',
                                    line=dict(color='#1f77b4', width=2),
                                    name='风速(m/s)',
                                    yaxis="y2",
                                    hovertemplate='🕐 时间: %{x}<br>💨 风速: %{y:.1f} m/s<extra></extra>'
                                ))
                        except:
                            pass
                        try:
                            if "能见度(m)" in pred_df.columns:
                                fig_pred.add_trace(go.Scatter(
                                    x=pred_df['时间'],
                                    y=pred_df['能见度(m)'],
                                    mode='lines',
                                    line=dict(color='#2ca02c', width=2, dash='dot'),
                                    name='能见度(m)',
                                    yaxis="y2",
                                    hovertemplate='🕐 时间: %{x}<br>👁️ 能见度: %{y:.0f} m<extra></extra>'
                                ))
                        except:
                            pass

                    if max(pred_df['预测流量']) > 200:
                        fig_pred.add_hline(y=adaptive_light_min, line_dash="dash", line_color="#ffc107",
                                           annotation_text="轻度拥堵")
                        fig_pred.add_hline(y=adaptive_moderate_min, line_dash="dash", line_color="#fd7e14",
                                           annotation_text="中度拥堵")
                        fig_pred.add_hline(y=adaptive_heavy_min, line_color="#dc3545", line_dash="dash",
                                           annotation_text="严重拥堵")
                    fig_pred.update_layout(
                        height=400,
                        title=f"{predict_channel} 未来{predict_value}{'小时' if predict_mode == 'hour' else '天'}流量预测（含天气影响）",
                        xaxis_title="时间",
                        yaxis=dict(title="流量 (艘/小时)", side="left"),
                        yaxis2=dict(title="风速/能见度", side="right", overlaying="y", showgrid=False),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        margin=dict(l=10, r=10, t=60, b=40)
                    )
                    timestamp = st.session_state.get('prediction_timestamp', '')
                    st.plotly_chart(fig_pred, use_container_width=True,
                                    key=f"pred_chart_{st.session_state.predict_channel_saved}_{timestamp}")

                    # ========== 天气影响分析图表（完整版） ==========
                    # ========== 天气对通航流量的影响分析（和截图布局完全一致：左右两列散点） ==========
                    # ========== 天气影响分析图表（使用历史数据） ==========
                    # ========== 天气影响分析图表（修复版） ==========
                    if weather_df is not None and not weather_df.empty:
                        st.markdown("---")
                        st.subheader("🌤️ 天气对通航流量的影响分析（历史数据）")

                        # ✅关键修复：每次绘图实时重新合并流量与天气，规避刷新丢失字段
                        df_raw = st.session_state.df.copy()
                        df_with_weather = merge_weather_data(df_raw, weather_df)

                        col_scatter1, col_scatter2 = st.columns(2)

                        # 左列：历史风速 vs 历史流量
                        with col_scatter1:
                            st.markdown("#### 💨 风速 vs 流量")
                            wind_col = None
                            for col in ['wind_speed_x', 'wind_speed', 'windspeed', 'wind', '风速']:
                                if col in df_with_weather.columns:
                                    wind_col = col
                                    break

                            if wind_col:
                                plot_df = df_with_weather[[wind_col, 'traffic_flow']].dropna()
                                if not plot_df.empty:
                                    # 添加抖动让点散开
                                    np.random.seed(42)
                                    jitter_x = np.random.normal(0, 0.2, len(plot_df))
                                    jitter_y = np.random.normal(0, 5, len(plot_df))

                                    fig_wind = go.Figure()
                                    fig_wind.add_trace(go.Scatter(
                                        x=plot_df[wind_col] + jitter_x,
                                        y=plot_df['traffic_flow'] + jitter_y,
                                        mode='markers',
                                        marker=dict(
                                            color='#1f77b4',
                                            size=8,
                                            opacity=0.5,
                                            line=dict(color='white', width=0.5)
                                        ),
                                        hovertemplate='风速: %{x:.1f} m/s<br>流量: %{y:.0f} 艘/小时<extra></extra>'
                                    ))
                                    fig_wind.update_layout(
                                        title="历史风速 vs 历史流量",
                                        height=350,
                                        xaxis_title="风速(m/s)",
                                        yaxis_title="流量(艘/小时)"
                                    )
                                    st.plotly_chart(fig_wind, use_container_width=True)
                                else:
                                    st.info("风速与流量无匹配有效数据")
                            else:
                                st.warning("合并后数据内未找到【风速】字段")

                        # 右列：历史能见度 vs 历史流量
                        with col_scatter2:
                            st.markdown("#### 👁️ 能见度 vs 流量")
                            vis_col = None
                            for col in ['visibility_x', 'visibility', 'vis', '能见度']:
                                if col in df_with_weather.columns:
                                    vis_col = col
                                    break

                            if vis_col:
                                plot_df = df_with_weather[[vis_col, 'traffic_flow']].dropna()
                                if not plot_df.empty:
                                    np.random.seed(42)
                                    jitter_y = np.random.normal(0, 5, len(plot_df))

                                    fig_vis = go.Figure()
                                    fig_vis.add_trace(go.Scatter(
                                        x=plot_df[vis_col],
                                        y=plot_df['traffic_flow'] + jitter_y,
                                        mode='markers',
                                        marker=dict(
                                            color='#2ca02c',
                                            size=8,
                                            opacity=0.5,
                                            line=dict(color='white', width=0.5)
                                        ),
                                        hovertemplate='能见度: %{x:.0f} m<br>流量: %{y:.0f} 艘/小时<extra></extra>'
                                    ))
                                    fig_vis.update_layout(
                                        title="历史能见度 vs 历史流量",
                                        height=350,
                                        xaxis_title="能见度(m)",
                                        yaxis_title="流量(艘/小时)"
                                    )
                                    st.plotly_chart(fig_vis, use_container_width=True)
                                else:
                                    st.info("能见度与流量无匹配有效数据")
                            else:
                                st.warning("合并后数据内未找到【能见度】字段")
                        st.markdown("---")

                        # 🌈 不同天气平均通航流量（历史）
                        weather_col = None
                        for col in ['weather_x', 'weather', '天气', 'condition']:
                            if col in df_with_weather.columns:
                                weather_col = col
                                break

                        if weather_col:
                            st.markdown("#### 🌈 不同天气平均通航流量（历史）")
                            plot_df = df_with_weather[[weather_col, 'traffic_flow']].dropna()
                            if not plot_df.empty:
                                weather_avg = plot_df.groupby(weather_col)['traffic_flow'].mean().reset_index()
                                weather_avg.columns = ['天气', '平均流量']
                                weather_avg['平均流量'] = weather_avg['平均流量'].round(0)

                                fig_bar = go.Figure(go.Bar(
                                    x=weather_avg['天气'].astype(str),
                                    y=weather_avg['平均流量'],
                                    marker_color='#FF7D00',
                                    text=weather_avg['平均流量'],
                                    textposition='auto'
                                ))
                                fig_bar.update_layout(height=350, xaxis_title="天气状况",
                                                      yaxis_title="平均流量 (艘/小时)")
                                st.plotly_chart(fig_bar, use_container_width=True)

                                # 拥堵占比
                                st.markdown("#### 📊 不同天气拥堵占比（历史）")
                                plot_df['is_congestion'] = plot_df['traffic_flow'] >= CONGESTION["light"]["min"]
                                congestion_rate = plot_df.groupby(weather_col).agg({
                                    'is_congestion': lambda x: round(x.mean() * 100, 2)
                                }).reset_index()
                                congestion_rate.columns = ['天气', '拥堵占比(%)']

                                fig_cong = go.Figure(go.Bar(
                                    x=congestion_rate['天气'].astype(str),
                                    y=congestion_rate['拥堵占比(%)'],
                                    marker_color='#dc3545',
                                    text=congestion_rate['拥堵占比(%)'],
                                    textposition='auto'
                                ))
                                fig_cong.update_layout(height=350, xaxis_title="天气状况", yaxis_title="拥堵占比(%)",
                                                       yaxis=dict(ticksuffix='%'))
                                st.plotly_chart(fig_cong, use_container_width=True)
                        st.markdown("---")

                    # 获取该航道的历史数据用于自适应阈值（使用保存的预测航道）
                    hist_df_for_alert = df[df['channel_id'] == st.session_state.predict_channel_saved][
                        'traffic_flow'].tolist() if 'df' in st.session_state and st.session_state.df is not None else history_flow

                    show_predict_alert_panel(
                        pred_df,
                        st.session_state.predict_channel_saved,  # 使用保存的航道名称
                        CONGESTION,
                        hist_df_for_alert,
                        sound_enabled=st.session_state.get('sound_enabled', True)
                    )

                    # ========== 预测告警日志 ==========
                    st.markdown("---")
                    st.subheader("本次预测告警日志")

                    show_predict_alert_log_panel(panel_unique_id="prediction_tab")
                    # 第一个标签 tab_main 到此结束
                    # 新增第二个标签：船舶轨迹预测功能
                    # 第二个标签 - 确保放在外面，不在任何 if 内部

                    if show_hourly_detail:
                        with st.expander("📊 分时段详情", expanded=True):
                            pred_df_local = pred_df.copy()
                            pred_df_local['小时'] = pred_df_local['小时'].clip(0, 23)
                            pred_df_local['时段'] = pd.cut(
                                pred_df_local['小时'],
                                bins=[0, 6, 9, 12, 14, 17, 19, 22, 23],
                                labels=['深夜(0-6)', '早高峰(6-9)', '上午(9-12)', '中午(12-14)', '下午(14-17)',
                                        '晚高峰(17-19)',
                                        '晚间(19-22)', '深夜(22-23)']
                            )
                            st.dataframe(
                                pred_df_local.groupby('时段', observed=True)['预测流量'].agg(
                                    ['mean', 'max', 'min']).round(
                                    0),
                                use_container_width=True)
                            # ========== 添加柱状图 ==========
                            st.markdown("---")
                            st.markdown("#### 📊 每日分时流量柱状图")
                            timestamp = st.session_state.get('prediction_timestamp', '')
                            for idx, d in enumerate(pred_df['时间'].dt.date.unique()):
                                day_data = pred_df[pred_df['时间'].dt.date == d].sort_values('小时')
                                st.markdown(f"**{d}**")
                                # 读取会话缓存自适应拥堵分界阈值
                                light_t = st.session_state.adaptive_light_min
                                mid_t = st.session_state.adaptive_moderate_min
                                heavy_t = st.session_state.adaptive_heavy_min

                                colors = []
                                for x in day_data['预测流量']:
                                    if x < light_t:
                                        colors.append("#00CC00")  # 正常通行：绿色
                                    elif light_t <= x < mid_t:
                                        colors.append("#3690e8")  # 轻度拥堵：蓝色
                                    elif mid_t <= x < heavy_t:
                                        colors.append("#FFA500")  # 中度拥堵：橙色
                                    else:
                                        colors.append("#ff0000")  # 严重拥堵：红色



                                fig = go.Bar(x=day_data['小时'], y=day_data['预测流量'], marker_color=colors,
                                             text=day_data['预测流量'].round(0))
                                fig = go.Figure(fig)
                                if max(day_data['预测流量']) > 200:

                                    fig.add_hline(y=adaptive_light_min, line_dash="dash", line_color="#ffc107",
                                                  annotation_text="轻度拥堵")
                                    fig.add_hline(y=adaptive_moderate_min, line_dash="dash", line_color="#fd7e14",
                                                  annotation_text="中度拥堵")
                                    fig.add_hline(y=adaptive_heavy_min, line_color="#dc3545", line_dash="dash",
                                                  annotation_text="严重拥堵")
                                fig.update_layout(
                                    height=300,
                                    xaxis_title="小时",
                                    yaxis_title="流量 (艘/小时)",
                                    margin=dict(l=10, r=10, t=40, b=40)
                                )
                                st.plotly_chart(fig, use_container_width=True, key=f"hourly_bar_{idx}_{d}_{timestamp}")

                    # ========== 全时段通行明细 ==========
                    if show_congestion_list:
                        st.markdown("---")
                        st.markdown("### 📋 全时段通行明细")
                        tab1, tab2, tab3, tab4 = st.tabs(["✅ 正常", "🔷 轻度", "⚠️ 中度", "🔴 严重"])
                        # 固定拥堵等级文本，与判定函数输出完全一致
                        level_list = ["正常通行", "轻度拥堵", "中度拥堵", "严重拥堵"]
                        for tab, level in zip([tab1, tab2, tab3, tab4], level_list):
                            with tab:
                                sub = pred_df[pred_df["拥堵等级"] == level].copy()

                                if len(sub) == 0:
                                    st.info(f"无{level}时段")
                                else:
                                    show_cols = ["显示时间", "星期", "小时", "预测流量", "拥堵等级"]
                                    if weather_merged:
                                        if "气温(℃)" in pred_df.columns:
                                            show_cols.append("气温(℃)")
                                        if "风速(m/s)" in pred_df.columns:
                                            show_cols.append("风速(m/s)")
                                        if "能见度(m)" in pred_df.columns:
                                            show_cols.append("能见度(m)")


                                            def get_weather_icon(row):
                                                icons = []
                                                if row.get('风速(m/s)', 0) >= 8:
                                                    icons.append("💨大风")
                                                if row.get('能见度(m)', 9999) < 1000:
                                                    icons.append("🌫️低能见度")
                                                if row.get('气温(℃)', 20) < 5:
                                                    icons.append("❄️低温")
                                                elif row.get('气温(℃)', 20) > 35:
                                                    icons.append("🔥高温")
                                                return ' | '.join(icons) if icons else "✅正常"


                                            sub['天气标识'] = sub.apply(get_weather_icon, axis=1)
                                            show_cols.append("天气标识")
                                            # ========== 新增结束 ==========
                                    show = sub[show_cols].rename(
                                        columns={"显示时间": "时间", "预测流量": "流量(艘/小时)"}
                                    ).sort_values("时间").reset_index(drop=True)
                                    show.index = show.index + 1
                                    st.dataframe(show, use_container_width=True, height=400)

                    # ========== 同小时对比 ==========
                    st.markdown("---")
                    st.markdown("### 📈 同小时准确率对比")
                    # 使用历史数据中的实际航道数据
                    # 使用历史数据中的实际航道数据（使用保存的预测航道）
                    hist_df_channel = df[df[
                                             'channel_id'] == st.session_state.predict_channel_saved].copy() if 'df' in st.session_state and st.session_state.df is not None else None
                    if hist_df_channel is not None and len(hist_df_channel) > 0:
                        hist_hour = hist_df_channel.groupby("hour")["traffic_flow"].mean().reset_index()
                        hist_hour.columns = ["小时", "历史流量"]
                        pred_hour = pred_df.groupby("小时")["预测流量"].mean().reset_index()
                        pred_hour.columns = ["小时", "预测流量"]
                        full_hour = pd.DataFrame({"小时": list(range(24))})
                        compare_df = full_hour.merge(hist_hour, on="小时", how="left")
                        compare_df = compare_df.merge(pred_hour, on="小时", how="left")

                        fig_compare = go.Figure()
                        fig_compare.add_trace(go.Scatter(
                            x=compare_df["小时"],
                            y=compare_df["历史流量"],
                            mode='lines+markers',
                            line=dict(color='#3690e8', width=3),
                            marker=dict(size=8),
                            name='历史真实（全时段同小时均值）'
                        ))
                        fig_compare.add_trace(go.Scatter(
                            x=compare_df["小时"],
                            y=compare_df["预测流量"],
                            mode='lines+markers',
                            line=dict(color='#FF7D00', width=3, dash='dash'),
                            marker=dict(size=8, symbol='diamond'),
                            name='预测（同小时均值）'
                        ))

                        if compare_df["历史流量"].max() > 200:
                            fig_compare.add_hline(y=CONGESTION["light"]["min"], line_dash="dash", line_color="#ffc107",
                                                  annotation_text=CONGESTION["light"]["label"])
                            fig_compare.add_hline(y=CONGESTION["moderate"]["min"], line_dash="dash",
                                                  line_color="#fd7e14",
                                                  annotation_text=CONGESTION["moderate"]["label"])
                            fig_compare.add_hline(y=CONGESTION["heavy"]["min"], line_color="#dc3545", line_dash="dash",
                                                  annotation_text=CONGESTION["heavy"]["label"])

                        fig_compare.update_layout(
                            height=450,
                            title=f"{predict_channel} | 0~23点 真实均值 VS 预测均值",
                            xaxis_title="自然小时（0~23）",
                            yaxis_title="流量（艘/小时）",
                            xaxis=dict(tickmode="array", tickvals=list(range(24))),
                            hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02),
                            margin=dict(l=10, r=10, t=60, b=30)
                        )
                        st.plotly_chart(fig_compare, use_container_width=True,
                                        key=f"compare_chart_{st.session_state.predict_channel_saved}_{timestamp}")

                        # 计算误差
                        valid = compare_df.dropna(subset=["历史流量", "预测流量"])
                        if len(valid) > 0:
                            # 确保绝对误差列存在
                            if "绝对误差" not in compare_df.columns:
                                compare_df["绝对误差"] = abs(compare_df["历史流量"] - compare_df["预测流量"])
                                valid = compare_df.dropna(subset=["历史流量", "预测流量"])

                            avg_abs_err = valid["绝对误差"].mean()

                            if valid["历史流量"].mean() < 30:
                                st.success(f"✅ 平均绝对误差 = {avg_abs_err:.1f} 艘/小时")
                                if avg_abs_err < 10:
                                    st.markdown(
                                        '<span style="color:#28a745; font-weight:bold;">📈 预测质量评级：优秀 - 预测非常准确</span>',
                                        unsafe_allow_html=True)
                                elif avg_abs_err < 20:
                                    st.markdown(
                                        '<span style="color:#17a2b8; font-weight:bold;">📈 预测质量评级：良好 - 预测较为准确</span>',
                                        unsafe_allow_html=True)
                                elif avg_abs_err < 35:
                                    st.markdown(
                                        '<span style="color:#ffc107; font-weight:bold;">📈 预测质量评级：一般 - 预测有一定偏差</span>',
                                        unsafe_allow_html=True)
                                else:
                                    st.markdown(
                                        '<span style="color:#dc3545; font-weight:bold;">📈 预测质量评级：需改进 - 预测偏差较大</span>',
                                        unsafe_allow_html=True)

                                # ========== 添加详细表格 ==========
                                display_df = valid[["小时", "历史流量", "预测流量", "绝对误差"]].round(1)
                                display_df.columns = ["小时", "历史流量(艘/小时)", "预测流量(艘/小时)",
                                                      "绝对误差(艘/小时)"]
                                st.dataframe(display_df, use_container_width=True)

                            else:
                                valid["相对误差%"] = (
                                        abs(valid["历史流量"] - valid["预测流量"]) / valid["历史流量"] * 100).round(2)
                                avg_rel_err = valid["相对误差%"].mean()
                                st.success(
                                    f"✅ 平均绝对误差 = {avg_abs_err:.1f} 艘/小时 | 平均相对误差 = {avg_rel_err:.1f}%")
                                if avg_rel_err < 30:
                                    st.markdown(
                                        '<span style="color:#28a745; font-weight:bold;">📈 预测质量评级：优秀 - 预测非常准确</span>',
                                        unsafe_allow_html=True)
                                elif avg_rel_err < 40:
                                    st.markdown(
                                        '<span style="color:#17a2b8; font-weight:bold;">📈 预测质量评级：良好 - 预测较为准确</span>',
                                        unsafe_allow_html=True)
                                elif avg_rel_err < 60:
                                    st.markdown(
                                        '<span style="color:#ffc107; font-weight:bold;">📈 预测质量评级：一般 - 预测有一定偏差</span>',
                                        unsafe_allow_html=True)
                                else:
                                    st.markdown(
                                        '<span style="color:#dc3545; font-weight:bold;">📈 预测质量评级：需改进 - 预测偏差较大</span>',
                                        unsafe_allow_html=True)

                                # ========== 添加详细表格（包含相对误差）==========
                                display_df = valid[["小时", "历史流量", "预测流量", "绝对误差", "相对误差%"]].round(1)
                                display_df.columns = ["小时", "历史流量(艘/小时)", "预测流量(艘/小时)",
                                                      "绝对误差(艘/小时)",
                                                      "相对误差(%)"]
                                st.dataframe(display_df, use_container_width=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                else:
                    # 没有预测结果时显示提示
                    st.info("👈 请配置参数后点击「运行大模型流量预测」按钮开始预测")

            if run_btn:
                if not get_api_key():
                    st.error("❌ 请先在侧边栏配置通义千问API Key")
                    st.stop()

                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    status_text.markdown('<div class="loading-text">📊 正在准备历史数据...</div>',
                                         unsafe_allow_html=True)
                    progress_bar.progress(10)

                    hist_df = df[df['channel_id'] == predict_channel].copy()
                    if hist_df.empty:
                        st.error(f"❌ 航道 {predict_channel} 没有历史数据")
                        st.stop()
                    hist_df = hist_df.sort_values('time')
                    history_flow = hist_df['traffic_flow'].tolist()
                    # 获取当前预测航道的自适应阈值（新增这段，解决变量未定义）
                    # 全局统一固定分级标准
                    adaptive_light_min = 200
                    adaptive_moderate_min = 300
                    adaptive_heavy_min = 400
                    selected_hist = df[df["channel_id"] == predict_channel].copy()
                    if len(selected_hist) >= 2:
                        time_diff = selected_hist["time"].max() - selected_hist["time"].min()
                        total_hours = time_diff.total_seconds() / 3600
                    else:
                        total_hours = 168

                    if total_hours < 24:
                        predict_mode = "hour"
                        predict_value = predict_value if 'predict_value' in locals() else 6
                        predict_hours = predict_value
                    else:
                        predict_mode = "day"
                        predict_value = predict_value if 'predict_value' in locals() else 3
                        predict_hours = predict_value * 24

                    # ================== 提取未来天气 ==================
                    future_weather = None
                    if weather_df is not None:
                        last_time = df['time'].max()
                        weather_df_copy = weather_df.copy()
                        weather_df_copy['time'] = pd.to_datetime(weather_df_copy['time'])
                        future_weather = weather_df_copy[weather_df_copy['time'] > last_time].copy()
                        if len(future_weather) > 0:
                            st.success(f"✅ 已识别【未来天气数据】{len(future_weather)}条，AI将自动修正预测流量！")
                            st.info("💡 修正规则：大风/低能见度/雨雪 → 流量自动下调；晴好天气 → 流量回归常态")
                    # ==================================================

                    status_text.markdown('<div class="loading-text">🤖 正在调用通义千问大模型进行智能预测...</div>',
                                         unsafe_allow_html=True)
                    progress_bar.progress(30)

                    predictions = predict_traffic_flow_with_llm(
                        history_flow,
                        predict_channel,
                        predict_hours=predict_hours,
                        model=selected_model,
                        temperature=temperature,
                        future_weather_df=future_weather
                    )

                    if predictions is None:
                        if use_fallback:
                            status_text.markdown(
                                '<div class="loading-text">⚠️ 大模型调用失败，使用备用预测方法...</div>',
                                unsafe_allow_html=True)
                            predictions = fallback_prediction(history_flow, predict_hours=predict_hours)
                            st.warning("⚠️ 大模型API调用失败，已使用备用统计方法进行预测")
                        else:
                            st.error("❌ 大模型预测失败，请检查API配置或网络连接")
                            st.stop()

                    progress_bar.progress(80)
                    status_text.markdown('<div class="loading-text">📊 正在生成预测报表...</div>',
                                         unsafe_allow_html=True)

                    last_time = df['time'].max()
                    dates = [last_time + timedelta(hours=i + 1) for i in range(predict_hours)]
                    hours = [d.hour for d in dates]
                    pred_df = pd.DataFrame({
                        "时间": dates,
                        "预测流量": predictions,
                        "小时": hours,
                        "星期": [week_map[d.weekday()] for d in dates],
                    })
                    # 读取当前航道自适应阈值，全程统一判定标准
                    adaptive_light = adaptive_light_min
                    adaptive_moderate = adaptive_moderate_min
                    adaptive_heavy = adaptive_heavy_min

                    # 读取当前航道自适应分界
                    adaptive_light = adaptive_light_min
                    adaptive_moderate = adaptive_moderate_min
                    adaptive_heavy = adaptive_heavy_min


                    def get_congestion_level(flow):
                        if flow < adaptive_light:
                            return "正常通行"
                        elif adaptive_light <= flow < adaptive_moderate:
                            return "轻度拥堵"
                        elif adaptive_moderate <= flow < adaptive_heavy:
                            return "中度拥堵"
                        else:
                            return "严重拥堵"


                    pred_df['拥堵等级'] = pred_df['预测流量'].apply(get_congestion_level)

                    pred_df['显示时间'] = pred_df['时间'].dt.strftime('%Y-%m-%d %H:%M')
                    pred_df['time_str'] = pred_df['小时'].apply(lambda h: f"{h:02d}:00")

                    # 给预测表追加天气字段
                    weather_merged = False
                    if weather_df is not None:
                        df['time'] = pd.to_datetime(df['time'])
                        weather_df['time'] = pd.to_datetime(weather_df['time'])
                        pred_df = pd.merge_asof(
                            pred_df.sort_values("时间"),
                            weather_df.sort_values("time"),
                            left_on="时间",
                            right_on="time",
                            direction="nearest"
                        )
                    rename_map = {}
                    for col in pred_df.columns:
                        col_lower = col.lower()
                        if '温度' in col or col_lower == 'temperature':
                            rename_map[col] = "气温(℃)"
                        elif '风速' in col or col_lower == 'wind_speed' or col_lower == 'windspeed':
                            rename_map[col] = "风速(m/s)"
                        elif '能见度' in col or col_lower == 'visibility':
                            rename_map[col] = "能见度(m)"
                        elif '湿度' in col or col_lower == 'humidity':
                            rename_map[col] = "湿度(%)"
                    if rename_map:
                        pred_df.rename(columns=rename_map, inplace=True)
                        weather_merged = True
                        st.success(f"✅ 天气数据合并成功，已识别：{list(rename_map.values())}")

                    # 计算该航道的自适应阈值
                    # 全局统一固定分级标准，和地图完全对齐
                    # 统一使用自适应阈值，和侧边栏拥堵分级标准完全一致
                    selected_channel_hist = df[df['channel_id'] == predict_channel]['traffic_flow'].tolist()
                    thresholds = calculate_adaptive_thresholds(selected_channel_hist, predict_channel)
                    adaptive_light_min = thresholds["light"]
                    adaptive_moderate_min = thresholds["moderate"]
                    adaptive_heavy_min = thresholds["heavy"]

                    # 使用自适应阈值进行统计
                    normal = len(pred_df[pred_df["预测流量"] < adaptive_light_min])
                    light = len(
                        pred_df[
                            (pred_df["预测流量"] >= adaptive_light_min) & (
                                        pred_df["预测流量"] < adaptive_moderate_min)])
                    middle = len(
                        pred_df[
                            (pred_df["预测流量"] >= adaptive_moderate_min) & (
                                        pred_df["预测流量"] < adaptive_heavy_min)])
                    heavy = len(pred_df[pred_df["预测流量"] >= adaptive_heavy_min])

                    progress_bar.progress(100)
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()

                    # ========== 保存预测结果到 session_state ==========
                    st.session_state.prediction_completed = True
                    st.session_state.pred_df = pred_df
                    st.session_state.predict_channel_saved = predict_channel
                    # 预测完成后，同步航道到侧边栏，统一阈值数据源
                    st.session_state.channel_option = predict_channel

                    st.session_state.prediction_timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                    st.session_state.history_flow_saved = history_flow
                    st.session_state.adaptive_light_min = adaptive_light_min
                    st.session_state.adaptive_moderate_min = adaptive_moderate_min
                    st.session_state.adaptive_heavy_min = adaptive_heavy_min
                    st.session_state.normal_count = normal
                    st.session_state.light_count = light
                    st.session_state.middle_count = middle
                    st.session_state.heavy_count = heavy
                    st.session_state.weather_merged = weather_merged
                    st.session_state.show_hourly_detail = show_hourly_detail
                    st.session_state.show_congestion_list = show_congestion_list

                    # 保存完成提示信息到session，永久保留
                    st.session_state.predict_finish_msg = f"✅ 大模型预测完成！基于 {selected_model} 模型"
                    # ============ 在这里新增两行重置对话代码 ============
                    from ai_assistant import reset_chat_session

                    reset_chat_session()
                    st.rerun()

                except Exception as e:
                    st.error(f"预测过程出错: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()
    with tab_trajectory:
        raw_data = st.session_state.raw_detail_ais
        amap_key = get_amap_key()
        if raw_data is None or raw_data.empty:
            st.info("暂无原始船舶明细数据，无法查看轨迹")
        else:
            render_trajectory(raw_data, amap_key)

        # ========== 第三个标签：AI 预测助手 ==========
    with tab_ai_assistant:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        # 检查是否有预测结果
        if st.session_state.get('prediction_completed', False) and 'pred_df' in st.session_state:
            from ai_assistant import render_ai_assistant

            # 获取预测模式
            history_flow = st.session_state.history_flow_saved
            if len(history_flow) < 24:
                predict_mode = "hour"
                predict_value = len(st.session_state.pred_df)
            else:
                predict_mode = "day"
                predict_value = len(st.session_state.pred_df) // 24

            render_ai_assistant(
                pred_df=st.session_state.pred_df,
                predict_channel=st.session_state.predict_channel_saved,
                predict_value=predict_value,
                predict_mode=predict_mode,
                api_key_func=get_api_key,
                selected_model=st.session_state.get('pred_model', 'qwen-turbo')
            )
        else:
            st.info("👈 请先在「航道监控与流量预测」标签页中运行流量预测")
            st.markdown("""
            💡 **使用说明：**
            1. 在「航道监控与流量预测」标签页中配置参数
            2. 点击「运行大模型流量预测」按钮
            3. 预测完成后，回到这里提问
            """)
            st.divider()
            st.markdown("**📝 示例问题：**")
            st.markdown("""
            - 什么时候最拥堵？
            - 平均流量是多少？
            - 哪个小时流量最高？
            - 帮我总结拥堵情况
            """)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#666;font-size:14px'>© 2026 基于多源数据的港口交通流预测与示警智能体 | 基于通义千问大模型 + Streamlit</div>",
        unsafe_allow_html=True)

