# alert_system.py - 最终完整版（修复弹窗叠加+全局样式加载+优化交互）
import time  # 在文件顶部添加
import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import numpy as np  # 确保导入 numpy

# ==============================================
# 告警分级配置
# ==============================================

# 告警分级配置
ALERT_GRADE = {
    "emergency": {  # 一级紧急告警
        "name": "紧急告警",
        "color": "#dc2626",
        "bg_color": "#fef2f2",
        "border_color": "#dc2626",
        "icon": "🔴",
        "animation": "emergencyFlash",
        "sound": "emergency",
        "speed": "fast",
        "priority": 1
    },
    "alert": {  # 二级重要告警
        "name": "重要告警",
        "color": "#f97316",
        "bg_color": "#fff7ed",
        "border_color": "#f97316",
        "icon": "🟠",
        "animation": "alertFlash",
        "sound": "alert",
        "speed": "medium",
        "priority": 2
    },
    "warning": {  # 三级提醒/预测告警
        "name": "提醒",
        "color": "#eab308",
        "bg_color": "#fefce8",
        "border_color": "#eab308",
        "icon": "🟡",
        "animation": "warningFlash",
        "sound": "warning",
        "speed": "slow",
        "priority": 3
    }
}

# 告警类型标签
ALERT_TYPE_LABELS = {
    "real_time": {"name": "实时告警", "color": "#3b82f6"},
    "predict": {"name": "预测告警", "color": "#8b5cf6"}
}

# ==============================================
# 告警等级常量配置
# ==============================================
ALERT_LEVEL = {
    "normal": {"color": "#00CC00", "label": "正常", "priority": 0},
    "warning": {"color": "#FFA500", "label": "提醒", "priority": 1},
    "alert": {"color": "#FF0000", "label": "预警", "priority": 2},
    "emergency": {"color": "#8B0000", "label": "紧急", "priority": 3},
}

CONGESTION = {
    "normal": {"max": 299, "color": "#00CC00", "label": "正常通行"},
    "light": {"min": 300, "max": 399, "color": "#FFA500", "label": "轻度拥堵"},
    "moderate": {"min": 400, "max": 499, "color": "#FF0000", "label": "中度拥堵"},
    "heavy": {"min": 500, "color": "#8B0000", "label": "严重拥堵"},
}


def calculate_adaptive_thresholds(history_flow, channel_name=""):
    """
    根据历史数据动态计算拥堵阈值（基于百分位数）
    返回: {"light": 阈值, "moderate": 阈值, "heavy": 阈值}
    """
    if len(history_flow) == 0:
        return {"light": 300, "moderate": 400, "heavy": 500}

    hist_array = np.array(history_flow)

    # 使用历史流量的百分位数作为阈值
    # 轻度拥堵：超过历史70%的时段
    # 中度拥堵：超过历史85%的时段
    # 严重拥堵：超过历史95%的时段
    light_threshold = np.percentile(hist_array, 70)
    moderate_threshold = np.percentile(hist_array, 85)
    heavy_threshold = np.percentile(hist_array, 95)

    # 保留下限，确保合理性
    light_threshold = max(light_threshold, 150)
    moderate_threshold = max(moderate_threshold, 250)
    heavy_threshold = max(heavy_threshold, 400)

    # 确保阈值递增
    if light_threshold >= moderate_threshold:
        moderate_threshold = light_threshold + 50
    if moderate_threshold >= heavy_threshold:
        heavy_threshold = moderate_threshold + 50

    return {
        "light": int(light_threshold),
        "moderate": int(moderate_threshold),
        "heavy": int(heavy_threshold)
    }


# ==============================================
# CSS样式函数
# ==============================================
def get_alert_css():
    """获取告警弹窗CSS样式（增强版）"""
    return """
    <style>
    /* 告警弹窗遮罩层 */
    .alert-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0,0,0,0.6);
        backdrop-filter: blur(8px);
        z-index: 10000;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: fadeIn 0.3s ease-out;
    }

    /* 弹窗卡片 */
    .alert-card {
        background: white;
        border-radius: 24px;
        max-width: 500px;
        width: 90%;
        box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
        animation: slideUp 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        overflow: hidden;
        border: 2px solid;
    }

    /* 弹窗头部 */
    .alert-header {
        padding: 18px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 2px solid #e5e7eb;
    }

    /* 弹窗内容 */
    .alert-content {
        padding: 24px;
        max-height: 450px;
        overflow-y: auto;
    }

    /* 弹窗底部 */
    .alert-footer {
        padding: 16px 24px;
        background: #f9fafb;
        display: flex;
        justify-content: flex-end;
        gap: 12px;
        border-top: 1px solid #e5e7eb;
    }

    /* 告警项卡片 - 增强闪烁效果 */
    .alert-item {
        background: white;
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 16px;
        border-left: 6px solid;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .alert-item:hover {
        transform: translateX(8px) scale(1.02);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }

    /* 状态灯 - 增强脉冲效果 */
    .status-light {
        display: inline-block;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        margin-right: 10px;
        animation: pulse-strong 0.8s infinite;
        box-shadow: 0 0 8px currentColor;
    }

    /* 紧急告警卡片专属样式 - 红色边框+闪烁背景 */
    .emergency-card {
        background: linear-gradient(135deg, #fff5f5 0%, #fee2e2 100%);
        border-left-color: #dc2626 !important;
        animation: emergencyPulse 0.6s infinite;
    }

    /* 重要告警卡片专属样式 */
    .alert-card-item {
        background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
        border-left-color: #f97316 !important;
        animation: alertPulse 0.8s infinite;
    }

    /* 增强动画定义 */
    @keyframes emergencyPulse {
        0%, 100% { background: linear-gradient(135deg, #fff5f5 0%, #fee2e2 100%); box-shadow: 0 0 0 0 rgba(220,38,38,0.7); }
        50% { background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); box-shadow: 0 0 0 15px rgba(220,38,38,0); }
    }

    @keyframes alertPulse {
        0%, 100% { background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%); box-shadow: 0 0 0 0 rgba(249,115,22,0.7); }
        50% { background: linear-gradient(135deg, #ffedd5 0%, #fed7aa 100%); box-shadow: 0 0 0 15px rgba(249,115,22,0); }
    }

    @keyframes pulse-strong {
        0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 0 0 currentColor; }
        50% { opacity: 0.4; transform: scale(1.3); box-shadow: 0 0 0 6px currentColor; }
    }

    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
        20%, 40%, 60%, 80% { transform: translateX(5px); }
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    @keyframes slideUp {
        from { transform: translateY(80px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    /* 批量确认按钮样式 */
    .batch-confirm-btn {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        cursor: pointer;
        font-size: 14px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .batch-confirm-btn:hover {
        background: #2563eb;
        transform: scale(1.03);
    }

    /* 标签样式 */
    .alert-type-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        margin-left: 8px;
    }

    /* 页面顶部横幅闪烁 */
    @keyframes topBannerFlash {
        0%, 100% { background-color: #dc2626; opacity: 0.95; }
        50% { background-color: #ef4444; opacity: 0.7; }
    }
    .top-alert-banner {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background: #dc2626;
        color: white;
        text-align: center;
        padding: 12px;
        font-size: 16px;
        font-weight: bold;
        z-index: 9999;
        animation: topBannerFlash 0.6s infinite;
        cursor: pointer;
    }
    </style>
    """


# 全局预加载告警样式，防止弹窗样式丢失
st.markdown(get_alert_css(), unsafe_allow_html=True)


# ==============================================
# 弹窗组件函数
# 修正版 show_alert_popup 核心部分
def show_alert_popup(alerts, alert_type="predict"):
    if not alerts:
        return

    # 先删除旧弹窗
    clear_script = """
    <script>
    document.querySelectorAll('#alertContainer').forEach(el => el.remove());
    if(window._alertAudioCtx) { window._alertAudioCtx.close(); window._alertAudioCtx = null; }
    window._alertSoundPlaying = false;
    </script>
    """
    st.components.v1.html(clear_script, height=0, width=0)

    # 排序告警
    sorted_alerts = sorted(alerts, key=lambda x: ALERT_GRADE.get(x["grade"], ALERT_GRADE["warning"])["priority"])
    has_emergency = any(alert["grade"] == "emergency" for alert in sorted_alerts)
    has_alert = any(alert["grade"] == "alert" for alert in sorted_alerts)

    # 样式变量
    if has_emergency:
        border_color = "#dc2626"
        header_bg = "linear-gradient(135deg, #dc2626 0%, #b91c1c 100%)"
    elif has_alert:
        border_color = "#f97316"
        header_bg = "linear-gradient(135deg, #f97316 0%, #ea580c 100%)"
    else:
        border_color = "#eab308"
        header_bg = "linear-gradient(135deg, #eab308 0%, #ca8a04 100%)"

    # 构建告警项HTML
    alerts_html = ""
    for idx, alert in enumerate(sorted_alerts):
        grade = alert["grade"]
        if grade == "emergency":
            card_bg = "#fef2f2"
            border_left_color = "#dc2626"
            text_color = "#dc2626"
            flash_animation = "emergencyFlash 0.6s infinite"
        elif grade == "alert":
            card_bg = "#fff7ed"
            border_left_color = "#f97316"
            text_color = "#f97316"
            flash_animation = "alertFlash 0.8s infinite"
        else:
            card_bg = "#fefce8"
            border_left_color = "#eab308"
            text_color = "#eab308"
            flash_animation = "warningFlash 1s infinite"

        animation_delay = idx * 0.1
        alerts_html += f"""
        <div class="alert-card-item" style="background: {card_bg}; border-left: 6px solid {border_left_color}; border-radius: 12px; padding: 16px; margin-bottom: 12px; animation: {flash_animation}, slideInRight 0.3s ease-out; animation-delay: {animation_delay}s;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div class="status-light" style="width: 14px; height: 14px; border-radius: 50%; background: {border_left_color}; animation: pulse-strong 0.6s infinite;"></div>
                    <span style="font-weight: 700; font-size: 16px; color: {text_color};">{'🔴' if grade == 'emergency' else '🟠' if grade == 'alert' else '🟡'} {ALERT_GRADE[grade]['name']}</span>
                </div>
                <span style="font-size: 12px; background: #f0f0f0; padding: 4px 12px; border-radius: 20px;">{alert.get('time', '')}</span>
            </div>
            <div style="margin-top: 10px; font-weight: 600; font-size: 15px; color: {text_color};">{alert.get('title', '')}</div>
            <div style="font-size: 13px; color: #555; margin-top: 4px;">📊 流量：{alert.get('message', '')}</div>
            <div style="font-size: 12px; color: #888; margin-top: 6px;">💡 {alert.get('suggestion', '')}</div>
            <div style="margin-top: 10px;">
                <div style="width: 100%; height: 4px; background: #e5e7eb; border-radius: 4px; overflow: hidden;">
                    <div style="width: 100%; height: 100%; background: {border_left_color}; animation: progressBar 3s linear forwards;"></div>
                </div>
            </div>
        </div>
        """

    # 关键修复：popup_html 不再用 Python f-string 包裹 JS 代码，改为用 format 或普通字符串
    popup_html = """
    <div id="alertContainer" style="position: fixed; top: 0; left: 0; right: 0; bottom: 0; z-index: 99999; display: flex; align-items: center; justify-content: center; background: rgba(0,0,0,0.7); backdrop-filter: blur(5px); animation: fadeIn 0.3s ease-out;">
        <div style="background: white; border-radius: 28px; max-width: 580px; width: 90%; max-height: 85vh; overflow: hidden; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5); border: 3px solid {border_color}; animation: popIn 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55), borderPulse 1s infinite;">
            <div style="padding: 20px 24px; background: {header_bg}; display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 36px; animation: shake 0.8s infinite;">{icon}</span>
                    <div>
                        <span style="font-weight: 800; font-size: 22px; color: white;">{title}</span>
                        <span style="display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 700; margin-left: 8px; background: rgba(255,255,255,0.2); color: white;">{type_name}</span>
                    </div>
                </div>
                <button onclick="closeAlert()" style="background: rgba(255,255,255,0.2); border: none; font-size: 24px; cursor: pointer; color: white; width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; transition: all 0.2s;">&times;</button>
            </div>
            <div style="padding: 20px; max-height: 55vh; overflow-y: auto;">
                <div style="margin-bottom: 16px; font-size: 14px; color: #666; display: flex; justify-content: space-between; align-items: center;">
                    <span>⚠️ 共 <b style="color: {border_color}; font-size: 18px;">{count}</b> 条告警</span>
                    <div style="display: flex; gap: 8px;">
                        <div style="width: 10px; height: 10px; border-radius: 50%; background: {border_color}; animation: pulse-strong 0.6s infinite;"></div>
                        <span style="font-size: 12px;">请及时处理</span>
                    </div>
                </div>
                {alerts_html}
            </div>
            <div style="padding: 16px 24px; background: #f9fafb; border-top: 1px solid #e5e7eb; display: flex; justify-content: flex-end; gap: 12px;">
                <button onclick="confirmAlerts()" class="btn-confirm" style="background: #10b981; color: white; border: none; border-radius: 40px; padding: 10px 24px; cursor: pointer; font-weight: 600; font-size: 14px; transition: all 0.2s;">✅ 批量确认</button>
                <button onclick="closeAlert()" class="btn-close" style="background: #6b7280; color: white; border: none; border-radius: 40px; padding: 10px 24px; cursor: pointer; font-weight: 600; font-size: 14px; transition: all 0.2s;">🔕 关闭</button>
            </div>
        </div>
    </div>

    <script>
        if(!window._alertAudioCtx) window._alertAudioCtx = null;
        window._alertSoundPlaying = true;

       function playAlertSound() {
    if(!window._alertSoundPlaying) return;
    try {
        if(window._alertAudioCtx) window._alertAudioCtx.close();
        window._alertAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
        var gainNode = window._alertAudioCtx.createGain();
        gainNode.connect(window._alertAudioCtx.destination);
        gainNode.gain.value = 0.35;

        var hasEmergency = {has_emergency};
        var hasAlert = {has_alert};

        // 存储循环定时器，用于关闭时销毁
        window._alertAudioLoop = null;
        window._alertAudioPlayingNow = true;

        if(hasEmergency) {
            // 紧急告警：持续高频循环滴滴声
            window._alertAudioLoop = setInterval(function() {
                if(!window._alertSoundPlaying || !window._alertAudioPlayingNow) {
                    clearInterval(window._alertAudioLoop);
                    return;
                }
                var osc = window._alertAudioCtx.createOscillator();
                osc.connect(gainNode);
                osc.frequency.value = 900;
                osc.start();
                osc.stop(window._alertAudioCtx.currentTime + 0.12);
            }, 200);
        } else if(hasAlert) {
            // 重要告警：中频循环提示音
            window._alertAudioLoop = setInterval(function() {
                if(!window._alertSoundPlaying || !window._alertAudioPlayingNow) {
                    clearInterval(window._alertAudioLoop);
                    return;
                }
                var osc = window._alertAudioCtx.createOscillator();
                osc.connect(gainNode);
                osc.frequency.value = 600;
                osc.start();
                osc.stop(window._alertAudioCtx.currentTime + 0.2);
            }, 400);
        } else {
            // 普通提醒：慢速循环
            window._alertAudioLoop = setInterval(function() {
                if(!window._alertSoundPlaying || !window._alertAudioPlayingNow) {
                    clearInterval(window._alertAudioLoop);
                    return;
                }
                var osc = window._alertAudioCtx.createOscillator();
                osc.connect(gainNode);
                osc.frequency.value = 523;
                osc.start();
                osc.stop(window._alertAudioCtx.currentTime + 0.3);
            }, 800);
        }
    } catch(e) { console.log("Audio error:", e); }
}

function stopSound() {
    window._alertSoundPlaying = false;
    window._alertAudioPlayingNow = false;
    // 清除循环定时器
    if(window._alertAudioLoop) clearInterval(window._alertAudioLoop);
    if(window._alertAudioCtx) {
        try { window._alertAudioCtx.close(); } catch(e) {}
    }
}



        function closeAlert() {
            stopSound();
            var container = document.getElementById('alertContainer');
            if(container) container.remove();
        }

        function confirmAlerts() {
            stopSound();
            closeAlert();
            window.dispatchEvent(new CustomEvent('alerts_confirmed', {{ detail: {{ confirmed: true }} }}));
        }

        playAlertSound();
        setTimeout(closeAlert, 30000);
    </script>
    """.format(
        border_color=border_color,
        header_bg=header_bg,
        icon='🚨' if alert_type == 'real_time' else '🔮',
        title='⚠️ 实时告警' if alert_type == 'real_time' else '🔮 预测告警',
        type_name=ALERT_TYPE_LABELS[alert_type]['name'],
        count=len(sorted_alerts),
        alerts_html=alerts_html,
        has_emergency=str(has_emergency).lower(),
        has_alert=str(has_alert).lower()
    )

    st.components.v1.html(popup_html, height=0, width=0)


def play_alert_sound(grade):
    """播放持续循环告警音效，仅页面刷新/手动关闭停止"""
    if grade == "emergency":
        # 紧急：快速持续滴滴
        sound_html = """
        <script>
            let audioCtx = null;
            let audioLoopTimer = null;
            let soundRunning = true;

            function playEmergencySound() {
                if(!soundRunning) return;
                try {
                    if(audioCtx) audioCtx.close();
                    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    var gainNode = audioCtx.createGain();
                    gainNode.connect(audioCtx.destination);
                    gainNode.gain.value = 0.35;

                    audioLoopTimer = setInterval(() => {
                        if(!soundRunning) {
                            clearInterval(audioLoopTimer);
                            return;
                        }
                        var osc = audioCtx.createOscillator();
                        osc.connect(gainNode);
                        osc.frequency.value = 900;
                        osc.start();
                        osc.stop(audioCtx.currentTime + 0.12);
                    }, 200);
                } catch(e) { console.log(e); }
            }
            playEmergencySound();
            // 全局标记，后续关闭弹窗可销毁
            window.globalAlertAudioTimer = audioLoopTimer;
            window.globalAlertAudioCtx = audioCtx;
        </script>
        """
    elif grade == "alert":
        # 重要告警：中等速度循环
        sound_html = """
        <script>
            let audioCtx = null;
            let audioLoopTimer = null;
            let soundRunning = true;

            function playAlertSound() {
                if(!soundRunning) return;
                try {
                    if(audioCtx) audioCtx.close();
                    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    var gainNode = audioCtx.createGain();
                    gainNode.connect(audioCtx.destination);
                    gainNode.gain.value = 0.3;

                    audioLoopTimer = setInterval(() => {
                        if(!soundRunning) {
                            clearInterval(audioLoopTimer);
                            return;
                        }
                        var osc = audioCtx.createOscillator();
                        osc.connect(gainNode);
                        osc.frequency.value = 600;
                        osc.start();
                        osc.stop(audioCtx.currentTime + 0.2);
                    }, 400);
                } catch(e) { console.log(e); }
            }
            playAlertSound();
            window.globalAlertAudioTimer = audioLoopTimer;
            window.globalAlertAudioCtx = audioCtx;
        </script>
        """
    else:
        # 普通提醒：慢速循环
        sound_html = """
        <script>
            let audioCtx = null;
            let audioLoopTimer = null;
            let soundRunning = true;

            function playWarningSound() {
                if(!soundRunning) return;
                try {
                    if(audioCtx) audioCtx.close();
                    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    var gainNode = audioCtx.createGain();
                    gainNode.connect(audioCtx.destination);
                    gainNode.gain.value = 0.25;

                    audioLoopTimer = setInterval(() => {
                        if(!soundRunning) {
                            clearInterval(audioLoopTimer);
                            return;
                        }
                        var osc = audioCtx.createOscillator();
                        osc.connect(gainNode);
                        osc.frequency.value = 523;
                        osc.start();
                        osc.stop(audioCtx.currentTime + 0.3);
                    }, 800);
                } catch(e) { console.log(e); }
            }
            playWarningSound();
            window.globalAlertAudioTimer = audioLoopTimer;
            window.globalAlertAudioCtx = audioCtx;
        </script>
        """
    st.components.v1.html(sound_html, height=0, width=0)


# ==============================================
# 初始化会话状态
# ==============================================
def init_alert_state():
    """初始化告警系统的会话状态"""
    if 'alert_config' not in st.session_state:
        st.session_state.alert_config = {
            "congestion_normal_max": 299,
            "congestion_light_min": 300,
            "congestion_light_max": 399,
            "congestion_moderate_min": 400,
            "congestion_moderate_max": 499,
            "congestion_heavy_min": 500,
            "surge_threshold": 50,
            "wind_speed_threshold": 8,
            "visibility_threshold": 1000,
        }
    if 'alert_log' not in st.session_state:
        st.session_state.alert_log = []
    if 'predict_alert_log' not in st.session_state:
        st.session_state.predict_alert_log = []
    if 'confirmed_alerts' not in st.session_state:
        st.session_state.confirmed_alerts = []


# ==============================================
# 天气数据合并函数
# ==============================================
def merge_weather_data(traffic_df, weather_df):
    """安全合并天气数据（兼容列名大小写、空值）"""
    try:
        if weather_df is None or weather_df.empty:
            return traffic_df

        traffic_df = traffic_df.copy()
        weather_df = weather_df.copy()

        # 统一列名，大小写兼容
        weather_df.columns = [col.strip().lower() for col in weather_df.columns]
        traffic_df.columns = [col.strip().lower() for col in traffic_df.columns]

        # 时间格式处理
        traffic_df['time'] = pd.to_datetime(traffic_df['time'], errors='coerce').dt.floor('h')
        weather_df['time'] = pd.to_datetime(weather_df['time'], errors='coerce').dt.floor('h')

        traffic_df = traffic_df.dropna(subset=['time'])
        weather_df = weather_df.dropna(subset=['time'])

        if traffic_df.empty or weather_df.empty:
            return traffic_df

        # 合并数据
        merged_df = pd.merge_asof(
            traffic_df.sort_values('time'),
            weather_df.sort_values('time'),
            on='time',
            direction='nearest'
        )
        return merged_df
    except Exception as e:
        st.warning(f"天气数据合并失败: {e}")
        return traffic_df


def get_congestion_level(flow, channel_history=None, channel_name="", config=None):
    """
    根据流量获取拥堵等级（支持动态自适应阈值）
    - flow: 当前流量
    - channel_history: 该航道的历史流量数据（用于计算动态阈值）
    - channel_name: 航道名称（用于日志）
    - config: 全局配置（备用）
    """
    # 优先使用历史数据计算动态阈值
    if channel_history is not None and len(channel_history) > 0:
        thresholds = calculate_adaptive_thresholds(channel_history, channel_name)

        if flow >= thresholds["heavy"]:
            return "heavy"
        elif flow >= thresholds["moderate"]:
            return "moderate"
        elif flow >= thresholds["light"]:
            return "light"
        else:
            return "normal"

    # 备用：使用全局配置
    if config is None:
        config = st.session_state.alert_config
    if flow <= config["congestion_normal_max"]:
        return "normal"
    elif config["congestion_light_min"] <= flow <= config["congestion_light_max"]:
        return "light"
    elif config["congestion_moderate_min"] <= flow <= config["congestion_moderate_max"]:
        return "moderate"
    elif flow >= config["congestion_heavy_min"]:
        return "heavy"
    return "normal"


def get_alert_level(congestion_level, surge_ratio=0, weather_data=None):
    """获取告警等级（修复版：安全访问天气字段）"""
    config = st.session_state.alert_config
    alert_level = "normal"

    # 拥堵等级映射
    if congestion_level == "light":
        alert_level = "warning"
    elif congestion_level == "moderate":
        alert_level = "alert"
    elif congestion_level == "heavy":
        alert_level = "emergency"

    # 流量突增升级
    if surge_ratio >= config["surge_threshold"]:
        if alert_level == "normal":
            alert_level = "warning"
        elif alert_level == "warning":
            alert_level = "alert"
        elif alert_level == "alert":
            alert_level = "emergency"

    # 天气因素升级（安全访问，使用 .get() 方法）
    if weather_data is not None and isinstance(weather_data, dict):
        wind_speed = weather_data.get("wind_speed", 0)
        visibility = weather_data.get("visibility", 9999)

        # 处理可能的 NaN 值
        if pd.isna(wind_speed):
            wind_speed = 0
        if pd.isna(visibility):
            visibility = 9999

        if wind_speed > config["wind_speed_threshold"] or visibility < config["visibility_threshold"]:
            if alert_level == "normal":
                alert_level = "warning"
            elif alert_level == "warning":
                alert_level = "alert"

    return alert_level


def calculate_hourly_surge(history_data):
    """计算流量突增比例"""
    if len(history_data) < 2:
        return 0.0
    last = history_data[-1]
    prev = history_data[-2]
    if prev == 0:
        return 0.0
    return max(0, (last - prev) / prev * 100)


def add_alert_log(channel_id, flow, congestion_level, alert_level, alert_type, timestamp=None):
    """添加实时告警日志"""
    if timestamp is None:
        timestamp = datetime.now()
    st.session_state.alert_log.append({
        "timestamp": timestamp,
        "channel_id": channel_id,
        "traffic_flow": flow,
        "congestion_level": congestion_level,
        "alert_level": alert_level,
        "alert_type": alert_type
    })
    # 限制日志数量
    if len(st.session_state.alert_log) > 1000:
        st.session_state.alert_log = st.session_state.alert_log[-1000:]


def add_predict_alert_log(channel_id, flow, congestion_level, alert_level, alert_type, predict_time):
    """添加预测告警日志（修复版：确保所有字段正确写入）"""
    log_entry = {
        "predict_time": predict_time,
        "channel_id": channel_id,
        "traffic_flow": flow,
        "congestion_level": congestion_level,
        "alert_level": alert_level,
        "alert_type": alert_type,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 新增日志生成时间
    }
    st.session_state.predict_alert_log.append(log_entry)

    # 限制日志数量，防止内存溢出
    if len(st.session_state.predict_alert_log) > 1000:
        st.session_state.predict_alert_log = st.session_state.predict_alert_log[-1000:]

# ==============================================
# 侧边栏配置
# ==============================================
def show_alert_sidebar():
    """在侧边栏显示告警配置"""
    with st.sidebar:
        st.markdown("### 🚨 告警阈值配置")
        cfg = st.session_state.alert_config




        col1, col2 = st.columns(2)
        with col1:
            normal_max = st.number_input("正常上限", 100, 400, cfg["congestion_normal_max"])
            light_min = st.number_input("轻度下限", normal_max + 1, 400, cfg["congestion_light_min"])
            moderate_min = st.number_input("中度下限", light_min + 1, 500, cfg["congestion_moderate_min"])
            heavy_min = st.number_input("严重下限", moderate_min + 1, 800, cfg["congestion_heavy_min"])
        with col2:
            light_max = st.number_input("轻度上限", light_min, moderate_min - 1, cfg["congestion_light_max"])
            moderate_max = st.number_input("中度上限", moderate_min, heavy_min - 1, cfg["congestion_moderate_max"])
            surge_threshold = st.number_input("流量突增阈值(%)", 10, 200, cfg["surge_threshold"])

        st.markdown("#### 天气告警阈值")
        wind_threshold = st.number_input("风速阈值(m/s)", 0, 30, cfg["wind_speed_threshold"])
        vis_threshold = st.number_input("能见度阈值(m)", 100, 5000, cfg["visibility_threshold"])


        if st.button("💾 保存配置", type="primary"):
            st.session_state.alert_config.update({
                "congestion_normal_max": normal_max,
                "congestion_light_min": light_min,
                "congestion_light_max": light_max,
                "congestion_moderate_min": moderate_min,
                "congestion_moderate_max": moderate_max,
                "congestion_heavy_min": heavy_min,
                "surge_threshold": surge_threshold,
                "wind_speed_threshold": wind_threshold,
                "visibility_threshold": vis_threshold,
            })
            st.success("✅ 配置已保存")


def show_predict_alert_panel(pred_df, channel_name, congestion_config=None, channel_history=None, sound_enabled=None):
    """
    预测告警面板【修复版：批量确认一次生效，保持按钮原位置】
    """
    if pred_df is None or pred_df.empty:
        return

    # 如果没有传入声音开关状态，从 session_state 获取
    if sound_enabled is None:
        sound_enabled = st.session_state.get('sound_enabled', False)

    # --- 1. 计算所有告警数据 ---
    use_adaptive = channel_history is not None and len(channel_history) > 0
    if use_adaptive:
        thresholds = calculate_adaptive_thresholds(channel_history, channel_name)
        light_min = thresholds["light"]
        moderate_min = thresholds["moderate"]
        heavy_min = thresholds["heavy"]
        with st.expander(f"📊 {channel_name} 自适应拥堵标准"):
            st.info(f"""
            - 🟡 **轻度拥堵**: ≥ {light_min} 艘/小时
            - 🟠 **中度拥堵**: ≥ {moderate_min} 艘/小时
            - 🔴 **严重拥堵**: ≥ {heavy_min} 艘/小时
            """)
    else:
        if congestion_config is not None:
            light_min = congestion_config["light"]["min"]
            moderate_min = congestion_config["moderate"]["min"]
            heavy_min = congestion_config["heavy"]["min"]
        else:
            light_min = CONGESTION["light"]["min"]
            moderate_min = CONGESTION["moderate"]["min"]
            heavy_min = CONGESTION["heavy"]["min"]

    alert_predictions = []
    moderate_alerts = []
    heavy_alerts = []

    for _, row in pred_df.iterrows():
        flow = row["预测流量"]
        display_time = row["显示时间"] if "显示时间" in row else row["时间"].strftime("%Y-%m-%d %H:%M")
        if flow >= heavy_min:
            cong_level = "heavy"
        elif flow >= moderate_min:
            cong_level = "moderate"
        else:
            cong_level = "normal"
        if cong_level == "moderate":
            alert_entry = {
                "时间": display_time,
                "预测流量": f"{flow}艘/时",
                "拥堵等级": "moderate",
                "建议": "建议提前疏导，准备应急预案",
                "grade": "alert"
            }
            moderate_alerts.append(alert_entry)
            add_predict_alert_log(channel_id=channel_name, flow=flow, congestion_level=cong_level,
                                   alert_level="alert", alert_type="预测拥堵", predict_time=display_time)
        elif cong_level == "heavy":
            alert_entry = {
                "时间": display_time,
                "预测流量": f"{flow}艘/时",
                "拥堵等级": "heavy",
                "建议": "紧急！立即启动应急响应",
                "grade": "emergency"
            }
            heavy_alerts.append(alert_entry)
            add_predict_alert_log(channel_id=channel_name, flow=flow, congestion_level=cong_level,
                                   alert_level="emergency", alert_type="预测拥堵", predict_time=display_time)

    alert_predictions = moderate_alerts + heavy_alerts

    # 初始化横幅确认状态（使用航道名称作为key）
    banner_key = f'banner_confirmed_{channel_name}'
    if banner_key not in st.session_state:
        st.session_state[banner_key] = False

    # --- 预测结果内容 ---
    if alert_predictions:
        emergency_count = len(heavy_alerts)
        alert_count = len(moderate_alerts)
        total_alerts = len(alert_predictions)

        # 顶部告警横幅（只在未确认时显示闪烁）
        if not st.session_state[banner_key]:
            st.markdown("""
            <style>
            @keyframes blinkRed {
                0% { background: linear-gradient(135deg, #dc2626, #b91c1c); }
                50% { background: linear-gradient(135deg, #ef4444, #dc2626); box-shadow: 0 0 30px rgba(220,38,38,0.8); }
                100% { background: linear-gradient(135deg, #dc2626, #b91c1c); }
            }
            @keyframes blinkOrange {
                0% { background: linear-gradient(135deg, #f97316, #ea580c); }
                50% { background: linear-gradient(135deg, #fb923c, #f97316); box-shadow: 0 0 20px rgba(249,115,22,0.6); }
                100% { background: linear-gradient(135deg, #f97316, #ea580c); }
            }
            .alert-banner-heavy {
                background: linear-gradient(135deg, #dc2626, #b91c1c);
                color: white;
                padding: 20px;
                text-align: center;
                border-radius: 12px;
                font-size: 24px;
                font-weight: bold;
                animation: blinkRed 0.5s infinite;
                box-shadow: 0 0 20px rgba(220,38,38,0.5);
                margin: 15px 0;
            }
            .alert-banner-moderate {
                background: linear-gradient(135deg, #f97316, #ea580c);
                color: white;
                padding: 15px;
                text-align: center;
                border-radius: 10px;
                font-size: 18px;
                font-weight: bold;
                animation: blinkOrange 0.8s infinite;
                box-shadow: 0 0 15px rgba(249,115,22,0.4);
                margin: 10px 0;
            }
            </style>
            """, unsafe_allow_html=True)

            if heavy_alerts:
                st.markdown(f"""
                <div class="alert-banner-heavy">
                    🚨🚨🚨 紧急告警！{len(heavy_alerts)} 个严重拥堵预测！ 🚨🚨🚨
                </div>
                """, unsafe_allow_html=True)
            elif moderate_alerts:
                st.markdown(f"""
                <div class="alert-banner-moderate">
                    ⚠️⚠️⚠️ 重要告警！{len(moderate_alerts)} 个中度拥堵预测！ ⚠️⚠️⚠️
                </div>
                """, unsafe_allow_html=True)
        else:
            # 已确认，显示静态提示（不闪烁）
            if heavy_alerts:
                st.info(f"✅ 已确认：{len(heavy_alerts)} 个严重拥堵预测已处理")
            elif moderate_alerts:
                st.success(f"✅ 已确认：{len(moderate_alerts)} 个中度拥堵预测已处理")

        # 评级卡片
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔴 一级紧急", emergency_count, delta="立即处置" if emergency_count > 0 else None,
                      delta_color="inverse")
        with col2:
            st.metric("🟠 二级重要", alert_count, delta="需关注" if alert_count > 0 else None)
        with col3:
            st.metric("📊 告警总数", total_alerts)

        # 警告文字（根据确认状态显示不同内容）
        if not st.session_state[banner_key]:
            st.warning(f"⚠️ 未来共有 {total_alerts} 个拥堵时段需要关注")
        else:
            st.success(f"✅ 所有 {total_alerts} 个拥堵时段已确认处理")

        # 告警列表
        st.markdown("#### 📋 详细告警列表")
        for alert in alert_predictions:
            grade = alert["grade"]
            if grade == "emergency":
                bg_color = "#fef2f2"
                border_color = "#dc2626"
                icon = "🔴"
                title = "紧急告警"
            else:
                bg_color = "#fff7ed"
                border_color = "#f97316"
                icon = "🟠"
                title = "重要告警"

            st.markdown(f"""
            <div style="
                background: {bg_color}; 
                border-left: 4px solid {border_color}; 
                border-radius: 8px; 
                padding: 12px; 
                margin-bottom: 8px; 
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            ">
                <div style="display: flex; justify-content: space-between;">
                    <span><b>{icon} {title}</b></span>
                    <span style="font-size: 12px;">{alert['时间']}</span>
                </div>
                <div>📊 预测流量：<b>{alert['预测流量']}</b></div>
                <div style="font-size: 13px;">💡 {alert['建议']}</div>
            </div>
            """, unsafe_allow_html=True)

        # 音效播放（仅未确认且未播放过时播放）
        if sound_enabled and not st.session_state[banner_key]:
            sound_played_key = f'sound_played_{channel_name}'
            if sound_played_key not in st.session_state or not st.session_state[sound_played_key]:
                st.session_state[sound_played_key] = True
                sound_html = '''
                <script>
                    let audioCtx = null;
                    let loopTimer = null;
                    let playFlag = true;
                    function loopAlert() {
                        if(!playFlag) return;
                        try {
                            if(audioCtx) audioCtx.close();
                            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                            var gain = audioCtx.createGain();
                            gain.connect(audioCtx.destination);
                            gain.gain.value = 0.3;
                            loopTimer = setInterval(()=>{
                                if(!playFlag) clearInterval(loopTimer);
                                var osc = audioCtx.createOscillator();
                                osc.connect(gain);
                                osc.frequency.value = 700;
                                osc.start();
                                osc.stop(audioCtx.currentTime + 0.15);
                            },300)
                        }catch(e){}
                    }
                    loopAlert();
                    window.predictAudioTimer = loopTimer;
                    window.predictAudioCtx = audioCtx;
                </script>
                '''

                st.components.v1.html(sound_html, height=0, width=0)

        # ========== 批量确认按钮（保持原位置，一次生效）==========
        if not st.session_state[banner_key]:
            if st.button("✅ 批量确认告警", key=f"batch_confirm_{channel_name}", use_container_width=True, type="primary"):
                st.session_state[banner_key] = True
                for alert in alert_predictions:
                    st.session_state.confirmed_alerts.append({
                        "time": alert["时间"],
                        "channel": channel_name,
                        "flow": alert["预测流量"],
                        "level": alert["拥堵等级"],
                        "confirmed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                st.rerun()  # 强制刷新，让横幅和按钮状态立即更新
        else:
            # 已确认后显示重置按钮（保持原位置）
            if st.button("🔄 重置横幅", key=f"reset_banner_{channel_name}", use_container_width=True):
                st.session_state[banner_key] = False
                # 重置音效播放标记
                sound_played_key = f'sound_played_{channel_name}'
                if sound_played_key in st.session_state:
                    del st.session_state[sound_played_key]
                toast_played_key = f'toast_played_{channel_name}'
                if toast_played_key in st.session_state:
                    del st.session_state[toast_played_key]
                st.rerun()

        # Toast 提示（仅未确认且未显示过时显示）
        toast_played_key = f'toast_played_{channel_name}'
        if not st.session_state[banner_key] and (toast_played_key not in st.session_state or not st.session_state[toast_played_key]):
            st.session_state[toast_played_key] = True
            for alert in alert_predictions[:3]:
                if alert["grade"] == "emergency":
                    st.toast(f"🚨 {alert['拥堵等级']}：航道 {channel_name} 预测流量 {alert['预测流量']}", icon="🚨")
                else:
                    st.toast(f"⚠️ {alert['拥堵等级']}：航道 {channel_name} 预测流量 {alert['预测流量']}", icon="⚠️")

    else:
        st.success("✅ 未来时段无中度/重度拥堵预测")


def show_alert_log_panel(panel_unique_id: str):
    """显示实时告警日志【修复：增加唯一标识动态生成下载key】"""
    st.markdown("### 📜 实时告警日志")
    logs = st.session_state.alert_log

    if logs:
        log_df = pd.DataFrame(logs)
        log_df["timestamp"] = pd.to_datetime(log_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")

        # 映射等级显示
        cong_map = {"normal": "正常", "light": "轻度拥堵", "moderate": "中度拥堵", "heavy": "严重拥堵"}
        alert_map = {"normal": "正常", "warning": "提醒", "alert": "预警", "emergency": "紧急"}

        log_df["拥堵等级"] = log_df["congestion_level"].map(cong_map)
        log_df["告警等级"] = log_df["alert_level"].map(alert_map)

        display_cols = ["timestamp", "channel_id", "traffic_flow", "拥堵等级", "告警等级", "alert_type"]
        st.dataframe(log_df[display_cols], use_container_width=True, hide_index=True)

        # 导出按钮 - 动态唯一key
        csv = log_df.to_csv(index=False, encoding="utf-8-sig")
        unique_key = f"download_real_time_alert_log_{panel_unique_id}"
        st.download_button(
            label="📥 导出实时日志",
            data=csv,
            file_name=f"real_time_alert_log_{panel_unique_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv",
            mime="text/csv",
            key=unique_key
        )
    else:
        st.info("暂无实时告警记录")


def show_predict_alert_log_panel(panel_unique_id: str):
    """显示预测告警日志（修复版：支持新增字段 + 动态唯一下载key）"""
    st.markdown("### 📜 预测告警日志")
    logs = st.session_state.predict_alert_log

    if logs:
        log_df = pd.DataFrame(logs)

        # 确保时间格式正确
        if "predict_time" in log_df.columns:
            log_df["predict_time"] = pd.to_datetime(log_df["predict_time"], errors='coerce').dt.strftime(
                "%Y-%m-%d %H:%M")

        # 映射等级显示
        cong_map = {"normal": "正常", "light": "轻度拥堵", "moderate": "中度拥堵", "heavy": "严重拥堵"}
        alert_map = {"normal": "正常", "warning": "提醒", "alert": "预警", "emergency": "紧急"}

        log_df["拥堵等级"] = log_df["congestion_level"].map(cong_map)
        log_df["告警等级"] = log_df["alert_level"].map(alert_map)

        # 按时间倒序显示最新日志
        display_cols = ["predict_time", "channel_id", "traffic_flow", "拥堵等级", "告警等级", "alert_type"]
        if "timestamp" in log_df.columns:
            display_cols.append("timestamp")
            log_df = log_df.sort_values("timestamp", ascending=False)
        else:
            log_df = log_df.sort_values("predict_time", ascending=False)

        st.dataframe(log_df[display_cols], use_container_width=True, hide_index=True)

        # 导出按钮 - 独立前缀动态唯一key
        csv = log_df.to_csv(index=False, encoding="utf-8-sig")
        unique_key = f"download_predict_alert_log_{panel_unique_id}"
        st.download_button(
            label="📥 导出预测日志",
            data=csv,
            file_name=f"predict_alert_log_{panel_unique_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv",
            mime="text/csv",
            key=unique_key
        )
    else:
        st.info("暂无预测告警记录")


# ==============================================
# 天气影响分析面板
# ==============================================
def show_weather_analysis_panel(traffic_df, weather_df):
    """天气对通航流量的影响分析（完整修复版）"""
    st.markdown("### 🌤️ 天气对通航流量的影响分析")

    if traffic_df is None or traffic_df.empty:
        st.info("暂无交通流量数据")
        return

    if weather_df is None or weather_df.empty:
        st.info("请上传天气数据以查看分析")
        return

    try:
        # 确保数据副本
        traffic_copy = traffic_df.copy()
        weather_copy = weather_df.copy()

        # 统一时间格式
        traffic_copy['time'] = pd.to_datetime(traffic_copy['time'], errors='coerce')
        weather_copy['time'] = pd.to_datetime(weather_copy['time'], errors='coerce')

        # 对齐到小时
        traffic_copy['hour'] = traffic_copy['time'].dt.floor('h')
        weather_copy['hour'] = weather_copy['time'].dt.floor('h')

        # 合并数据
        merged_df = pd.merge(
            traffic_copy,
            weather_copy,
            on='hour',
            how='inner',
            suffixes=('', '_weather')
        )

        if merged_df.empty:
            st.info("交通数据与天气数据时间不匹配，无法进行关联分析")
            return

        st.success(f"✅ 成功匹配 {len(merged_df)} 条天气-流量关联数据")

        # 风速 vs 流量
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 💨 风速 vs 流量")
            # 检查风速字段（多种可能的列名）
            wind_col = None
            for col in ['wind_speed', 'windspeed', 'wind', '风速']:
                if col in merged_df.columns:
                    wind_col = col
                    break

            if wind_col and 'traffic_flow' in merged_df.columns:
                # 过滤有效数据
                plot_df = merged_df[[wind_col, 'traffic_flow']].dropna()
                plot_df = plot_df[plot_df[wind_col] >= 0]

                if not plot_df.empty:
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(
                        x=plot_df[wind_col],
                        y=plot_df['traffic_flow'],
                        mode='markers',
                        marker=dict(color='#5470c6', size=8, opacity=0.6),
                        name='数据点'
                    ))
                    fig1.update_layout(
                        xaxis_title='风速(m/s)',
                        yaxis_title='流量(艘/时)',
                        height=350
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                else:
                    st.info("暂无有效的风速数据")
            else:
                st.info("暂无风速数据字段")

        # 能见度 vs 流量
        with col2:
            st.markdown("#### 👁️ 能见度 vs 流量")
            vis_col = None
            for col in ['visibility', 'vis', '能见度']:
                if col in merged_df.columns:
                    vis_col = col
                    break

            if vis_col and 'traffic_flow' in merged_df.columns:
                plot_df = merged_df[[vis_col, 'traffic_flow']].dropna()
                plot_df = plot_df[plot_df[vis_col] >= 0]

                if not plot_df.empty:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=plot_df[vis_col],
                        y=plot_df['traffic_flow'],
                        mode='markers',
                        marker=dict(color='#2ca02c', size=8, opacity=0.6),
                        name='数据点'
                    ))
                    fig2.update_layout(
                        xaxis_title='能见度(m)',
                        yaxis_title='流量(艘/时)',
                        height=350
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("暂无有效的能见度数据")
            else:
                st.info("暂无能见度数据字段")

        # 不同天气的平均流量
        st.markdown("---")
        st.markdown("#### 🌈 不同天气平均通航流量")

        weather_col = None
        for col in ['weather', '天气', 'condition', 'Weather']:
            if col in merged_df.columns:
                weather_col = col
                break

        if weather_col and 'traffic_flow' in merged_df.columns:
            # 过滤有效数据
            plot_df = merged_df[[weather_col, 'traffic_flow']].dropna()

            if not plot_df.empty:
                weather_avg = plot_df.groupby(weather_col)['traffic_flow'].agg(['mean', 'count']).reset_index()
                weather_avg.columns = [weather_col, '平均流量', '样本数']
                weather_avg['平均流量'] = weather_avg['平均流量'].round(0)

                fig3 = go.Figure()
                fig3.add_trace(go.Bar(
                    x=weather_avg[weather_col].astype(str),
                    y=weather_avg['平均流量'],
                    marker_color='#FF7D00',
                    text=weather_avg['平均流量'],
                    textposition='auto',
                    name='平均流量'
                ))
                fig3.update_layout(
                    xaxis_title='天气状况',
                    yaxis_title='平均流量(艘/时)',
                    height=400
                )
                st.plotly_chart(fig3, use_container_width=True)

                # 显示详细表格
                with st.expander("📊 详细数据"):
                    st.dataframe(weather_avg, use_container_width=True)
            else:
                st.info("暂无有效的天气数据")
        else:
            st.info("天气数据中缺少 'weather' 字段，无法进行天气分类分析")
            st.info(f"当前天气数据字段：{list(weather_copy.columns)}")

    except Exception as e:
        st.error(f"天气影响分析出错: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


# ==============================================
# 实时告警面板（完整修复版）
# ==============================================
# ==============================================
# 实时告警面板（最终修改版：仅移除实时顶部横幅、缩小实时高危框，预测面板无改动）
# ==============================================
# ==============================================
# 实时告警面板（最终调整版：移除顶部横幅，缩小下方告警框匹配原横幅尺寸）
# ==============================================
def show_real_time_alert_panel(df, weather_df=None, congestion_config=None):
    """显示实时告警面板（完整修复版：移除顶部全局横幅、调整下方高危框尺寸对齐原横幅）"""
    # 更新拥堵配置
    if congestion_config is not None:
        global CONGESTION
        CONGESTION = congestion_config

    # 1. 前置空值检查
    if df is None or df.empty:
        st.info("暂无数据，无法显示告警")
        return

    # 确保必要的列存在
    if "time" not in df.columns:
        if "timestamp" in df.columns:
            df.rename(columns={"timestamp": "time"}, inplace=True)
        elif "show_time" in df.columns:
            df.rename(columns={"show_time": "time"}, inplace=True)
        else:
            st.warning("主数据中无时间列，无法显示告警")
            return

    if "traffic_flow" not in df.columns:
        if "流量" in df.columns:
            df.rename(columns={"流量": "traffic_flow"}, inplace=True)
        elif "flow" in df.columns:
            df.rename(columns={"flow": "traffic_flow"}, inplace=True)

    if "channel_id" not in df.columns:
        if "航道" in df.columns:
            df.rename(columns={"航道": "channel_id"}, inplace=True)
        elif "channel" in df.columns:
            df.rename(columns={"channel": "channel_id"}, inplace=True)

    # 转换时间
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])

    if df.empty:
        st.warning("无有效时间数据")
        return

    # 计算最新告警数据
    latest = df.sort_values("time").groupby("channel_id").last().reset_index()
    alert_list = []
    critical_count = 0

    for _, row in latest.iterrows():
        ch = row["channel_id"]
        flow = row["traffic_flow"]
        time = row["time"]

        # 计算突增比例
        hist = df[df["channel_id"] == ch]["traffic_flow"].tolist()
        surge = calculate_hourly_surge(hist)

        # ========== 修复：安全获取天气数据 ==========
        weather = None
        wind_speed = 0
        visibility = 9999
        weather_condition = None

        if weather_df is not None and not weather_df.empty:
            try:
                wdf = weather_df.copy()
                wdf["time"] = pd.to_datetime(wdf["time"], errors="coerce")
                wdf = wdf.dropna(subset=["time"])
                if not wdf.empty:
                    weather_time = wdf["time"].dt.floor("h")
                    current_time = time.floor("h")
                    matched = wdf[weather_time == current_time]
                    if not matched.empty:
                        # 获取匹配的行（作为 Series）
                        matched_row = matched.iloc[0]

                        # 安全提取风速（支持多种列名）
                        for col in ["wind_speed", "windspeed", "wind", "风速"]:
                            if col in matched_row.index:
                                wind_speed = matched_row[col]
                                if pd.isna(wind_speed):
                                    wind_speed = 0
                                break

                        # 安全提取能见度
                        for col in ["visibility", "vis", "能见度"]:
                            if col in matched_row.index:
                                visibility = matched_row[col]
                                if pd.isna(visibility):
                                    visibility = 9999
                                break

                        # 安全提取天气状况（关键修复：支持 weather 字段）
                        for col in ["weather", "天气", "condition"]:
                            if col in matched_row.index:
                                weather_condition = matched_row[col]
                                break

                        # 构建字典供其他函数使用
                        weather = {
                            "wind_speed": wind_speed,
                            "visibility": visibility,
                            "weather": weather_condition
                        }
            except Exception as e:
                # 静默失败，不影响主流程
                pass

        # 获取等级
        # 获取该航道的历史数据
        channel_hist = df[df["channel_id"] == ch]["traffic_flow"].tolist()
        cong_level = get_congestion_level(flow, channel_hist, ch, congestion_config)
        alert_level = get_alert_level(cong_level, surge, weather)

        if alert_level in ["alert", "emergency"]:
            critical_count += 1

        # 确定告警类型
        types = []
        if cong_level != "normal":
            types.append("拥堵")
        if surge >= st.session_state.alert_config["surge_threshold"]:
            types.append("流量突增")
        if weather is not None:
            if wind_speed > st.session_state.alert_config["wind_speed_threshold"]:
                types.append("大风")
            if visibility < st.session_state.alert_config["visibility_threshold"]:
                types.append("低能见度")
        alert_type = "|".join(types) if types else "无"

        # 添加日志
        if alert_level != "normal":
            add_alert_log(ch, flow, cong_level, alert_level, alert_type, time)

        # 等级映射
        cong_label_map = {"normal": "正常", "light": "轻度拥堵", "moderate": "中度拥堵", "heavy": "严重拥堵"}
        alert_label_map = {"normal": "正常", "warning": "提醒", "alert": "预警", "emergency": "紧急"}

        alert_list.append({
            "航道": ch,
            "流量": f"{flow}艘/时",
            "拥堵": cong_label_map.get(cong_level, "正常"),
            "告警": alert_label_map.get(alert_level, "正常"),
            "类型": alert_type,
            "时间": time.strftime("%m-%d %H:%M"),
            "color": ALERT_LEVEL.get(alert_level, ALERT_LEVEL["normal"])["color"],
            "level_key": alert_level
        })

    # ========== 【已彻底删除顶部全局横幅代码块】 ==========

    # ========== 告警表格 ==========
    st.markdown("### 🚨 实时告警面板")

    if alert_list:
        cols = st.columns([1, 1.5, 1.5, 1.2, 1.5, 1.8])
        heads = ["航道", "流量", "拥堵等级", "告警等级", "告警类型", "时间"]
        for c, h in zip(cols, heads):
            c.markdown(f"**{h}**")

        for a in alert_list:
            c1, c2, c3, c4, c5, c6 = st.columns([1, 1.5, 1.5, 1.2, 1.5, 1.8])
            c1.markdown(f"<span style='color:{a['color']}'>{a['航道']}</span>", unsafe_allow_html=True)
            c2.write(a["流量"])
            c3.write(a["拥堵"])
            c4.write(a["告警"])
            c5.write(a["类型"])
            c6.write(a["时间"])
            st.divider()
    else:
        st.info("✅ 当前无告警，所有航道运行正常")

    # ========== 实时高危告警框（调整样式：横向通栏、高度和原顶部横幅匹配） ==========
    critical = [a for a in alert_list if a["level_key"] in ["alert", "emergency"]]
    if critical:
        st.markdown(f"""
        <style>
        @keyframes flash-border-red {{
            0% {{ box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.8); }}
            50% {{ box-shadow: 0 0 0 15px rgba(255, 0, 0, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.8); }}
        }}
        .alert-box-red {{
            animation: flash-border-red 0.8s infinite;
            background: #ffebee;
            border: 2px solid #ff0000;
            border-radius: 8px;
            padding: 12px 16px; /* 调整内边距，高度对齐原顶部横幅 */
            margin: 10px 0;
            text-align: center; /* 文字居中，和原来横幅样式统一 */
        }}
        .alert-box-red h3 {{
            margin: 0;
            font-size: 20px;
            display: inline;
            margin-right: 8px;
        }}
        .alert-box-red p {{
            font-size:20px;
            margin: 0;
            display: inline;
        }}
        </style>
        <div class="alert-box-red">
            <h3 style="color: #d32f2f;">⚠️ 实时高危告警</h3>
            <p>当前共有 <b>{len(critical)}</b> 个航道处于【预警/紧急】状态，请立即处置！</p>
        </div>
        """, unsafe_allow_html=True)

# ==============================================
# 对外暴露接口
# ==============================================
__all__ = [
    'init_alert_state',
    'show_alert_sidebar',
    'show_real_time_alert_panel',
    'show_predict_alert_panel',
    'show_alert_log_panel',
    'show_predict_alert_log_panel',
    'show_weather_analysis_panel',
    'merge_weather_data',
    'calculate_adaptive_thresholds',
    'show_alert_popup',
    'play_alert_sound',
    'get_alert_css',
    'ALERT_GRADE',
    'ALERT_TYPE_LABELS',
    'ALERT_LEVEL',
    'CONGESTION'
]