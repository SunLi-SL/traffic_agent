# trajectory_func.py
# 船舶轨迹预测 & 会遇风险预警 独立功能文件
import math
import json
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from alert_system import show_alert_log_panel
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from alert_system import show_alert_log_panel

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return super().default(obj)


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlon = lon2_rad - lon1_rad
    x = math.cos(lat2_rad) * math.sin(dlon)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def predict_ship_trajectory(ship_history, predict_min=10, sample_interval=30):
    if len(ship_history) < 2:
        return [], []

    ship_history = sorted(ship_history, key=lambda x: x.get('time', datetime.min))
    if not ship_history:
        # 无历史船舶数据，直接返回空，避免下标报错
        return None
    latest = ship_history[-1]
    curr_lat = latest.get("lat")
    curr_lon = latest.get("lon")
    curr_speed = latest.get("speed", 10)

    if curr_lat is None or curr_lon is None:
        return [], []

    curr_heading = latest.get("heading")
    if curr_heading is None or pd.isna(curr_heading):
        if len(ship_history) >= 2:
            prev = ship_history[-2]
            if prev.get("lat") is not None and prev.get("lon") is not None:
                curr_heading = calculate_bearing(prev["lat"], prev["lon"], curr_lat, curr_lon)
        if curr_heading is None:
            curr_heading = 0

    start_time = latest.get("time")
    if start_time is None:
        start_time = datetime.now()

    speed_mps = curr_speed * 0.5144
    total_seconds = predict_min * 60
    step_sec = sample_interval

    history_points = []
    for p in ship_history:
        if p.get("lat") is not None and p.get("lon") is not None:
            history_points.append({
                "lat": float(p["lat"]),
                "lon": float(p["lon"]),
                "time": p.get("time", start_time)
            })

    predict_points = []
    current_lat, current_lon = curr_lat, curr_lon
    heading_rad = math.radians(curr_heading)
    earth_radius = 6371000

    for sec in range(step_sec, total_seconds + 1, step_sec):
        move_dist = speed_mps * step_sec
        d_lat = (move_dist / earth_radius) * (180 / math.pi) * math.cos(heading_rad)
        d_lon = (move_dist / (earth_radius * math.cos(math.radians(current_lat)))) * (180 / math.pi) * math.sin(
            heading_rad)
        current_lat += d_lat
        current_lon += d_lon
        pred_time = start_time + timedelta(seconds=sec)
        predict_points.append({
            "lat": round(current_lat, 6),
            "lon": round(current_lon, 6),
            "time": pred_time,
            "speed": curr_speed,
            "heading": curr_heading
        })

    return history_points, predict_points


def detect_encounter(traj1, traj2, safe_distance=200):
    if not traj1 or not traj2:
        return False, float("inf"), None
    min_dist = float("inf")
    risk_time = None
    for p1 in traj1:
        for p2 in traj2:
            dist = haversine_distance(p1["lat"], p1["lon"], p2["lat"], p2["lon"])
            if dist < min_dist:
                min_dist = dist
                risk_time = max(p1.get("time", datetime.min), p2.get("time", datetime.min))
    return min_dist < safe_distance, round(min_dist, 1), risk_time


TRAJ_STYLE = """
<style>
.alert-box-red {
    background: #ffebee;
    border: 2px solid #ff0000;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    animation: flash-border-red 0.8s infinite;
}
@keyframes flash-border-red {
    0% { box-shadow: 0 0 0 0 rgba(255, 0, 0.8); }
    50% { box-shadow: 0 0 0 15px rgba(255, 0, 0); }
    100% { box-shadow: 0 0 0 0 rgba(255, 0, 0.8); }
}
.card {
    background: #ffffff;
    padding: 24px;
    border-radius: 24px;
    border: 1px solid #e0e4e8;
    margin-bottom: 28px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.stat-item {
    flex: 1;
    min-width: 140px;
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    padding: 16px;
    border-radius: 12px;
    text-align: center;
    color: #0369a1;
    border: 1px solid #bae6fd;
    transition: all 0.3s;
}
.stat-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.stat-value {
    font-size: 26px;
    font-weight: 800;
    color: #0c4a6e;
}
/* 使用独立的类名，避免与主页面冲突 */
.traj-stat-label {
    font-size: 13px;
    margin-top: 6px;
    opacity: 0.8;
    color: #6c757d;
}
</style>
"""


def render_trajectory(raw_data, amap_key):
    st.markdown(TRAJ_STYLE, unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🚢 船舶轨迹预测与会遇风险预警")

    mmsi_list = sorted(raw_data["mmsi"].unique().astype(str).tolist())
    st.info(f"🚢 共识别到 {len(mmsi_list)} 艘船舶")

    selected_mmsi = st.selectbox("🎯 选择目标船舶(MMSI)", mmsi_list)

    ship_df = raw_data[raw_data["mmsi"] == selected_mmsi].sort_values("time")
    ship_history = ship_df.to_dict("records")
    st.caption(f"📊 船舶 {selected_mmsi} 共有 {len(ship_history)} 个轨迹点")
    # 空数据拦截修复
    if not ship_history:
        st.warning(f"⚠️ 船舶 {selected_mmsi} 无有效轨迹点位，无法展示轨迹与预测功能，请更换其他船舶！")
        return
    with st.expander("🔍 查看船舶轨迹原始数据", expanded=False):
        preview = ship_df[["time", "lat", "lon", "speed"]].head(10)
        preview["time"] = preview["time"].dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(preview, use_container_width=True)

    history_traj, predict_traj = predict_ship_trajectory(ship_history, predict_min=10, sample_interval=30)

    latest = ship_history[-1]
    curr_head = latest.get('heading', 0)
    if pd.isna(curr_head):
        curr_head = 0

    try:
        speed_value = float(latest['speed'])
        speed_display = f"{speed_value:.1f}"
    except:
        speed_display = str(latest['speed'])

    try:
        head_value = float(curr_head)
        head_display = f"{head_value:.1f}"
    except:
        head_display = str(curr_head)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="stat-item"><div class="stat-value">{selected_mmsi}</div><div class="traj-stat-label">船舶MMSI</div></div>',
            unsafe_allow_html=True)
    with col2:
        st.markdown(
            f'<div class="stat-item"><div class="stat-value">{speed_display}</div><div class="traj-stat-label">当前航速(节)</div></div>',
            unsafe_allow_html=True)
    with col3:
        st.markdown(
            f'<div class="stat-item"><div class="stat-value">{head_display}</div><div class="traj-stat-label">当前航向(°)</div></div>',
            unsafe_allow_html=True)
    with col4:
        st.markdown(
            f'<div class="stat-item"><div class="stat-value">{len(history_traj)}</div><div class="traj-stat-label">历史轨迹点数</div></div>',
            unsafe_allow_html=True)

    st.divider()

    with st.expander("📋 轨迹点位明细", expanded=False):
        st.subheader("📍 历史轨迹")
        if history_traj:
            hist_table = pd.DataFrame(history_traj)
            if "time" in hist_table.columns:
                hist_table["time"] = pd.to_datetime(hist_table["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            st.dataframe(hist_table, use_container_width=True)
        else:
            st.info("暂无历史轨迹数据")

        st.subheader("🔮 预测轨迹")
        if predict_traj:
            pred_table = pd.DataFrame(predict_traj)
            if "time" in pred_table.columns:
                pred_table["time"] = pd.to_datetime(pred_table["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            st.dataframe(pred_table, use_container_width=True)
        else:
            st.info("暂无预测轨迹数据")

    st.divider()

    st.markdown("#### ⚠️ 船舶会遇风险检测")
    other_mmsi = [m for m in mmsi_list if m != selected_mmsi]

    if len(other_mmsi) > 0:
        compare_mmsi = st.selectbox("选择对比船舶(检测会遇)", other_mmsi)
        compare_df = raw_data[raw_data["mmsi"] == compare_mmsi].sort_values("time")
        compare_history = compare_df.to_dict("records")
        _, compare_pred = predict_ship_trajectory(compare_history, predict_min=10, sample_interval=30)

        risk_flag, min_distance, risk_time = detect_encounter(predict_traj, compare_pred, safe_distance=200)

        if risk_flag:
            st.markdown(f"""
            <div class="alert-box-red">
                <h4>🔴 紧急告警：检测到船舶会遇风险！</h4>
                <p>目标船: {selected_mmsi} | 对比船: {compare_mmsi}</p>
                <p>最小间距: {min_distance} 米 (安全距离 200 米)</p>
                <p>风险时间: {risk_time.strftime('%Y-%m-%d %H:%M:%S') if risk_time else "未知"}</p>
                <p>💡 建议：立即提醒船舶减速、调整航向，规避碰撞风险</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success(f"✅ 无会遇风险，两船最小间距：{min_distance} 米")
    else:
        st.info("当前仅检测到单艘船舶，无需会遇判断")

    st.divider()

    # ========== 动画控制（美化版） ==========
    st.markdown("### 🎬 动画控制面板")

    # 动画状态会话存储
    anim_key = f"anim_state_{selected_mmsi}"
    if anim_key not in st.session_state:
        st.session_state[anim_key] = "stopped"
    anim_state = st.session_state[anim_key]

    # 美化后的控制按钮
    col_play, col_pause, col_reset = st.columns(3)
    with col_play:
        if st.button("▶️ 播放动画", use_container_width=True, type="primary"):
            st.session_state[anim_key] = "playing"
            st.rerun()
    with col_pause:
        if st.button("⏸️ 暂停动画", use_container_width=True):
            st.session_state[anim_key] = "stopped"
            st.rerun()
    with col_reset:
        if st.button("🔄 重置动画", use_container_width=True):
            st.session_state[anim_key] = "stopped"
            st.rerun()

    # 动画状态提示
    if anim_state == "playing":
        st.info("🎬 动画正在播放中，船舶将沿着橙色虚线移动")
    else:
        st.info("⏸️ 动画已暂停，点击「播放动画」开始观看船舶预测轨迹")

    st.divider()

    # ========== 高德地图轨迹可视化 ==========
    st.markdown("#### 🗺️ 轨迹地图（蓝色=历史轨迹 | 橙色虚线=预测轨迹 | 🚢=预测船舶动画）")

    if not amap_key:
        st.warning("⚠️ 未配置高德地图Key，无法加载地图")
    else:
        if len(history_traj) == 0 and len(predict_traj) == 0:
            st.warning("⚠️ 无轨迹数据可显示")
        else:
            MAX_HISTORY = 60
            sampled_history = history_traj
            if len(history_traj) > MAX_HISTORY:
                step = max(1, len(history_traj) // MAX_HISTORY)
                sampled_history = history_traj[::step]

            sampled_predict = predict_traj

            hist_path = []
            for p in sampled_history:
                if p.get("lat") is not None and p.get("lon") is not None:
                    hist_path.append([float(p['lon']), float(p['lat'])])

            pred_path = []
            for p in sampled_predict:
                if p.get("lat") is not None and p.get("lon") is not None:
                    pred_path.append([float(p['lon']), float(p['lat'])])

            if hist_path:
                center_lng = hist_path[0][0]
                center_lat = hist_path[0][1]
            elif pred_path:
                center_lng = pred_path[0][0]
                center_lat = pred_path[0][1]
            else:
                center_lat, center_lng = 38.93, 117.70

            hist_path_json = json.dumps(hist_path, cls=DateTimeEncoder)
            pred_path_json = json.dumps(pred_path, cls=DateTimeEncoder)

            security_code = "d605610cad7cda1970e462d6103307e0"

            map_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
            <meta charset="utf-8">
            <style>
            body,html{{margin:0;padding:0;height:100%;width:100%;}}
            #map{{height:580px;width:100%;}}
            .legend{{
                position:absolute;bottom:20px;right:20px;
                background:rgba(0,0,0,0.85);
                padding:12px 20px;
                border-radius:12px;
                z-index:999;
                color:white;
                font-size:12px;
                backdrop-filter:blur(8px);
                font-family: monospace;
                border-left: 4px solid #1f77b4;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            }}
            .legend-item{{margin:6px 0;display:flex;align-items:center;gap:10px;}}
            .legend-color{{display:inline-block;width:28px;height:3px;border-radius:2px;}}
            .legend-dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;}}
            .anim-panel{{
                position:absolute;bottom:20px;left:20px;
                background:rgba(0,0,0,0.85);
                padding:10px 18px;
                border-radius:20px;
                z-index:999;
                color:#ffaa33;
                font-size:13px;
                font-weight:bold;
                backdrop-filter:blur(8px);
                font-family: monospace;
                border-left: 3px solid #ff7d00;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            }}
            .title-bar {{
                position:absolute;top:15px;right:15px;
                background:rgba(0,0,0,0.85);
                color:white;
                padding:8px 16px;
                border-radius:20px;
                z-index:999;
                font-size:13px;
                font-weight:500;
                backdrop-filter:blur(8px);
                font-family: monospace;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            }}
            .title-bar span{{
                color: #ff7d00;
                font-weight: bold;
            }}
            @keyframes pulse {{
                0% {{ opacity: 0.6; transform: scale(1); }}
                50% {{ opacity: 1; transform: scale(1.2); }}
                100% {{ opacity: 0.6; transform: scale(1); }}
            }}
            </style>
            <script>
            window._AMapSecurityConfig = {{
                securityJsCode: '{security_code}'
            }};
            </script>
            <script src="https://webapi.amap.com/maps?v=2.0&key={amap_key}"></script>
            </head>
            <body>
            <div id="map"></div>
            <div class="title-bar">
                ⚓ 航速: <span>{speed_display}</span> 节 | 🎬 预测动画
            </div>
            <div class="legend">
                <div class="legend-item"><div class="legend-color" style="background:#1f77b4;"></div> 历史轨迹 ({len(hist_path)}点)</div>
                <div class="legend-item"><div class="legend-color" style="background:#ff7d00;"></div> 预测轨迹 ({len(pred_path)}点)</div>
                <div class="legend-item"><div class="legend-dot" style="background:#ffaa33;"></div> 船舶尾迹</div>
                <div class="legend-item">🚢 预测船舶位置</div>
            </div>
            <div class="anim-panel" id="animStatus">⏳ 动画未播放，请点击上方播放按钮</div>
            <script>
            try {{
                setTimeout(function() {{
                    var map = new AMap.Map('map', {{
                        center: [{center_lng}, {center_lat}],
                        zoom: 14,
                        resizeEnable: true,
                        viewMode: '2D'
                    }});

                    var histPath = {hist_path_json};
                    var predPath = {pred_path_json};
                    var animStatus = document.getElementById('animStatus');
                    var animRunning = {'true' if anim_state == 'playing' else 'false'};

                    // 历史轨迹
                    if(histPath.length >= 2) {{
                        var histLine = new AMap.Polyline({{
                            path: histPath,
                            strokeColor: '#1f77b4',
                            strokeWeight: 4,
                            strokeOpacity: 0.9,
                            lineJoin: 'round',
                            lineCap: 'round'
                        }});
                        map.add(histLine);
                    }}

                    // 预测轨迹
                    if(predPath.length >= 2) {{
                        var predLine = new AMap.Polyline({{
                            path: predPath,
                            strokeColor: '#ff7d00',
                            strokeWeight: 4,
                            strokeOpacity: 0.95,
                            strokeDasharray: [12, 10],
                            lineJoin: 'round',
                            lineCap: 'round'
                        }});
                        map.add(predLine);

                        // 预测点标记
                        for(var i = 0; i < predPath.length; i++) {{
                            var waypoint = new AMap.CircleMarker({{
                                center: predPath[i],
                                radius: 4,
                                fillColor: '#ffaa33',
                                fillOpacity: 0.6,
                                strokeColor: '#ff7d00',
                                strokeWeight: 1
                            }});
                            map.add(waypoint);
                        }}
                    }}

                    // 起点标记
                    if(histPath.length > 0) {{
                        var startMarker = new AMap.Marker({{
                            position: histPath[0],
                            label: {{content: '📍 起点', offset: new AMap.Pixel(0, 22)}}
                        }});
                        map.add(startMarker);
                    }}

                    // 当前位置标记
                    if(histPath.length > 0) {{
                        var currentPosMarker = new AMap.Marker({{
                            position: histPath[histPath.length-1],
                            label: {{content: '📌 当前位置', offset: new AMap.Pixel(0, 22)}}
                        }});
                        map.add(currentPosMarker);
                    }}

                    // 船舶动画
                    var shipPoints = predPath;
                    var shipMarker, glowMarker;
                    var trailMarkers = [];
                    var currentIndex = 0;
                    var animTimer = null;
                    var MAX_TRAIL = 15;
                    var ANIM_INTERVAL = 180;

                    function initShipMarkers() {{
                        if(!shipPoints || shipPoints.length == 0) return;
                        shipMarker = new AMap.Marker({{
                            position: shipPoints[0],
                            label: {{
                                content: '🚢',
                                offset: new AMap.Pixel(-22, -22),
                                direction: 'top'
                            }},
                            zIndex: 100,
                            extData: {{type: 'ship'}}
                        }});
                        map.add(shipMarker);

                        glowMarker = new AMap.CircleMarker({{
                            center: shipPoints[0],
                            radius: 28,
                            fillColor: '#ff7d00',
                            fillOpacity: 0.25,
                            strokeColor: '#ffaa33',
                            strokeWeight: 1.5,
                            strokeOpacity: 0.8
                        }});
                        map.add(glowMarker);
                    }}

                    function clearAllTrails() {{
                        for(var t of trailMarkers) {{
                            map.remove(t);
                        }}
                        trailMarkers = [];
                    }}

                    function animateShip() {{
                        if(currentIndex >= shipPoints.length) {{
                            animStatus.innerHTML = '✅ 动画完成 | 船舶已到达预测终点';
                            document.querySelector('.title-bar').innerHTML = '⚓ 航速: <span>' + '{speed_display}' + '</span> 节 | ✅ 动画完成';
                            if(glowMarker) map.remove(glowMarker);
                            if(shipPoints.length > 0) {{
                                var endMarker = new AMap.Marker({{
                                    position: shipPoints[shipPoints.length-1],
                                    label: {{content: '🏁 预测终点', offset: new AMap.Pixel(0, 25)}}
                                }});
                                map.add(endMarker);
                            }}
                            return;
                        }}

                        shipMarker.setPosition(shipPoints[currentIndex]);
                        if(glowMarker) glowMarker.setCenter(shipPoints[currentIndex]);

                        var trail = new AMap.CircleMarker({{
                            center: shipPoints[currentIndex],
                            radius: 6,
                            fillColor: '#ffaa33',
                            fillOpacity: 0.9,
                            strokeColor: '#ff7d00',
                            strokeWeight: 1.5
                        }});
                        map.add(trail);
                        trailMarkers.push(trail);

                        while(trailMarkers.length > MAX_TRAIL) {{
                            var old = trailMarkers.shift();
                            map.remove(old);
                        }}

                        var progress = Math.round((currentIndex + 1) / shipPoints.length * 100);
                        animStatus.innerHTML = '🚢 船舶移动中 | 进度: ' + progress + '% (' + (currentIndex + 1) + '/' + shipPoints.length + ')';
                        currentIndex++;
                        animTimer = setTimeout(animateShip, ANIM_INTERVAL);
                    }}

                    initShipMarkers();

                    if(animRunning && shipPoints.length > 0) {{
                        animStatus.innerHTML = '🚀 动画播放中...';
                        // 调整视野到预测轨迹区域
                        if(shipPoints.length > 1) {{
                            var bounds = new AMap.Bounds(shipPoints[0], shipPoints[shipPoints.length-1]);
                            map.setBounds(bounds, false, [80, 80, 80, 80]);
                        }}
                        setTimeout(animateShip, 500);
                    }} else if(shipPoints.length > 0) {{
                        // 不播放动画时，仍然调整视野
                        var bounds = new AMap.Bounds(shipPoints[0], shipPoints[shipPoints.length-1]);
                        map.setBounds(bounds, false, [80, 80, 80, 80]);
                        animStatus.innerHTML = '⏸️ 动画已暂停，点击上方播放按钮';
                    }}

                }}, 200);
            }} catch(err) {{
                console.error("地图脚本异常:", err);
                if(document.getElementById("animStatus")) {{
                    document.getElementById("animStatus").innerHTML = "⚠️ 地图渲染异常，请刷新页面重试";
                }}
            }}
            </script>
            </body>
            </html>
            """
            try:
                st.components.v1.html(map_html, height=620)
            except Exception as e:
                st.error(f"地图加载失败: {str(e)}")

            st.info("""
            💡 **动画控制说明**：
            - 🎬 点击「播放动画」船舶将沿橙色预测虚线移动
            - ⏸️ 播放过程中可随时暂停
            - 🔄 重置动画会清空尾迹并回到起点
            - ✨ 黄色圆点为船舶经过的尾迹（保留最近15个）
            """)

    st.markdown('</div>', unsafe_allow_html=True)
    st.divider()

    try:
        show_alert_log_panel(panel_unique_id="trajectory_tab")
    except Exception:
        pass