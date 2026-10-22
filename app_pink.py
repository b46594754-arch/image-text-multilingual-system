
import io
import hashlib
from datetime import datetime
from collections import Counter

import cv2
import numpy as np
import streamlit as st
from PIL import Image

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None


# ==================== 页面设置 ====================
st.set_page_config(
    page_title="智能图像文字识别与多语言处理系统",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== 界面样式（淡粉色主题） ====================
st.markdown(
    """
    <style>
    :root {
        --main-bg: #ffffff;
        --panel-bg: #ffffff;
        --border-color: #f3c6da;
        --text-main: #5a2a40;
        --text-muted: #9c6c81;
        --accent: #e75480;
        --accent-soft: #ffe3ee;
        --ok: #1a7a4f;
        --ok-bg: #e9f8f0;
        --warn: #b45309;
        --warn-bg: #fff7ed;
    }
    /* 主色调改为粉色（影响 Streamlit 主按钮等） */
    :root { --primary-color: #e75480 !important; }

    .stApp { background: var(--main-bg); color: var(--text-main); }
    .block-container { padding-top: 1.4rem; padding-bottom: 2.2rem; max-width: 1480px; }
    header[data-testid="stHeader"] { background: transparent; }

    h1, h2, h3 { color: var(--text-main); letter-spacing: -0.25px; }
    h1 { font-size: 1.85rem !important; font-weight: 800 !important; margin-bottom: .25rem !important; }
    h2 { font-size: 1.2rem !important; font-weight: 700 !important; }
    h3 { font-size: 1rem !important; font-weight: 650 !important; }

    /* 侧边栏：淡粉渐变 + 白字（增强方式等标题均为白色） */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ef93b6 0%, #e87ba6 100%);
        border-right: 1px solid #e07a9f;
        color: #ffffff;
    }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    [data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
    [data-testid="stSidebar"] .stMarkdown hr { border-color: #f3b9d2; }

    /* 关键修复：侧边栏的选择框必须为“白底黑字”，否则白底上看不清 */
    [data-testid="stSidebar"] .stSelectbox,
    [data-testid="stSidebar"] .stSelectbox * { color: #43223a !important; }
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
        background-color: #ffffff !important;
        border: 1px solid #f3cdda !important;
        border-radius: 10px !important;
    }

    /* 主区选择框：醒目的“白底黑字 + 粉色边框” */
    .stSelectbox [data-baseweb="select"] {
        background-color: #ffffff !important;
        border: 1.5px solid #f0a9c4 !important;
        border-radius: 10px !important;
    }
    .stSelectbox [data-baseweb="select"] * { color: #43223a !important; }
    .stSelectbox:hover [data-baseweb="select"] { border-color: #e75480 !important; }

    /* 卡片 */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--panel-bg);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        box-shadow: 0 1px 2px rgba(122,31,72,.05), 0 6px 18px rgba(122,31,72,.06);
        padding: 1.1rem 1.2rem;
    }

    /* 指标卡 */
    div[data-testid="stMetric"] {
        background: var(--panel-bg);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(122,31,72,.05);
    }
    div[data-testid="stMetricLabel"] { color: var(--text-muted); font-size: .78rem; }
    div[data-testid="stMetricValue"] { color: var(--text-main); font-size: 1.4rem; }

    .subtle { color: var(--text-muted); font-size: .9rem; }

    .hero {
        background: linear-gradient(120deg, #f2a6c6 0%, #ec8fb2 100%);
        padding: 1.5rem 1.8rem;
        border-radius: 18px;
        color: #ffffff;
        margin-bottom: 1.1rem;
        box-shadow: 0 10px 30px rgba(231,84,128,.18);
    }
    .hero-title { color: #ffffff; font-size: 1.7rem; font-weight: 800; line-height: 1.35; }
    .hero-subtitle { color: #fff5fa; font-size: .92rem; margin-top: .4rem; }

    .badge { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .76rem; font-weight: 600; margin-right: 6px; }
    .badge-ok { color: var(--ok); background: var(--ok-bg); border: 1px solid #bbf7d0; }
    .badge-warn { color: var(--warn); background: var(--warn-bg); border: 1px solid #fed7aa; }
    .badge-accent { color: var(--accent); background: var(--accent-soft); border: 1px solid #f7b8d1; }

    .status-line { border-radius: 10px; padding: 9px 12px; font-size: .85rem; }
    .status-ok { color: var(--ok); background: var(--ok-bg); border: 1px solid #bbf7d0; }
    .status-dirty { color: var(--warn); background: var(--warn-bg); border: 1px solid #fed7aa; }

    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 2.6rem;
        transition: transform .05s ease;
    }
    .stButton > button:active, .stDownloadButton > button:active { transform: scale(.98); }

    .stTextArea textarea { font-size: .95rem; line-height: 1.7; border-radius: 10px; }
    div[data-testid="stImage"] img { border-radius: 12px; border: 1px solid var(--border-color); }
    div[data-testid="stFileUploader"] {
        background: #fff;
        border: 1.5px dashed #f0a9c4;
        border-radius: 14px;
        padding: .6rem;
    }
    hr { border-color: var(--border-color); margin-top: .9rem; margin-bottom: .9rem; }
    .footer { color: var(--text-muted); font-size: .82rem; text-align: center; margin-top: 1.4rem; }
    @media (max-width: 900px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .hero-title { font-size: 1.3rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================== 初始化状态 ====================
for _k, _v in {
    "recognized_text": "",
    "saved_text": "",
    "edit_buffer": "",
    "last_image_hash": "",
    "ocr_finished": False,
    "save_version": 0,
    "agent_stats": {},
    "agent_report": "",
    "agent_execution_log": [],
    "agent_executed_task": "",
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def clear_text_state():
    st.session_state["recognized_text"] = ""
    st.session_state["saved_text"] = ""
    st.session_state["edit_buffer"] = ""
    st.session_state["ocr_finished"] = False


# ==================== OCR 模型 ====================
@st.cache_resource
def load_ocr():
    if PaddleOCR is None:
        st.error(
            "未检测到 PaddleOCR 依赖，OCR 功能不可用。\n"
            "请在兼容的 Python 环境中执行：pip install paddleocr"
        )
        st.stop()
    return PaddleOCR(
        lang="ch",
        device="cpu",
        use_textline_orientation=True,
        enable_mkldnn=False,
    )


# ==================== 图像处理函数 ====================
def enhance_image(image_rgb, mode, denoise_strength):
    """图像增强：降噪、对比度增强、锐化。"""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if denoise_strength > 0:
        bgr = cv2.fastNlMeansDenoisingColored(
            bgr, None, denoise_strength, denoise_strength, 7, 21
        )

    if mode == "文档增强":
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        bgr = cv2.cvtColor(
            cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR
        )
        blurred = cv2.GaussianBlur(bgr, (0, 0), 1.0)
        bgr = cv2.addWeighted(bgr, 1.35, blurred, -0.35, 0)

    elif mode == "黑白扫描":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        result = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
        )
        return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    elif mode == "灰度增强":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def order_points(points):
    """按照左上、右上、右下、左下排列四个角点。"""
    points = np.asarray(points, dtype=np.float32)
    result = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    result[0] = points[np.argmin(sums)]
    result[2] = points[np.argmax(sums)]
    result[1] = points[np.argmin(differences)]
    result[3] = points[np.argmax(differences)]
    return result


def correct_perspective(image_rgb):
    """尝试检测文档边缘并进行透视矫正；失败时返回原图。"""
    height, width = image_rgb.shape[:2]
    scale = min(1.0, 900 / max(height, width))
    if scale < 1:
        small = cv2.resize(
            image_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    else:
        small = image_rgb.copy()

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    small_height, small_width = small.shape[:2]
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) != 4:
            continue
        if cv2.contourArea(polygon) < small_height * small_width * 0.15:
            continue

        points = order_points(polygon.reshape(4, 2)) / scale
        tl, tr, br, bl = points
        out_width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        out_height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        if out_width < 100 or out_height < 100:
            continue

        destination = np.array(
            [[0, 0], [out_width - 1, 0], [out_width - 1, out_height - 1], [0, out_height - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(points, destination)
        corrected = cv2.warpPerspective(image_rgb, matrix, (out_width, out_height))
        return corrected, True

    return image_rgb, False


def image_to_bytes(image_rgb, image_format="PNG"):
    """将图像转换为可下载的文件数据。"""
    buffer = io.BytesIO()
    image = Image.fromarray(image_rgb)
    if image_format.upper() == "PDF":
        image.save(buffer, format="PDF", resolution=150)
    else:
        image.save(buffer, format="PNG")
    return buffer.getvalue()


def extract_text(ocr_results):
    """兼容 PaddleOCR 返回的字典结构。"""
    texts = []
    for item in ocr_results or []:
        if isinstance(item, dict):
            rec_texts = item.get("rec_texts", [])
            if rec_texts:
                texts.extend(
                    str(text).strip() for text in rec_texts if str(text).strip()
                )
    return "\n".join(texts)


def count_text(text):
    """统计文本信息。"""
    import re
    non_whitespace = sum(1 for char in text if not char.isspace())
    chinese = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    lines = len([line for line in text.splitlines() if line.strip()])
    words = len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))
    return {
        "字符数": len(text),
        "有效字符": non_whitespace,
        "中文字符": chinese,
        "文本行数": lines,
        "单词数": words,
    }


# ==================== 侧边处理设置 ====================
with st.sidebar:
    st.markdown("## 图像处理设置")
    st.caption("IMAGE PROCESSING")
    st.divider()

    st.markdown("#### 01 · 图像矫正")
    try_perspective = st.checkbox(
        "尝试文档透视矫正",
        value=False,
        help="适用于存在明显倾斜、且文档边缘较清晰的图片。",
    )

    st.markdown("#### 02 · 图像增强")
    enhancement_mode = st.selectbox(
        "增强方式",
        ["文档增强", "黑白扫描", "灰度增强", "不增强"],
        index=0,
        help="下拉框为白底黑字，请在此选择图像处理模式。",
    )
    denoise_strength = st.slider(
        "降噪强度",
        min_value=0,
        max_value=15,
        value=3,
        help="数值越高，降噪越明显；过高可能损失细小文字。",
    )
    st.divider()
    st.success("本地 PaddleOCR · CPU 推理 · 无需付费 API")
    with st.expander("使用说明"):
        st.markdown(
            "1. 上传图片后调整左侧参数\n"
            "2. 选择智能体任务并执行\n"
            "3. 校对识别结果并导出 TXT / PNG / PDF"
        )

# ==================== 页面标题 ====================
st.markdown(
    """
    <div class="hero">
      <div class="hero-title">智能图像文字识别与多语言处理系统</div>
      <div class="hero-subtitle">图像增强 · 文字识别 · 文本校对 · 结果导出</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================== 上传区 ====================
with st.container(border=True):
    st.markdown("### 01　上传图像")
    st.caption("上传需要识别的图片，系统将在下方显示原图、处理结果和识别文本。")
    uploaded_file = st.file_uploader(
        "点击选择图片，或将图片拖到此处",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        help="支持 PNG、JPG、BMP、WEBP 格式。",
    )

# ==================== 智能体任务选择 ====================
run_ocr = False
agent_task = "增强识别与文本分析"

if uploaded_file is not None:
    st.markdown("### 智能体工作流")
    agent_task = st.selectbox(
        "选择执行任务",
        ["快速文字识别", "增强识别与文本分析", "识别并生成分析报告"],
        help="下拉框为白底黑字，请选择任务后执行。",
        key="agent_task",
    )

    workflow_plans = {
        "快速文字识别": [
            "读取上传的图片",
            "使用当前图像处理设置",
            "调用本地 PaddleOCR 识别文字",
            "展示识别结果",
        ],
        "增强识别与文本分析": [
            "读取上传的图片",
            "执行图像预处理与增强",
            "调用本地 PaddleOCR 识别文字",
            "统计字符数、中文字符数和文本行数",
        ],
        "识别并生成分析报告": [
            "读取上传的图片",
            "执行图像预处理与增强",
            "调用本地 PaddleOCR 识别文字",
            "分析文本统计信息",
            "整理识别结果与任务记录",
        ],
    }

    with st.container(border=True):
        s1, s2 = st.columns([1, 2])
        with s1:
            st.markdown("#### 执行计划")
        with s2:
            st.markdown(
                "".join(
                    f'<span class="badge badge-accent">{i}. {step}</span>'
                    for i, step in enumerate(workflow_plans[agent_task], start=1)
                ),
                unsafe_allow_html=True,
            )

    run_ocr = st.button(
        "▶ 执行智能体工作流",
        type="primary",
        use_container_width=True,
        help="按照所选任务执行本地图片处理与文字识别。",
    )

if uploaded_file is None:
    st.info("请先上传图片。上传后即可调整左侧处理参数、查看图像对比并开始文字识别。")
    with st.container(border=True):
        st.markdown("### 系统功能")
        feature_cols = st.columns(3)
        with feature_cols[0]:
            st.markdown("**图像预处理**")
            st.caption("文档增强、灰度增强、黑白扫描、降噪和透视矫正。")
        with feature_cols[1]:
            st.markdown("**本地文字识别**")
            st.caption("调用本地 PaddleOCR 模型，无需付费 API。")
        with feature_cols[2]:
            st.markdown("**文本编辑与导出**")
            st.caption("校对识别结果，保存文本并导出 TXT、PNG 或 PDF。")
    st.stop()

# ==================== 读取图像与缓存处理 ====================
file_bytes = uploaded_file.getvalue()
image_hash = hashlib.md5(file_bytes).hexdigest()

try:
    original_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    original_array = np.array(original_image)

    if st.session_state["last_image_hash"] != image_hash:
        st.session_state["last_image_hash"] = image_hash
        st.session_state["recognized_text"] = ""
        st.session_state["saved_text"] = ""
        st.session_state["edit_buffer"] = ""
        st.session_state["ocr_finished"] = False

    if try_perspective:
        working_array, perspective_applied = correct_perspective(original_array)
    else:
        working_array = original_array.copy()
        perspective_applied = False

    if enhancement_mode == "不增强":
        enhanced_array = working_array.copy()
    else:
        enhanced_array = enhance_image(working_array, enhancement_mode, denoise_strength)

except Exception as error:
    st.error(f"图像读取或预处理失败：{error}")
    st.stop()

# ==================== 执行智能体工作流 ====================
if run_ocr:
    execution_log = []
    started_at = datetime.now()
    start_text = started_at.strftime("%Y-%m-%d %H:%M:%S")

    try:
        with st.spinner("智能体正在执行任务，请稍候……"):
            image_height, image_width = original_array.shape[:2]
            execution_log.append(f"[完成] 任务已启动：{agent_task}")
            execution_log.append(f"[完成] 图片已读取：{uploaded_file.name}")

            execution_log.append(f"[完成] 图像预处理：{enhancement_mode}")
            if try_perspective and perspective_applied:
                execution_log.append("[完成] 已尝试透视矫正")
            elif try_perspective:
                execution_log.append("[提示] 未检测到可靠的文档边缘，继续使用当前图像")

            ocr = load_ocr()
            ocr_results = ocr.predict(input=enhanced_array)
            recognized = extract_text(ocr_results)
            execution_log.append("[完成] PaddleOCR 识别结束")

            st.session_state["recognized_text"] = recognized
            st.session_state["saved_text"] = recognized
            st.session_state["edit_buffer"] = recognized
            st.session_state["ocr_finished"] = True
            st.session_state["save_version"] = 0

            stats = count_text(recognized)
            st.session_state["agent_stats"] = stats
            st.session_state["agent_executed_task"] = agent_task
            execution_log.append("[完成] 文本统计已执行")
            st.session_state["agent_execution_log"] = execution_log

            if agent_task == "识别并生成分析报告":
                finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                report_lines = [
                    "智能图像文字识别与多语言处理系统",
                    "智能体任务执行报告",
                    "=" * 40,
                    f"任务名称：{agent_task}",
                    f"开始时间：{start_text}",
                    f"完成时间：{finished_at}",
                    f"图片文件名：{uploaded_file.name}",
                    f"图片尺寸：{image_width} × {image_height}",
                    f"图像增强方式：{enhancement_mode}",
                    f"降噪强度：{denoise_strength}",
                    f"是否尝试透视矫正：{'是' if try_perspective else '否'}",
                    f"是否成功矫正：{'是' if perspective_applied else '否'}",
                    "",
                    "一、文本统计",
                ]
                for label, value in stats.items():
                    report_lines.append(f"{label}：{value}")
                report_lines.extend(["", "二、执行记录"])
                report_lines.extend(st.session_state["agent_execution_log"])
                report_lines.extend(
                    ["", "三、OCR 识别结果", "-" * 40,
                     recognized if recognized else "未识别到文字", "",
                     "四、说明",
                     "本报告由本地规则驱动的智能体工作流生成。",
                     "OCR 结果可能存在误差，请人工校对。"]
                )
                st.session_state["agent_report"] = "\n".join(report_lines)
                st.session_state["agent_execution_log"].append("[完成] 综合 TXT 报告已生成")

        if recognized:
            st.success("智能体任务执行完成，可以查看结果。")
        else:
            st.warning("没有识别到文字，请尝试调整图片处理参数。")

    except Exception as error:
        st.error("智能体任务执行失败，请查看具体错误。")
        st.exception(error)

# ==================== 顶部统计条 ====================
image_height, image_width = original_array.shape[:2]
m1, m2, m3, m4 = st.columns(4)
m1.metric("图片宽度", f"{image_width} px")
m2.metric("图片高度", f"{image_height} px")
m3.metric("识别状态", "已完成" if st.session_state["ocr_finished"] else "待识别")
m4.metric("当前文本字符", f"{len(st.session_state['edit_buffer'])}")

st.write("")

# ==================== 结果标签页 ====================
tab_compare, tab_text, tab_report = st.tabs(
    ["图像对比", "识别文本与校对", "分析报告"]
)

# ---- 标签一：图像对比 ----
with tab_compare:
    with st.container(border=True):
        st.markdown("### 02　图像处理与对比")
        st.caption("左右并排查看原始图像与处理后的图像；可分别下载对应文件。")
        original_col, enhanced_col = st.columns(2, gap="medium")
        with original_col:
            st.markdown("**原始图像**")
            st.image(original_array, use_container_width=True)
            st.download_button(
                "下载原始图片", data=file_bytes, file_name=uploaded_file.name,
                mime=uploaded_file.type or "application/octet-stream",
                use_container_width=True, key="download_original_image",
            )
        with enhanced_col:
            st.markdown("**处理后图像**")
            st.image(enhanced_array, use_container_width=True)
            d_png, d_pdf = st.columns(2, gap="small")
            with d_png:
                st.download_button(
                    "下载处理图 PNG", data=image_to_bytes(enhanced_array, "PNG"),
                    file_name="处理后图片.png", mime="image/png",
                    use_container_width=True, key="download_processed_png",
                )
            with d_pdf:
                st.download_button(
                    "下载扫描 PDF", data=image_to_bytes(enhanced_array, "PDF"),
                    file_name="处理后图片.pdf", mime="application/pdf",
                    use_container_width=True, key="download_processed_pdf",
                )

        badges = []
        badges.append(
            f'<span class="badge badge-ok">增强：{enhancement_mode}</span>'
        )
        if try_perspective:
            badges.append(
                f'<span class="badge {"badge-ok" if perspective_applied else "badge-warn"}">'
                f'透视矫正：{"已应用" if perspective_applied else "未检测到边缘"}</span>'
            )
        st.markdown("".join(badges), unsafe_allow_html=True)

# ---- 标签二：识别文本与校对 ----
with tab_text:
    with st.container(border=True):
        text_header_col, text_status_col = st.columns([2, 1])
        with text_header_col:
            st.markdown("### 识别文本与校对")
            st.caption("直接编辑文本；保存后可下载已保存版本。")
        with text_status_col:
            if st.session_state["saved_text"] == st.session_state["edit_buffer"]:
                st.markdown('<div class="status-line status-ok">当前编辑内容与保存版本一致</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-line status-dirty">存在未保存修改</div>',
                            unsafe_allow_html=True)

        if st.session_state["ocr_finished"]:
            st.markdown("**快速复制（只读预览）**")
            st.code(st.session_state["edit_buffer"] or "（暂无识别结果）", language="text")

        edited_text = st.text_area(
            "文本编辑区",
            key="edit_buffer",
            height=190,
            placeholder="识别结果将显示在这里。你也可以直接输入或修改文字。",
            label_visibility="collapsed",
        )

        stats = count_text(edited_text)
        stat_cols = st.columns(5)
        for col, (label, value) in zip(stat_cols, stats.items()):
            col.metric(label, value)

        st.write("")
        action_save, action_download, action_clear = st.columns([1, 1, 0.8])
        with action_save:
            if st.button("保存修改", type="primary", use_container_width=True):
                st.session_state["saved_text"] = st.session_state["edit_buffer"]
                st.session_state["save_version"] += 1
                st.toast("文本修改已保存。")
        with action_download:
            st.download_button(
                "下载已保存 TXT", data=st.session_state["saved_text"],
                file_name="识别结果_已保存.txt", mime="text/plain; charset=utf-8",
                use_container_width=True,
            )
        with action_clear:
            st.button("清空", use_container_width=True, on_click=clear_text_state)

# ---- 标签三：分析报告 ----
with tab_report:
    if not st.session_state["ocr_finished"]:
        st.info("执行智能体工作流后，这里将展示文本分析与执行记录。")
    elif st.session_state["agent_executed_task"] != agent_task:
        st.warning("当前显示的是上一次任务的报告，请重新执行以匹配所选任务。")
    else:
        if agent_task in ["增强识别与文本分析", "识别并生成分析报告"]:
            with st.container(border=True):
                st.markdown("### 智能体文本分析")
                saved_stats = st.session_state["agent_stats"]
                stat_cols = st.columns(5)
                for col, (label, value) in zip(stat_cols, saved_stats.items()):
                    col.metric(label, value)

        if agent_task == "识别并生成分析报告":
            with st.container(border=True):
                st.markdown("### 智能体综合报告")
                st.caption("报告包含任务信息、图像处理设置、文本统计、执行记录和 OCR 识别结果。")
                st.text_area(
                    "报告预览", value=st.session_state["agent_report"], height=260,
                    disabled=True, key="agent_report_preview",
                )
                st.download_button(
                    "下载综合分析报告 TXT", data=st.session_state["agent_report"],
                    file_name="智能体分析报告.txt", mime="text/plain; charset=utf-8",
                    type="primary", use_container_width=True, key="download_agent_report",
                )
                st.markdown("#### 执行记录")
                for record in st.session_state["agent_execution_log"]:
                    st.write(record)

# ==================== 页脚 ====================
st.markdown(
    '<div class="footer">智能图像文字识别与多语言处理系统 · 本地 OCR 与图像处理 · '
    "识别结果可能存在误差，请在导出前校对。</div>",
    unsafe_allow_html=True,
)
