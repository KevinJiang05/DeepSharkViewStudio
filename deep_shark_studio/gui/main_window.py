"""PySide6 user interface for DeepShark View Studio."""

from __future__ import annotations

import sys
import time
import traceback
import shutil
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QDesktopServices,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from deep_shark_studio.gui.panels import StitchRuntimeModePanel
from deep_shark_studio.calibration import (
    calibrate_camera_from_images,
    draw_chessboard_detection,
    spec_from_config,
    undistort_image,
    write_calibration_report,
)
from deep_shark_studio.calibration_candidate import (
    generate_calibration_candidate,
    latest_calibration_candidate,
    load_calibration_candidate,
)
from deep_shark_studio.calibration_snapshot import (
    save_calibration_snapshot,
    topology_diagnostics,
)
from deep_shark_studio.calibration_session import (
    BoardDefinitionNotConfirmedError,
    CalibrationSession,
    MINIMUM_INTRINSIC_SAMPLES,
    MINIMUM_PAIR_SAMPLES,
    supported_aruco_dictionaries,
)
from deep_shark_studio.app_logging import (
    create_application_logger,
    redact_log_message,
)
from deep_shark_studio.config import (
    CONFIG_DIR,
    PROJECT_ROOT,
    ConfigConflictError,
    config_revision,
    load_config,
    save_config,
)
from deep_shark_studio.geometry_diagnostics import (
    ALPHA_50,
    NO_FEATHER,
    GeometryDiagnosticResult,
    run_geometry_diagnostics as compute_geometry_diagnostics,
)
from deep_shark_studio.project import backup_configs, export_runtime_config, load_project, save_project
from deep_shark_studio.project_package import (
    activate_project_package,
    export_project_package,
    validate_project_package,
)
from deep_shark_studio.qgc.video_output import FfmpegVideoSink, VideoOutputConfig
from deep_shark_studio.pairwise_diagnostics import (
    PairDefinition,
    PairwiseCandidateResult,
    estimate_pairwise_candidate,
    latest_snapshot_directory,
    load_snapshot_frames,
    pair_definitions,
    run_automatic_pairwise_candidates,
    save_pairwise_results,
)
from deep_shark_studio.stream_manager import (
    CameraStreamConfig,
    CameraStreamManager,
    STREAM_CONNECTING,
    STREAM_FAILED,
    STREAM_IDLE,
    STREAM_LIVE,
    STREAM_STOPPED,
    STREAM_STOPPING,
)
from deep_shark_studio.stitch_processing import StitchProcessingManager
from deep_shark_studio.stitch_runtime_controller import RuntimeStitchController
from deep_shark_studio.stitch_runtime_modes import (
    ProjectionSource,
    RuntimeStitchConfig,
    StitchRuntimeMode,
)
from deep_shark_studio.stitcher import SurroundStitcher, load_images_from_directory, save_image
from deep_shark_studio.projection import (
    B2CandidateProjectionProvider,
    B2_FAR_FIELD_PROJECTION_SOURCE,
    CurrentPerspectiveProjectionProvider,
    FisheyeIntrinsicsRuntimeError,
    FisheyeIntrinsicsRuntimeSource,
    FisheyeRectilinearParams,
    FisheyeRectilinearProjectionProvider,
    ProjectionResult,
    load_fisheye_intrinsics_source,
)
from deep_shark_studio.seam import (
    LAYOUT_TUNER_PRESETS_V2,
    CameraAdjustParams,
    FarFieldLayoutCandidateError,
    FarFieldLayoutRuntimeCandidate,
    LayoutCandidateRuntimeError,
    LayoutPreviewParamsV2,
    LayoutRuntimeCandidate,
    PairLayoutParams,
    default_far_field_layout_candidate_root,
    LayoutTunerParams,
    default_layout_candidate_root,
    layout_tuner_pair_candidates,
    load_far_field_layout_candidate,
    load_layout_candidate_for_runtime,
    normalize_layout_tuner_params,
    render_far_field_custom_from_projection,
    render_front_priority_layout_preview,
    save_far_field_layout_candidate,
    save_layout_tuner_candidate,
)
from deep_shark_studio.topology import (
    active_stitch_profile,
    active_topology_camera_keys,
    source_coordinate_diagnostics,
    validate_overlap_seams,
)


CAMERA_KEYS = ["front_left", "front_right", "front", "behind", "left", "right"]
SOURCE_TYPES = ["image_dir", "usb", "video_file", "rtsp"]

ZH_CN = {
    "Language": "语言",
    "English": "英文",
    "Chinese": "中文",
    "Ready": "就绪",
    "Realtime Preview": "实时预览",
    "Realtime Monitor": "实时监看",
    "Layout Tuning": "布局调参",
    "Calibration & Candidates": "标定与候选",
    "Projection Research": "投影研发",
    "Diagnostics & Logs": "诊断与日志",
    "Camera & Runtime Config": "相机与运行配置",
    "Project Files": "项目文件",
    "Camera Config": "相机配置",
    "Calibration": "标定与调参",
    "Project": "项目管理",
    "Logs": "日志",
    "Copy Log": "复制日志",
    "Clear View": "清空显示",
    "Log file": "日志文件",
    "Seam validation failed": "拼接缝校验失败",
    "Choose Image Directory": "选择图片目录",
    "Load Images": "加载图片",
    "Start Live": "开始实时预览",
    "Stop": "停止",
    "Save Results": "保存结果",
    "Multi-camera View": "多路视图",
    "Stitched View": "拼接视图",
    "Candidate Stitch": "候选拼接",
    "Template Stitch": "模板拼接",
    "B-2 Candidate View [Advanced]": "B-2 候选视图 [高级]",
    "Current Profile Template": "当前正式 Profile 模板",
    "Stitch Runtime Mode": "拼接算法模式",
    "Runtime stitching algorithm, independent from Grid / Focus / Stitched layout.": "拼接算法模式独立于 Grid / Focus / Stitched 显示布局。",
    "Mode": "模式",
    "Far-field / distance priority": "远景优先 / Far-field",
    "Near-field / front priority": "近景优先 / Near-field",
    "Auto / experimental disabled": "自动 / 实验占位",
    "Projection Source": "投影来源",
    "Current Perspective Runtime": "当前 Perspective Runtime",
    "Projection Source: Current Perspective Runtime": "投影来源：当前 Perspective Runtime",
    "Near-field Projection Source": "近景投影来源",
    "Current Perspective": "当前 Perspective",
    "Fisheye Rectilinear [Experimental]": "鱼眼 Rectilinear [实验]",
    "Far-field uses current runtime projection.": "Far-field 使用当前正式 runtime 投影。",
    "Far-field Custom uses B-2 candidate projection.": "Far-field Custom 使用 B-2 候选投影。",
    "Load Fisheye Intrinsics Source...": "加载鱼眼内参来源...",
    "Clear Fisheye Intrinsics Source": "清除鱼眼内参来源",
    "No fisheye intrinsics source loaded.": "未加载鱼眼内参来源。",
    "Loaded fisheye intrinsics source": "已加载鱼眼内参来源",
    "Choose Fisheye Intrinsics Source": "选择鱼眼内参来源",
    "Fisheye source (*.yaml);;All Files (*)": "鱼眼来源 (*.yaml);;所有文件 (*)",
    "Fisheye intrinsics source load failed": "鱼眼内参来源加载失败",
    "Fisheye Rectilinear is experimental and only affects Near-field preview/runtime.": "鱼眼 Rectilinear 是实验功能，只影响 Near-field 预览/runtime。",
    "Fisheye Rectilinear projection requires a loaded fisheye intrinsics source.": "鱼眼 Rectilinear 投影需要先加载鱼眼内参来源。",
    "Fisheye Balance": "鱼眼 Balance",
    "Fisheye FOV Scale": "鱼眼 FOV Scale",
    "Fisheye Rectilinear Candidate - Research Only": "鱼眼 Rectilinear 候选 - 仅研发",
    "Equirectangular Candidate - Research Only": "Equirectangular 候选 - 仅研发",
    "Layout Candidate": "Layout 候选",
    "Near-field Layout Candidate": "近景 Layout 候选",
    "Load Candidate...": "加载候选...",
    "Load Near-field Layout Candidate": "加载近景 Layout 候选",
    "Use Far-field Custom Layout": "使用远景自定义布局",
    "Load Far-field Layout Candidate": "加载远景 Layout 候选",
    "Clear Far-field Layout Candidate": "清除远景 Layout 候选",
    "No Far-field Layout Candidate loaded.": "未加载远景 Layout 候选。",
    "Loaded Far-field Layout Candidate": "已加载远景 Layout 候选",
    "Choose Far-field Layout Candidate": "选择远景 Layout 候选",
    "Far-field layout candidate load failed": "远景 Layout 候选加载失败",
    "Far-field custom layout requires a loaded Far-field Layout Candidate.": "远景自定义布局需要先加载远景 Layout 候选。",
    "Far-field Layout": "远景 Layout",
    "Near-field Layout": "近景 Layout",
    "Layout Tuner Target": "布局调参目标",
    "Near-field Layout Tuner Note": "近景 Layout 调参：使用当前捕获的投影/warp 图，保存 front-priority Layout 候选；不写正式 calibration.yaml。",
    "Far-field Layout Tuner Note": "远景 Layout 调参：基于 B-2 候选的每路投影/warp 图，再做三路 x/y/scale 与 B-2 权重选择合成；不使用近景 side suppression，也不写正式 calibration.yaml。",
    "Far-field layout tuning requires a B-2 candidate projection. Generate or load a B-2 candidate first.": "远景 Layout 调参需要先有 B-2 候选投影。请先生成或加载 B-2 候选。",
    "Save Far-field Layout Candidate": "保存远景 Layout 候选",
    "Clear Candidate": "清除候选",
    "Open Candidate Folder": "打开候选目录",
    "Apply to Preview Runtime": "应用到预览 Runtime",
    "Use for Current Preview Only": "仅用于当前预览",
    "Clear Near-field Layout Candidate": "清除近景 Layout 候选",
    "Apply Near-field Preview": "应用近景预览",
    "Only applies to the current preview worker; does not write calibration.yaml or the formal profile.": "只应用到当前预览 worker；不写 calibration.yaml，也不写正式 profile。",
    "No Layout Candidate V2 loaded.": "未加载 Layout Candidate V2。",
    "Loaded Layout Candidate V2": "已加载 Layout Candidate V2",
    "Projection candidate is research-only and not connected to runtime yet.": "Projection 候选仍是研发线，尚未接入 runtime。",
    "Near-field mode requires a loaded Layout Candidate V2.": "Near-field 模式需要先加载 Layout Candidate V2。",
    "Auto stitch runtime mode is not implemented; falling back to Far-field.": "Auto 拼接算法模式尚未实现；当前回退 Far-field。",
    "Choose Layout Candidate V2": "选择 Layout Candidate V2",
    "Layout Candidate (*.yaml);;All Files (*)": "Layout Candidate (*.yaml);;所有文件 (*)",
    "Runtime layout candidate load failed": "Runtime layout candidate 加载失败",
    "Runtime mode applied. Open the Canvas tab to see the effect.": "Runtime 模式已应用。打开 Canvas 拼接画布可看到效果。",
    "Near-field runtime uses current perspective warp; experimental candidate stitch was switched back to template.": "Near-field runtime 使用当前 perspective warp；已从实验候选拼接切回模板拼接。",
    "Near-field stitched view": "近景优先拼接画面",
    "Far-field stitched view": "远景优先拼接画面",
    "Experimental": "实验性",
    "Advanced": "高级",
    "Experimental candidate live view": "实验性候选实时画面",
    "Template stitching": "模板拼接",
    "B-2 candidate view": "B-2 候选视图",
    "Back to Grid": "返回多路",
    "Multi-camera monitoring": "多路监看",
    "Single-camera view: {camera}": "单路查看：{camera}",
    "Static preview": "静态预览",
    "Live stitched view": "实时拼接画面",
    "Static stitched view": "静态拼接画面",
    "Live preview stopped": "实时预览已停止",
    "Video stream unavailable": "视频流不可用",
    "Last error: {error}": "最后错误：{error}",
    "Use Back to Grid to leave this view.": "可点击“返回多路”离开此视图。",
    "Static image": "静态图片",
    "Stitched surround canvas": "环视拼接画布",
    "Canvas": "拼接画布",
    "Warped Views": "透视变换预览",
    "Camera": "相机",
    "Enabled": "启用",
    "Status": "状态",
    "Frames": "帧数",
    "Source": "输入源",
    "Active camera count": "启用相机数量",
    "Save Camera Config": "保存相机配置",
    "Reload Config": "重新加载配置",
    "Camera Sources": "相机输入源",
    "Use": "使用",
    "Name": "名称",
    "Type": "类型",
    "Browse": "浏览",
    "Width": "宽度",
    "Height": "高度",
    "Performance": "性能",
    "Stitch FPS": "拼接帧率",
    "Preview FPS": "预览刷新率",
    "Max input width": "最大输入宽度",
    "Original": "原始尺寸",
    "Refresh warped previews": "刷新变换预览",
    "Use undistort live": "实时去畸变",
    "Chessboard": "棋盘格",
    "Total square columns": "总格子列数",
    "Total square rows": "总格子行数",
    "Square size mm": "格子边长 mm",
    "Load Calibration Image": "加载标定图片",
    "Detect Chessboard": "检测棋盘格",
    "Calibrate From Folder": "从文件夹标定",
    "Preview Undistort": "预览去畸变",
    "Export Quality Report": "导出质量报告",
    "Save Board Settings": "保存标定板设置",
    "Save Calibration Snapshot": "保存三路校准快照",
    "Topology diagnostics": "拓扑诊断",
    "Calibration snapshot saved: {path}": "校准快照已保存：{path}",
    "Calibration snapshot failed": "校准快照保存失败",
    "Perspective": "透视参数",
    "Calibration image": "标定图像",
    "Perspective points for selected camera": "当前相机透视点",
    "Index": "序号",
    "Source X": "源点 X",
    "Source Y": "源点 Y",
    "Target X": "目标点 X",
    "Target Y": "目标点 Y",
    "Apply Point Edits To Config": "应用点位到配置",
    "Preview Warp": "预览透视变换",
    "Warp preview": "变换预览",
    "Seams & Image Set": "拼接缝与标定图集",
    "Geometry Diagnostics": "三路几何诊断",
    "Refresh Geometry Diagnostics": "刷新几何诊断",
    "50% Alpha": "50% 透明叠加",
    "No Feather": "关闭 Feather 硬切",
    "Final Canvas": "最终拼接画布",
    "Pairwise Candidates": "两两标定候选",
    "Run Pairwise Auto Detection": "运行两两自动检测",
    "Manual Correspondences": "人工对应点",
    "Clear Points": "清空点位",
    "Generate Candidate": "生成候选",
    "Cancel": "取消",
    "Fisheye Sample Session": "鱼眼标定样本",
    "Create Session": "新建采集会话",
    "Load Session": "加载采集会话",
    "Capture Sample": "采集样本",
    "Board type": "标定板类型",
    "Board definition confirmed": "已确认标定板物理参数",
    "Physical board ID": "物理标定板编号",
    "Same physical board confirmed": "确认两路为同一块物理板",
    "Quality gates": "质量门控",
    "Minimum board area": "最小标定板面积",
    "Minimum Laplacian variance": "最低清晰度（拉普拉斯方差）",
    "Maximum pair time delta": "Pair 最大时间差",
    "No active session": "当前没有采集会话",
    "Sample capture": "样本采集",
    "Inner corner columns": "内角点列数",
    "Inner corner rows": "内角点行数",
    "Marker columns": "Marker 列数",
    "Marker rows": "Marker 行数",
    "Charuco squares X": "Charuco 方格列数",
    "Charuco squares Y": "Charuco 方格行数",
    "Square size": "方格边长",
    "Marker length": "Marker 边长",
    "Charuco square length": "Charuco 方格边长",
    "Charuco marker length": "Charuco Marker 边长",
    "Marker separation": "Marker 间距",
    "ArUco dictionary": "ArUco 字典",
    "Target": "对象",
    "Accepted": "已接受",
    "Goal": "目标",
    "Rejected": "已拒绝",
    "Missing coverage": "覆盖不足区域",
    "Last rejection": "最近拒绝原因",
    "Intrinsic": "单目内参",
    "Pair": "相邻 Pair",
    "Calibration Wizard": "标定向导",
    "Sample Session Details": "采集会话（详细）",
    "Previous": "上一步",
    "Next": "下一步",
    "Check Current Pair": "检查当前两路画面",
    "Continue Session": "继续已有会话",
    "Refresh Seam Canvas": "刷新拼接缝画布",
    "Save Seam Points": "保存拼接缝",
    "Center Seams in Overlaps": "拼接缝置于重叠区中心",
    "Feather width": "羽化宽度",
    "Stitch seam endpoints": "拼接缝端点",
    "Seam": "拼接缝",
    "Point": "点",
    "Role": "作用",
    "Calibration Image Set": "标定图集",
    "Choose Folder": "选择文件夹",
    "Capture Current Frame": "采集当前帧",
    "Scan Chessboard": "扫描棋盘格",
    "File": "文件",
    "Readable": "可读取",
    "Corners": "角点数",
    "Project Management": "项目管理",
    "QGC Video Output": "QGC 视频输出",
    "Output Type": "输出方式",
    "UDP Local Low Latency": "UDP 本机低延迟",
    "RTSP Standard Service": "RTSP 标准服务",
    "RTSP publish URL": "RTSP 发布地址",
    "UDP MPEG-TS URL": "UDP MPEG-TS 地址",
    "Output URL": "输出地址",
    "Output FPS": "输出帧率",
    "Output Bitrate": "输出码率",
    "RTSP Transport": "RTSP 传输",
    "FFmpeg Path": "FFmpeg 路径",
    "Save QGC Output Settings": "保存 QGC 输出设置",
    "Start QGC Output Service": "启动 QGC 输出服务",
    "Stop QGC Output Service": "停止 QGC 输出服务",
    "QGC output service is stopped.": "QGC 输出服务未运行。",
    "QGC output service started: {url}": "QGC 输出服务已启动：{url}",
    "QGC output service stopped.": "QGC 输出服务已停止。",
    "QGC output service failed to start.": "QGC 输出服务启动失败。",
    "QGC output service already running.": "QGC 输出服务已在运行。",
    "QGC output service exited with code {code}.": "QGC 输出服务已退出，代码 {code}。",
    "QGC output service": "QGC 输出服务",
    "Current B-2 candidate view requires an available candidate directory.": "当前 B-2 候选视图需要可用的候选目录。",
    "Start streaming the current GUI stitched canvas to QGC.": "将当前 GUI 拼接画面开始推流到 QGC。",
    "Stop streaming the current GUI stitched canvas to QGC.": "停止向 QGC 推送当前 GUI 拼接画面。",
    "QGC output settings saved.": "QGC 输出设置已保存。",
    "QGC output settings are saved in cameras.yaml and do not modify calibration.yaml.": "QGC 输出设置保存在 cameras.yaml，不会修改 calibration.yaml。",
    "RTSP output publishes to an RTSP server. QGC should open the same stream URL.": "RTSP 输出会发布到 RTSP 服务端，QGC 应打开同一个流地址。",
    "UDP MPEG-TS sends directly to a UDP port and is usually better for same-machine low-latency QGC preview.": "UDP MPEG-TS 直接发送到 UDP 端口，通常更适合同机低延迟 QGC 预览。",
    "Use udp://127.0.0.1:5600 for local low-latency bridge, or rtsp://127.0.0.1:8554/deepshark when using MediaMTX/RTSP service.": "同机低延迟桥接建议使用 udp://127.0.0.1:5600；使用 MediaMTX/RTSP 服务时使用 rtsp://127.0.0.1:8554/deepshark。",
    "Projection Research Overview": "投影研发概览",
    "Projection Research Note": "Projection 研发线用于 fisheye rectilinear / equirectangular / A-B comparison 等实验。Near-field Fisheye 只影响 Near-field；Far-field Default 不受影响；Far-field Custom 使用 B-2 per-camera projection 基底。",
    "Runtime projection controls remain in Realtime Monitor for current preview switching; layout-specific projection controls are in Layout Tuning.": "运行态投影切换仍在实时监看的拼接算法面板中；布局候选相关投影参数在布局调参中。",
    "No project file loaded. Current configs are stored in configs/*.yaml.": "尚未加载项目文件。当前配置保存在 configs/*.yaml。",
    "Save Project As": "项目另存为",
    "Open Project": "打开项目",
    "Backup Current Configs": "备份当前配置",
    "Export Runtime Config": "导出运行配置",
    "Export Project Package": "导出项目包",
    "Import Project Package": "导入项目包",
    "Validate Project Package": "校验项目包",
    "Project Package (*.yaml *.yml);;All Files (*)": "项目包 (*.yaml *.yml);;所有文件 (*)",
    "Choose project package export folder": "选择项目包导出目录",
    "Open project package manifest": "打开项目包 manifest",
    "Validate project package manifest": "校验项目包 manifest",
    "Export project package failed": "导出项目包失败",
    "Import project package failed": "导入项目包失败",
    "Validate project package failed": "校验项目包失败",
    "Project package exported: {path}": "项目包已导出：{path}",
    "Project package valid: {path}": "项目包校验通过：{path}",
    "Project package invalid: {errors}": "项目包校验失败：{errors}",
    "Activate project package?": "是否激活项目包？",
    "The package is valid. Activate it now?\n\nThis will back up current configs, then replace active configs with the package configs. Candidate files remain inside the package and calibration.yaml is not edited during export/validation.": "项目包校验通过。现在激活吗？\n\n激活会先备份当前 configs，然后用项目包内的 configs 替换当前配置。候选文件仍保留在项目包内；导出/校验阶段不会编辑 calibration.yaml。",
    "Project package activated: {path}": "项目包已激活：{path}",
    "Project package validated but not activated: {path}": "项目包已校验，但未激活：{path}",
    "Project files bundle calibration.yaml, cameras.yaml, and network.yaml into one portable .dsvs.yaml file.\n\nUse backups before large calibration or seam edits. Runtime export is the compact configuration intended for a future service/QGC bridge.": "项目文件会把 calibration.yaml、cameras.yaml 和 network.yaml 打包成一个便携的 .dsvs.yaml 文件。\n\n大幅修改标定或拼接缝前建议先备份。运行配置导出用于后续独立服务或 QGC 桥接。",
    "Project Package exports configs and active candidates into a folder with package-relative paths. Import validates first; activation backs up current configs before replacing them.": "项目包会把配置和当前候选导出到一个使用包内相对路径的目录。导入会先校验；激活前会备份当前 configs，再替换配置。",
    "No images": "没有图片",
    "No matching camera images were found.": "没有找到匹配相机名称的图片。",
    "Stitch failed": "拼接失败",
    "No result": "没有结果",
    "Run preview first.": "请先运行预览。",
    "No image": "没有图片",
    "Load a calibration image first.": "请先加载标定图片。",
    "No intrinsics": "没有内参",
    "No saved camera intrinsics for {camera}.": "{camera} 没有已保存的相机内参。",
    "No calibration": "没有标定结果",
    "No calibration result saved for {camera}.": "{camera} 没有已保存的标定结果。",
    "No frame": "没有当前帧",
    "Camera failed: {error}": "相机失败：{error}",
    "Load images or start live preview first, then capture the selected camera frame.": "请先加载图片或开始实时预览，再采集当前相机帧。",
    "Reloaded": "已重新加载",
    "Configuration and editor controls were reloaded.": "配置与编辑控件已重新加载。",
    "Configuration changed on disk": "磁盘配置已变化",
    "Configuration changed on disk. Reload before saving to avoid overwriting newer settings.": "磁盘配置已被其他窗口或进程修改。请先重新加载，避免覆盖较新的设置。",
    "Choose image directory": "选择图片目录",
    "Choose video file": "选择视频文件",
    "Live preview running": "实时预览运行中",
    "Preview stopped": "预览已停止",
    "Loaded image directory": "已加载图片目录",
    "Live frame": "实时帧",
    "No warped frame": "没有变换画面",
    "stitched canvas": "拼接画布",
    "Saved configs to {path}": "配置已保存到 {path}",
    "Auto backup created: {path}": "已自动备份：{path}",
    "Auto backup failed: {error}": "自动备份失败：{error}",
    "Selected {path}": "已选择 {path}",
    "Saved results to {path}": "结果已保存到 {path}",
    "{status}: {count} active frames | stitch {ms:.1f} ms": "{status}：{count} 路有效画面 | 拼接 {ms:.1f} ms",
    "Save DeepShark View Studio project": "保存 DeepShark View Studio 项目",
    "DeepShark Project (*.yaml *.yml)": "DeepShark 项目 (*.yaml *.yml)",
    "Save project failed": "保存项目失败",
    "Saved project: {path}": "项目已保存：{path}",
    "Open DeepShark View Studio project": "打开 DeepShark View Studio 项目",
    "DeepShark Project (*.yaml *.yml);;All Files (*)": "DeepShark 项目 (*.yaml *.yml);;所有文件 (*)",
    "Open project failed": "打开项目失败",
    "Loaded project: {path}": "项目已加载：{path}",
    "Loaded project version {version} from {path}": "已从 {path} 加载项目版本 {version}",
    "Backup failed": "备份失败",
    "Backup created: {path}": "备份已创建：{path}",
    "Export runtime config": "导出运行配置",
    "YAML (*.yaml *.yml)": "YAML (*.yaml *.yml)",
    "Export failed": "导出失败",
    "Runtime config exported: {path}": "运行配置已导出：{path}",
    "Saved chessboard settings.": "标定板设置已保存。",
    "Choose calibration image": "选择标定图片",
    "Images (*.jpg *.jpeg *.png *.bmp);;All Files (*)": "图片 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)",
    "Load failed": "加载失败",
    "Could not read image: {path}": "无法读取图片：{path}",
    "Loaded calibration image: {path}": "已加载标定图片：{path}",
    "Chessboard detection: {status}; inner pattern={cols}x{rows}; corners={corners}; square={square}mm": "棋盘格检测：{status}；内角点={cols}x{rows}；角点数={corners}；格子={square}mm",
    "OK": "成功",
    "FAILED": "失败",
    "Choose calibration image folder": "选择标定图片文件夹",
    "No readable calibration images were found.": "没有找到可读取的标定图片。",
    "Calibration failed": "标定失败",
    "Calibrated {camera}: RMS={rms:.4f}, mean reprojection={error:.4f}px, usable images={count}. Report: {path}": "{camera} 标定完成：RMS={rms:.4f}，平均重投影误差={error:.4f}px，可用图片={count}。报告：{path}",
    "Undistort failed": "去畸变失败",
    "{camera} undistort preview": "{camera} 去畸变预览",
    "Previewed undistortion for {camera}.": "已预览 {camera} 去畸变。",
    "Export calibration quality report": "导出标定质量报告",
    "Markdown (*.md);;All Files (*)": "Markdown (*.md);;所有文件 (*)",
    "Report export failed": "报告导出失败",
    "Exported calibration report: {path}": "标定报告已导出：{path}",
    "Point {index}: {x:.1f}, {y:.1f}": "点 {index}：{x:.1f}, {y:.1f}",
    "Invalid points": "点位无效",
    "Saved perspective points for {camera}.": "{camera} 透视点已保存。",
    "Preview failed": "预览失败",
    "{camera} warp preview": "{camera} 透视变换预览",
    "Previewed warp for {camera}.": "已预览 {camera} 透视变换。",
    "{seam} point {index}: {x:.1f}, {y:.1f}": "{seam} 第 {index} 点：{x:.1f}, {y:.1f}",
    "Invalid seam points": "拼接缝点位无效",
    "Saved seam points to calibration.yaml.": "拼接缝点位已保存到 calibration.yaml。",
    "Captured calibration frame: {path}": "已采集标定帧：{path}",
    "Scanned {count} calibration images; chessboard OK: {ok}.": "已扫描 {count} 张标定图片；棋盘格成功：{ok}。",
    "yes": "是",
    "no": "否",
    "Live": "实时",
    "Frame": "单帧",
    "Idle": "空闲",
    "Connecting": "连接中",
    "Failed": "失败",
    "Stopped": "已停止",
    "Stopping": "停止中",
    "Layout Tuner": "布局调参",
    "Experimental front-priority layout tuner. Uses current perspective/template warped images only; projection candidates are not used here and formal calibration.yaml is never written.": "实验性前视优先布局调参。这里只使用当前透视/模板 warp 后的图像，不使用 projection 候选，也绝不写入正式 calibration.yaml。",
    "Preset": "预设",
    "Balanced": "均衡",
    "Conservative": "保守",
    "Wide": "宽视场",
    "Hard Seam": "硬切",
    "Front Smaller": "缩小前视",
    "Core Layout": "核心布局",
    "Pair Boundary Params": "左右边界参数",
    "Lock Left/Right Pair Params": "锁定左右参数",
    "Left Side Shift": "左侧收边",
    "Left Side Visible": "左侧可见比例",
    "Left Feather": "左侧 Feather",
    "Right Side Shift": "右侧收边",
    "Right Side Visible": "右侧可见比例",
    "Right Feather": "右侧 Feather",
    "Post-warp Camera Adjustment": "Warp 后相机画面调整",
    "X Offset": "X 偏移",
    "Y Offset": "Y 偏移",
    "Scale": "缩放",
    "Vertical Safe Ratio": "上下安全比例",
    "Side Vertical Fade": "侧路上下淡出",
    "front_left": "左前",
    "front": "前视",
    "front_right": "右前",
    "Reset Camera Adjust": "重置画面调整",
    "Reset Front Only": "只重置前视",
    "Reset Side Cameras": "重置左右侧路",
    "Lock Side Camera Scale": "锁定左右侧路缩放",
    "Advanced Vertical Safety": "高级：上下边缘保护",
    "Moves front_left right and front_right left.": "让 front_left 向右收、front_right 向左收，用于减少侧路重复显示。",
    "Moves this side camera inward after warp.": "在 warp 后将当前侧路画面向内收。",
    "Final width crop only; no scaling.": "最终输出宽度；只裁剪，不缩放。",
    "Allowed side-camera edge fraction.": "允许侧路相机显示的边缘比例。",
    "Narrow seam feather; 0 means hard seam.": "拼接边界附近的窄 feather；0 表示硬切。",
    "Preview-only x/y/scale after warp; not real extrinsics calibration.": "仅用于预览的 warp 后 x/y/缩放微调；不是外参标定。",
    "Horizontal preview offset after warp.": "Warp 后水平预览偏移。",
    "Vertical preview offset after warp.": "Warp 后垂直预览偏移。",
    "Preview-only scale around valid-image center.": "围绕有效画面中心做仅预览缩放。",
    "Final height crop only; no scaling.": "最终输出高度；只裁剪，不缩放。",
    "Suppress only side cameras near top/bottom bands.": "只压制侧路相机的上下边缘，不压制 front。",
    "Smooth fade width for side top/bottom suppression.": "侧路上下边缘压制的平滑过渡宽度。",
    "Capture Preview Frame": "捕获预览帧",
    "Refresh Preview": "刷新预览",
    "Reset to Baseline": "恢复基准",
    "Save Layout Candidate": "保存布局候选",
    "Save Near-field Layout Candidate": "保存近景 Layout 候选",
    "Export Preview PNG": "导出预览 PNG",
    "Open Candidate Folder": "打开候选目录",
    "Capture preview frame to tune layout.": "请先捕获预览帧，再调整布局。",
    "No cached warped images. Click Capture Preview Frame.": "没有缓存的 warped images。请点击“捕获预览帧”。",
    "Missing frames for layout tuner: {cameras}": "布局调参缺少画面：{cameras}",
    "Layout tuner capture failed: {error}": "布局调参捕获失败：{error}",
    "Layout tuner render failed: {error}": "布局调参渲染失败：{error}",
    "Layout tuner preview rendered in {ms:.1f} ms": "布局调参预览已渲染：{ms:.1f} ms",
    "Capture and render a preview before saving.": "请先捕获并渲染预览，再保存。",
    "Layout tuner candidate save failed: {error}": "布局候选保存失败：{error}",
    "Layout candidate saved: {path}": "布局候选已保存：{path}",
    "Capture and render a preview before export.": "请先捕获并渲染预览，再导出。",
    "Export layout tuner preview": "导出布局调参预览",
    "Layout tuner preview exported: {path}": "布局调参预览已导出：{path}",
}


def i18n(language: str, text: str, **kwargs: Any) -> str:
    translated = ZH_CN.get(text, text) if language == "zh_CN" else text
    return translated.format(**kwargs) if kwargs else translated


def rejection_advice(reason: str) -> str:
    """Translate a quality-gate reason into a concrete next action."""
    text = str(reason or "").strip()
    lowered = text.lower()
    if not text:
        return "暂无拒绝记录。"
    if "blur" in lowered or "模糊" in text:
        return "停稳标定板约 1 秒，避免手抖和运动模糊后再拍。"
    if (
        "area is too small" in lowered
        or "board area" in lowered
        or "面积太小" in text
    ):
        return "让标定板靠近相机一些，使它在画面中占据更大区域。"
    if (
        "duplicate" in lowered
        or "too similar" in lowered
        or "姿态重复" in text
        or "重复姿态" in text
    ):
        return "把标定板移到尚未覆盖的边缘或角落，并改变距离和倾角。"
    if (
        "time delta" in lowered
        or "timestamps unavailable" in lowered
        or "时间差" in text
    ):
        return "停稳同一块标定板约 1 秒，再采集这一组 Pair。"
    if (
        "coverage" in lowered
        or "覆盖不足" in text
        or "位置或距离" in text
    ):
        return "补拍画面四角、边缘，以及近、中、远不同距离和倾角。"
    if (
        "not detected" in lowered
        or "common confirmed board points" in lowered
        or "无法识别" in text
        or "共同点" in text
    ):
        return "确认整块标定板清晰可见、无遮挡，并核对行列数和板类型。"
    return "查看详细拒绝原因，调整标定板位置、清晰度或可见范围后重拍。"


def cv_to_pixmap(image: np.ndarray, max_width: int = 720, max_height: int = 480) -> QPixmap:
    """Convert a BGR OpenCV image to a scaled Qt pixmap."""
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width = rgb.shape[:2]
    bytes_per_line = 3 * width
    qimage = QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
    pixmap = QPixmap.fromImage(qimage)
    return pixmap.scaled(
        max(1, max_width),
        max(1, max_height),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class PreviewLayoutMode(Enum):
    GRID = "grid"
    FOCUS = "focus"
    STITCHED = "stitched"


class PreviewContentMode(Enum):
    STOPPED = "stopped"
    LIVE = "live"
    STILL = "still"


class NoWheelSpinBox(QSpinBox):
    """Spin box that ignores mouse-wheel value changes."""

    def wheelEvent(self, event) -> None:  # noqa: N802
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """Double spin box that ignores mouse-wheel value changes."""

    def wheelEvent(self, event) -> None:  # noqa: N802
        event.ignore()


class NoWheelComboBox(QComboBox):
    """Combo box that ignores mouse-wheel selection changes."""

    def wheelEvent(self, event) -> None:  # noqa: N802
        event.ignore()


class NoWheelSlider(QSlider):
    """Slider that ignores mouse-wheel value changes."""

    def wheelEvent(self, event) -> None:  # noqa: N802
        event.ignore()


class ImageView(QLabel):
    """Simple image display surface."""

    doubleClicked = Signal(str)

    def __init__(self, title: str, camera_id: str = ""):
        super().__init__(title)
        self.placeholder = title
        self.camera_id = camera_id
        self.overlay_text = ""
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(160, 100)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setWordWrap(True)
        self.setStyleSheet(
            "QLabel { background: #101820; color: #d7e0ea; border: 1px solid #334155; padding: 4px; }"
        )

    def set_image(self, image: np.ndarray, title: str = "") -> None:
        self.setPixmap(cv_to_pixmap(image, self.width() - 12, self.height() - 12))
        if title:
            self.setToolTip(title)

    def set_overlay(self, title: str, status: str = "") -> None:
        self.overlay_text = f"{title} · {status}" if status else title
        self.update()

    def set_placeholder(self, text: str | None = None) -> None:
        self.clear()
        self.setText(text or self.placeholder)

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        pixmap = self.pixmap()
        if not self.overlay_text or pixmap is None or pixmap.isNull():
            return
        painter = QPainter(self)
        painter.setPen(QColor("#f8fafc"))
        painter.setFont(QFont(self.font().family(), 9, QFont.Weight.DemiBold))
        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(self.overlay_text)
        text_height = metrics.height()
        background = QRectF(8, 8, text_width + 18, text_height + 10)
        painter.fillRect(background, QColor(15, 23, 42, 210))
        painter.drawText(
            background.adjusted(9, 5, -9, -5),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self.overlay_text,
        )

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if self.camera_id:
            self.doubleClicked.emit(self.camera_id)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class PointEditorView(QLabel):
    """Image view with draggable perspective source points."""

    pointMoved = Signal(int, float, float)

    def __init__(self):
        super().__init__("Load a calibration image, then drag source points.")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 220)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setMouseTracking(True)
        self.setStyleSheet(
            "QLabel { background: #101820; color: #d7e0ea; border: 1px solid #334155; padding: 4px; }"
        )
        self._image: np.ndarray | None = None
        self._pixmap: QPixmap | None = None
        self._points: list[list[float]] = []
        self._drag_index: int | None = None
        self._display_rect = QRectF()
        self.source_coordinate_space = ""
        self.source_reference_size: list[int] | None = None
        self._colors = [
            QColor("#f97316"),
            QColor("#22c55e"),
            QColor("#38bdf8"),
            QColor("#f43f5e"),
        ]

    def set_source_coordinate_contract(
        self,
        coordinate_space: str,
        reference_size: list[int] | None,
    ) -> None:
        self.source_coordinate_space = str(coordinate_space)
        self.source_reference_size = (
            [int(reference_size[0]), int(reference_size[1])]
            if isinstance(reference_size, (list, tuple))
            and len(reference_size) == 2
            else None
        )

    def set_editor_image(self, image: np.ndarray | None) -> None:
        self._image = None if image is None else image.copy()
        self._pixmap = None if image is None else cv_to_pixmap(image, self.width() - 12, self.height() - 12)
        self.update()

    def set_points(self, points: list[list[float]]) -> None:
        self._points = [[float(x), float(y)] for x, y in points[:4]]
        self.update()

    def points(self) -> list[list[float]]:
        return [[round(x, 3), round(y, 3)] for x, y in self._points]

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._image is not None:
            self._pixmap = cv_to_pixmap(self._image, self.width() - 12, self.height() - 12)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101820"))

        if self._pixmap is None or self._image is None:
            painter.setPen(QColor("#d7e0ea"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())
            painter.end()
            return

        x = (self.width() - self._pixmap.width()) / 2
        y = (self.height() - self._pixmap.height()) / 2
        self._display_rect = QRectF(x, y, self._pixmap.width(), self._pixmap.height())
        painter.drawPixmap(int(x), int(y), self._pixmap)

        if len(self._points) >= 2:
            polygon_points = [self._image_to_widget(px, py) for px, py in self._points]
            painter.setPen(QPen(QColor("#facc15"), 2, Qt.PenStyle.DashLine))
            for i in range(len(polygon_points)):
                painter.drawLine(polygon_points[i], polygon_points[(i + 1) % len(polygon_points)])

        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        for index, (px, py) in enumerate(self._points):
            point = self._image_to_widget(px, py)
            color = self._colors[index % len(self._colors)]
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor("#0f172a"), 2))
            painter.drawEllipse(point, 9, 9)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(QRectF(point.x() - 8, point.y() - 8, 16, 16), Qt.AlignmentFlag.AlignCenter, str(index + 1))
        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or not self._points:
            return
        pos = event.position()
        nearest_index = None
        nearest_distance = 999999.0
        for index, (px, py) in enumerate(self._points):
            point = self._image_to_widget(px, py)
            distance = (point.x() - pos.x()) ** 2 + (point.y() - pos.y()) ** 2
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_index = index
        if nearest_index is not None and nearest_distance <= 24 * 24:
            self._drag_index = nearest_index

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_index is None:
            return
        image_point = self._widget_to_image(event.position())
        if image_point is None:
            return
        x, y = image_point
        self._points[self._drag_index] = [x, y]
        self.pointMoved.emit(self._drag_index, x, y)
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_index = None

    def _image_to_widget(self, x: float, y: float) -> QPointF:
        if self._image is None or self._display_rect.isNull():
            return QPointF(x, y)
        image_h, image_w = self._image.shape[:2]
        scale_x = self._display_rect.width() / image_w
        scale_y = self._display_rect.height() / image_h
        return QPointF(self._display_rect.left() + x * scale_x, self._display_rect.top() + y * scale_y)

    def _widget_to_image(self, point: QPointF) -> tuple[float, float] | None:
        if self._image is None or self._display_rect.isNull():
            return None
        image_h, image_w = self._image.shape[:2]
        x = (point.x() - self._display_rect.left()) / self._display_rect.width() * image_w
        y = (point.y() - self._display_rect.top()) / self._display_rect.height() * image_h
        x = min(max(x, 0.0), float(image_w - 1))
        y = min(max(y, 0.0), float(image_h - 1))
        return x, y


class CorrespondenceImageView(QLabel):
    """Clickable raw-frame view used only for temporary pair correspondences."""

    pointRequested = Signal(float, float)

    def __init__(self, title: str):
        super().__init__(title)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 220)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setStyleSheet(
            "QLabel { background: #101820; color: #d7e0ea; "
            "border: 1px solid #334155; padding: 4px; }"
        )
        self._image: np.ndarray | None = None
        self._pixmap: QPixmap | None = None
        self._points: list[list[float]] = []
        self._display_rect = QRectF()

    def set_editor_image(self, image: np.ndarray) -> None:
        self._image = image.copy()
        self._rescale_pixmap()

    def set_points(self, points: list[list[float]]) -> None:
        self._points = [[float(x), float(y)] for x, y in points]
        self.update()

    def _rescale_pixmap(self) -> None:
        if self._image is None:
            self._pixmap = None
        else:
            self._pixmap = cv_to_pixmap(
                self._image,
                self.width() - 12,
                self.height() - 12,
            )
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._rescale_pixmap()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101820"))
        if self._pixmap is None or self._image is None:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())
            painter.end()
            return
        x = (self.width() - self._pixmap.width()) / 2
        y = (self.height() - self._pixmap.height()) / 2
        self._display_rect = QRectF(
            x,
            y,
            self._pixmap.width(),
            self._pixmap.height(),
        )
        painter.drawPixmap(int(x), int(y), self._pixmap)
        painter.setFont(QFont(self.font().family(), 9, QFont.Weight.Bold))
        for index, (point_x, point_y) in enumerate(self._points):
            image_height, image_width = self._image.shape[:2]
            widget_point = QPointF(
                self._display_rect.left()
                + point_x / image_width * self._display_rect.width(),
                self._display_rect.top()
                + point_y / image_height * self._display_rect.height(),
            )
            painter.setBrush(QBrush(QColor("#facc15")))
            painter.setPen(QPen(QColor("#111827"), 2))
            painter.drawEllipse(widget_point, 8, 8)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(
                QRectF(
                    widget_point.x() - 8,
                    widget_point.y() - 8,
                    16,
                    16,
                ),
                Qt.AlignmentFlag.AlignCenter,
                str(index + 1),
            )
        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if (
            event.button() != Qt.MouseButton.LeftButton
            or self._image is None
            or self._display_rect.isNull()
            or not self._display_rect.contains(event.position())
        ):
            return
        image_height, image_width = self._image.shape[:2]
        x = (
            (event.position().x() - self._display_rect.left())
            / self._display_rect.width()
            * image_width
        )
        y = (
            (event.position().y() - self._display_rect.top())
            / self._display_rect.height()
            * image_height
        )
        self.pointRequested.emit(
            min(max(x, 0.0), float(image_width - 1)),
            min(max(y, 0.0), float(image_height - 1)),
        )


class ManualCorrespondenceDialog(QDialog):
    """Collect matched raw-frame points without touching calibration YAML."""

    def __init__(
        self,
        pair: PairDefinition,
        left_image: np.ndarray,
        right_image: np.ndarray,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(
            f"Manual correspondences: "
            f"{pair.left_camera} <-> {pair.right_camera}"
        )
        self.left_points: list[list[float]] = []
        self.right_points: list[list[float]] = []
        root = QVBoxLayout(self)
        instructions = QLabel(
            "Select one point in the left image, then the matching point in "
            "the right image. Repeat for at least four pairs. The candidate "
            "is diagnostic only and will not update calibration.yaml."
        )
        instructions.setWordWrap(True)
        root.addWidget(instructions)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.left_view = CorrespondenceImageView(pair.left_camera)
        self.right_view = CorrespondenceImageView(pair.right_camera)
        self.left_view.set_editor_image(left_image)
        self.right_view.set_editor_image(right_image)
        self.left_view.pointRequested.connect(self._add_left_point)
        self.right_view.pointRequested.connect(self._add_right_point)
        splitter.addWidget(self.left_view)
        splitter.addWidget(self.right_view)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)
        self.status_label = QLabel()
        root.addWidget(self.status_label)
        buttons = QHBoxLayout()
        clear_button = QPushButton("Clear Points")
        generate_button = QPushButton("Generate Candidate")
        cancel_button = QPushButton("Cancel")
        clear_button.clicked.connect(self.clear_points)
        generate_button.clicked.connect(self.accept_candidate)
        cancel_button.clicked.connect(self.reject)
        buttons.addStretch(1)
        buttons.addWidget(clear_button)
        buttons.addWidget(generate_button)
        buttons.addWidget(cancel_button)
        root.addLayout(buttons)
        parent_width = parent.width() if parent is not None else 1200
        parent_height = parent.height() if parent is not None else 700
        self.resize(min(1200, parent_width), min(700, parent_height))
        self._refresh_status()

    def _add_left_point(self, x: float, y: float) -> None:
        if len(self.left_points) != len(self.right_points):
            self.status_label.setText(
                "Select the matching point in the right image first."
            )
            return
        self.left_points.append([x, y])
        self.left_view.set_points(self.left_points)
        self._refresh_status()

    def _add_right_point(self, x: float, y: float) -> None:
        if len(self.left_points) != len(self.right_points) + 1:
            self.status_label.setText(
                "Select a point in the left image first."
            )
            return
        self.right_points.append([x, y])
        self.right_view.set_points(self.right_points)
        self._refresh_status()

    def clear_points(self) -> None:
        self.left_points.clear()
        self.right_points.clear()
        self.left_view.set_points([])
        self.right_view.set_points([])
        self._refresh_status()

    def _refresh_status(self) -> None:
        self.status_label.setText(
            f"Matched pairs: {len(self.right_points)}. "
            "Minimum required: 4."
        )

    def accept_candidate(self) -> None:
        if (
            len(self.left_points) < 4
            or len(self.left_points) != len(self.right_points)
        ):
            self.status_label.setText(
                "At least four complete left/right point pairs are required."
            )
            return
        self.accept()


class CalibrationCandidateDialog(QDialog):
    """Read-only B-2 candidate report and preview window."""

    def __init__(
        self,
        candidate_directory: str | Path,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.candidate_directory = Path(candidate_directory)
        self.candidate, self.report = load_calibration_candidate(
            self.candidate_directory
        )
        experimental = bool(self.candidate.get("experimental"))
        self.setWindowTitle(
            "B-2 实验性候选标定"
            if experimental
            else "B-2 候选标定"
        )
        self.resize(1180, 760)
        root = QVBoxLayout(self)

        warning = QLabel(
            "实验性候选，仅供诊断"
            if experimental
            else "候选求解结果，尚未应用"
        )
        warning.setStyleSheet(
            "QLabel { color: #b45309; font-size: 17px; "
            "font-weight: 700; padding: 8px; }"
        )
        root.addWidget(warning)
        self.candidate_safety_notice = QLabel(self._safety_notice_text())
        self.candidate_safety_notice.setWordWrap(True)
        self.candidate_safety_notice.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.candidate_safety_notice.setStyleSheet(
            "QLabel { color: #7c2d12; background: #fff7ed; "
            "border: 1px solid #fdba74; padding: 8px; }"
        )
        root.addWidget(self.candidate_safety_notice)
        summary = QTextEdit()
        summary.setReadOnly(True)
        summary.setMaximumHeight(230)
        summary.setPlainText(self._summary_text())
        root.addWidget(summary)

        self.preview_tabs = QTabWidget()
        self._add_candidate_previews()
        root.addWidget(self.preview_tabs, 1)

        actions = QHBoxLayout()
        path_label = QLabel(str(self.candidate_directory))
        path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        actions.addWidget(path_label, 1)
        export_button = QPushButton("导出报告")
        export_button.clicked.connect(self.export_report)
        self.apply_candidate_button = QPushButton("应用候选标定")
        self.apply_candidate_button.setEnabled(False)
        self.apply_candidate_button.setToolTip(
            (
                "experimental 候选禁止写入正式配置。"
                if experimental
                else "本阶段未启用正式应用流程。"
            )
        )
        close_button = QPushButton("返回采样向导补拍")
        close_button.clicked.connect(self.accept)
        actions.addWidget(export_button)
        actions.addWidget(self.apply_candidate_button)
        actions.addWidget(close_button)
        root.addLayout(actions)

    def _safety_notice_text(self) -> str:
        reasons = []
        readiness = self.candidate.get("readiness", {})
        reasons.extend(readiness.get("blocking_reasons", []))
        reasons.extend(readiness.get("quality_issues", []))
        for name, result in self.candidate.get("intrinsics", {}).items():
            used = int(result.get("used_sample_count", 0))
            outliers = int(result.get("outlier_count", 0))
            if result.get("status") != "success":
                reasons.append(f"{name} 的镜头参数候选未成功生成。")
            elif used and outliers / max(1, used + outliers) > 0.2:
                reasons.append(f"{name} 的异常样本比例较高，需要补拍。")
        for name, result in self.candidate.get("stereo_pairs", {}).items():
            used = int(result.get("used_sample_count", 0))
            outliers = int(result.get("outlier_count", 0))
            if result.get("status") != "success":
                reasons.append(
                    f"{name.replace('__', ' ↔ ')} 的相机位置关系候选未成功生成。"
                )
            elif used and outliers / max(1, used + outliers) > 0.2:
                reasons.append(
                    f"{name.replace('__', ' ↔ ')} 的异常样本比例较高。"
                )
        panorama = self.candidate.get("virtual_panorama") or {}
        if "rotation_only" in str(panorama.get("projection", "")):
            reasons.append(
                "三台相机光心不重合；当前仅旋转全景优先保证远景，"
                "近处人物和物体仍可能重影。"
            )
        if not reasons:
            reasons.append("候选尚未经过正式复核，本阶段不允许写入正式配置。")
        unique_reasons = list(dict.fromkeys(str(reason) for reason in reasons))
        suggestions = [
            rejection_advice(reason)
            for reason in unique_reasons
            if "光心不重合" not in reason
        ]
        unique_suggestions = list(dict.fromkeys(suggestions))
        return (
            "为什么不能应用：\n• "
            + "\n• ".join(unique_reasons)
            + "\n\n下一步：\n• "
            + "\n• ".join(
                unique_suggestions
                or [
                    "在 Grid 或 Focus 中检查近景，补充相邻 Pair 的位置、"
                    "距离和倾角覆盖后重新生成候选。"
                ]
            )
        )

    def _summary_text(self) -> str:
        lines = [
            (
                "等级：experimental / report-only"
                if self.candidate.get("experimental")
                else "等级：candidate / report-only"
            ),
            f"Session：{self.candidate.get('session_id', '')}",
            f"分辨率：{self.candidate.get('resolution', [])}",
            f"Rig 完整：{'是' if self.candidate['rig']['complete'] else '否'}",
            "",
            "候选内参：",
        ]
        for camera, result in self.candidate["intrinsics"].items():
            lines.append(
                f"  {camera}: {result['status']}，"
                f"RMS={result.get('rms_px')} px，"
                f"使用 {result.get('used_sample_count', 0)}，"
                f"异常 {result.get('outlier_count', 0)}"
            )
        lines.append("")
        lines.append("候选相邻相机关系：")
        for pair, result in self.candidate["stereo_pairs"].items():
            time_stats = result.get("time_delta_statistics", {})
            lines.append(
                f"  {pair.replace('__', ' ↔ ')}: {result['status']}，"
                f"RMS={result.get('rms_px')} px，"
                f"使用 {result.get('used_sample_count', 0)}，"
                f"异常 {result.get('outlier_count', 0)}，"
                f"时间差风险 {time_stats.get('risk_count', 0)}"
            )
        lines.extend(
            [
                "",
                "限制：Pair 仅有软件时间戳；近景物体存在物理视差。",
                "补充 Pair 的位置、距离和倾角覆盖后，"
                "才能申请正式可应用求解。",
            ]
        )
        for issue in self.candidate["readiness"].get(
            "quality_issues",
            [],
        ):
            lines.append(f"  - {issue}")
        return "\n".join(lines)

    def _add_preview(
        self,
        title: str,
        relative_path: str | None,
    ) -> None:
        view = ImageView(title)
        if relative_path:
            image = cv2.imread(
                str(self.candidate_directory / relative_path)
            )
            if image is not None:
                view.set_image(image, title)
            else:
                view.set_placeholder("预览文件读取失败")
        else:
            view.set_placeholder("当前候选没有生成此预览")
        self.preview_tabs.addTab(view, title)

    def _add_candidate_previews(self) -> None:
        panorama = self.candidate.get("virtual_panorama") or {}
        files = panorama.get("files", {})
        self._add_preview("候选全景", files.get("canvas"))
        for camera, path in files.get("remap_previews", {}).items():
            self._add_preview(f"Remap · {camera}", path)
        for pair, pair_files in files.get("pair_previews", {}).items():
            display = pair.replace("__", " ↔ ")
            self._add_preview(
                f"{display} · 50% Alpha",
                pair_files.get("alpha_50"),
            )
            self._add_preview(
                f"{display} · Hard Seam",
                pair_files.get("hard_seam"),
            )

    def export_report(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择报告导出目录",
            str(PROJECT_ROOT),
        )
        if not selected:
            return
        base = Path(selected) / (
            f"DeepShark_candidate_report_{self.candidate_directory.name}"
        )
        target = base
        suffix = 1
        while target.exists():
            target = Path(f"{base}_{suffix:02d}")
            suffix += 1
        target.mkdir(parents=True, exist_ok=False)
        for filename in ("candidate.yaml", "report.yaml"):
            shutil.copy2(
                self.candidate_directory / filename,
                target / filename,
            )
        previews = self.candidate_directory / "previews"
        if previews.exists():
            shutil.copytree(previews, target / "previews")
        QMessageBox.information(
            self,
            "导出报告",
            f"候选报告已导出到：\n{target}",
        )


class SeamEditorView(QLabel):
    """Canvas-coordinate seam editor with draggable seam endpoints."""

    seamPointMoved = Signal(str, int, float, float)

    def __init__(self):
        super().__init__("Drag seam endpoints on the surround canvas.")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 220)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setMouseTracking(True)
        self.setStyleSheet(
            "QLabel { background: #0b1220; color: #d7e0ea; border: 1px solid #334155; padding: 4px; }"
        )
        self._image: np.ndarray | None = None
        self._pixmap: QPixmap | None = None
        self._canvas_size = (2440, 1800)
        self._stitch_points: dict[str, list[list[float]]] = {}
        self._overlap_ranges: dict[str, tuple[float, float]] = {}
        self._drag_key: tuple[str, int] | None = None
        self._display_rect = QRectF()
        self._line_colors = {
            "front_right": QColor("#f97316"),
            "right_behind": QColor("#22c55e"),
            "behind_left": QColor("#38bdf8"),
            "left_front": QColor("#f43f5e"),
        }

    def set_canvas(self, image: np.ndarray | None, width: int, height: int) -> None:
        self._image = None if image is None else image.copy()
        self._canvas_size = (width, height)
        if self._image is None:
            blank = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.rectangle(blank, (0, 0), (width - 1, height - 1), (45, 62, 80), 8)
            self._image = blank
        self._pixmap = cv_to_pixmap(self._image, self.width() - 12, self.height() - 12)
        self.update()

    def set_stitch_points(self, points: dict[str, list[list[float]]]) -> None:
        self._stitch_points = {
            name: [[float(x), float(y)] for x, y in endpoints[:2]]
            for name, endpoints in points.items()
        }
        self.update()

    def set_overlaps(self, overlaps: list[dict[str, Any]]) -> None:
        self._overlap_ranges = {}
        for overlap in overlaps:
            x_range = overlap.get("x_range", [])
            seam_name = str(overlap.get("seam", ""))
            if seam_name and len(x_range) == 2:
                self._overlap_ranges[seam_name] = tuple(
                    sorted((float(x_range[0]), float(x_range[1])))
                )
        self.update()

    def stitch_points(self) -> dict[str, list[list[float]]]:
        return {
            name: [[round(x, 3), round(y, 3)] for x, y in endpoints]
            for name, endpoints in self._stitch_points.items()
        }

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._image is not None:
            self._pixmap = cv_to_pixmap(self._image, self.width() - 12, self.height() - 12)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0b1220"))
        if self._pixmap is None:
            painter.setPen(QColor("#d7e0ea"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())
            painter.end()
            return

        x = (self.width() - self._pixmap.width()) / 2
        y = (self.height() - self._pixmap.height()) / 2
        self._display_rect = QRectF(x, y, self._pixmap.width(), self._pixmap.height())
        painter.drawPixmap(int(x), int(y), self._pixmap)

        painter.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        for name, (overlap_start, overlap_end) in self._overlap_ranges.items():
            start = self._canvas_to_widget(overlap_start, 0.0)
            end = self._canvas_to_widget(
                overlap_end,
                float(self._canvas_size[1]),
            )
            left = min(start.x(), end.x())
            right = max(start.x(), end.x())
            color = self._line_colors.get(name, QColor("#facc15"))
            fill = QColor(color)
            fill.setAlpha(45)
            painter.fillRect(
                QRectF(
                    left,
                    self._display_rect.top(),
                    max(1.0, right - left),
                    self._display_rect.height(),
                ),
                fill,
            )
            painter.setPen(QPen(color, 1, Qt.PenStyle.DashLine))
            painter.drawLine(
                QPointF(left, self._display_rect.top()),
                QPointF(left, self._display_rect.bottom()),
            )
            painter.drawLine(
                QPointF(right, self._display_rect.top()),
                QPointF(right, self._display_rect.bottom()),
            )

        for name, endpoints in self._stitch_points.items():
            if len(endpoints) < 2:
                continue
            color = self._line_colors.get(name, QColor("#facc15"))
            p0 = self._canvas_to_widget(endpoints[0][0], endpoints[0][1])
            p1 = self._canvas_to_widget(endpoints[1][0], endpoints[1][1])
            painter.setPen(QPen(color, 3))
            painter.drawLine(p0, p1)
            painter.setBrush(QBrush(color))
            for index, point in enumerate((p0, p1)):
                painter.setPen(QPen(QColor("#020617"), 2))
                painter.drawEllipse(point, 8, 8)
                painter.setPen(QColor("#ffffff"))
                label = f"{name.split('_')[0][0].upper()}{index + 1}"
                painter.drawText(QRectF(point.x() - 16, point.y() - 24, 32, 14), Qt.AlignmentFlag.AlignCenter, label)
        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        best: tuple[str, int] | None = None
        best_distance = 999999.0
        for name, endpoints in self._stitch_points.items():
            for index, (x, y) in enumerate(endpoints):
                point = self._canvas_to_widget(x, y)
                distance = (point.x() - pos.x()) ** 2 + (point.y() - pos.y()) ** 2
                if distance < best_distance:
                    best_distance = distance
                    best = (name, index)
        if best is not None and best_distance <= 28 * 28:
            self._drag_key = best

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_key is None:
            return
        canvas_point = self._widget_to_canvas(event.position())
        if canvas_point is None:
            return
        name, index = self._drag_key
        x, y = canvas_point
        overlap_range = self._overlap_ranges.get(name)
        if overlap_range is not None:
            x = min(max(x, overlap_range[0]), overlap_range[1])
        self._stitch_points[name][index] = [x, y]
        self.seamPointMoved.emit(name, index, x, y)
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_key = None

    def _canvas_to_widget(self, x: float, y: float) -> QPointF:
        width, height = self._canvas_size
        if self._display_rect.isNull():
            return QPointF(x, y)
        return QPointF(
            self._display_rect.left() + x / width * self._display_rect.width(),
            self._display_rect.top() + y / height * self._display_rect.height(),
        )

    def _widget_to_canvas(self, point: QPointF) -> tuple[float, float] | None:
        width, height = self._canvas_size
        if self._display_rect.isNull():
            return None
        x = (point.x() - self._display_rect.left()) / self._display_rect.width() * width
        y = (point.y() - self._display_rect.top()) / self._display_rect.height() * height
        x = min(max(x, 0.0), float(width - 1))
        y = min(max(y, 0.0), float(height - 1))
        return x, y


class CameraRow:
    """Widgets for one camera configuration row."""

    def __init__(self, camera_key: str, config: dict[str, Any]):
        self.camera_key = camera_key
        self.enabled = QCheckBox()
        self.enabled.setChecked(bool(config.get("enabled", True)))

        self.name = QLineEdit(str(config.get("display_name", camera_key)))

        self.source_type = NoWheelComboBox()
        self.source_type.addItems(SOURCE_TYPES)
        source_type = str(config.get("source_type", "image_dir"))
        self.source_type.setCurrentText(source_type if source_type in SOURCE_TYPES else "image_dir")

        self.source = QLineEdit(str(config.get("source", "")))
        self.browse = QPushButton("Browse")

    def to_config(self) -> dict[str, Any]:
        return {
            "display_name": self.name.text().strip() or self.camera_key,
            "enabled": self.enabled.isChecked(),
            "source_type": self.source_type.currentText(),
            "source": self.source.text().strip(),
        }

    def apply_config(self, config: dict[str, Any]) -> None:
        self.enabled.setChecked(bool(config.get("enabled", True)))
        self.name.setText(str(config.get("display_name", self.camera_key)))
        source_type = str(config.get("source_type", "image_dir"))
        self.source_type.setCurrentText(
            source_type if source_type in SOURCE_TYPES else "image_dir"
        )
        self.source.setText(str(config.get("source", "")))


class MainWindow(QMainWindow):
    """Main studio window with preview, camera config, and calibration workspaces."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepShark View Studio")

        self.input_dir = PROJECT_ROOT / "samples" / "input"
        self.output_dir = PROJECT_ROOT / "samples" / "output"
        self.log_path = PROJECT_ROOT / "logs" / "deep_shark_studio.log"
        self.app_logger = create_application_logger(self.log_path)
        self.calibration_config = load_config("calibration.yaml")
        self.camera_config = load_config("cameras.yaml")
        self._calibration_config_revision = config_revision("calibration.yaml")
        self._camera_config_revision = config_revision("cameras.yaml")
        self.performance_config = self.camera_config.get("performance", {})
        self.stitcher = self.create_stitcher()
        self.stitch_processor = StitchProcessingManager()
        self.qgc_video_sink: FfmpegVideoSink | None = None
        self.qgc_output_active = False
        self.qgc_output_frames_written = 0
        self.qgc_output_last_error = ""
        self.runtime_stitch_config = RuntimeStitchConfig()
        self.runtime_layout_candidate: LayoutRuntimeCandidate | None = None
        self.far_field_layout_candidate: FarFieldLayoutRuntimeCandidate | None = None
        self.runtime_fisheye_intrinsics_source: FisheyeIntrinsicsRuntimeSource | None = None
        self.language = str(self.camera_config.get("language", "en"))

        self.frames: dict[str, np.ndarray] = {}
        self.warped: dict[str, np.ndarray] = {}
        self.canvas: np.ndarray | None = None
        self.layout_tuner_warped: dict[str, np.ndarray] = {}
        self.layout_tuner_valid_masks: dict[str, np.ndarray] = {}
        self.layout_tuner_preview_result: Any | None = None
        self.layout_tuner_candidate_dir: Path | None = None
        self.layout_tuner_source_info: dict[str, Any] = {}
        self.layout_tuner_projection_metadata: dict[str, Any] = {}
        self.layout_tuner_fisheye_intrinsics_source: FisheyeIntrinsicsRuntimeSource | None = None
        self.stream_manager = CameraStreamManager()
        self.stream_snapshots = {}
        self.stream_error_log_counts: dict[str, int] = {}
        self.frame_counts: dict[str, int] = {}
        self.last_tick = time.time()
        self.last_process_time = 0.0
        self.last_preview_time = 0.0
        self.last_health_time = 0.0
        self.last_canvas_render_time = 0.0
        self.last_qgc_output_status_time = 0.0
        self.last_stitch_ms = 0.0
        self.last_frame_signature: tuple[tuple[str, int], ...] = ()
        self.preview_session_id = 0
        self.displayed_stitch_result_id = 0
        self.last_stitch_error_result_id = 0
        self.preview_layout_mode = PreviewLayoutMode.GRID
        self.preview_content_mode = PreviewContentMode.STOPPED
        self.focused_camera_id: str | None = None
        self._applying_view_state = False
        self._syncing_camera_widgets = False
        self._camera_config_dirty = False
        self.last_stitched_raw_preview_time = 0.0
        self.source_contract_log_messages: set[str] = set()
        self.last_stitch_ui_error = ""
        self.last_runtime_stitch_status = ""
        self.last_runtime_warnings: list[str] = []
        self.last_runtime_metrics: dict[str, Any] | None = None
        self.last_runtime_warning_log_text = ""
        self.calibration_session: CalibrationSession | None = None
        self.live_candidate_directory = latest_calibration_candidate(
            topology=str(
                self.calibration_config.get(
                    "stitch_topology",
                    "triple_front_panorama",
                )
            )
        )
        self.live_stitch_mode = (
            "candidate"
            if self.live_candidate_directory is not None
            else "template"
        )
        self._candidate_ui_cache_path: Path | None = None
        self._candidate_ui_cache: dict[str, Any] = {}
        self.b2_candidate_solver_launcher: (
            Callable[[dict[str, Any]], None] | None
        ) = None

        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(15)
        self.preview_timer.timeout.connect(self.update_live_preview)
        self.layout_tuner_preview_timer = QTimer(self)
        self.layout_tuner_preview_timer.setSingleShot(True)
        self.layout_tuner_preview_timer.setInterval(150)
        self.layout_tuner_preview_timer.timeout.connect(
            self.refresh_layout_tuner_preview
        )

        self.camera_rows: dict[str, CameraRow] = {}
        self.camera_views: dict[str, ImageView] = {}
        self.warped_views: dict[str, ImageView] = {}

        self._build_ui()
        self.fit_initial_window_to_available_screen()
        self.refresh_camera_count()
        self.statusBar().showMessage(self.t("Ready"))
        self.log(
            "Application ready: "
            f"topology={self.calibration_config.get('stitch_topology', '')}, "
            f"cameras={self.active_camera_keys()}"
        )
        for validation_error in validate_overlap_seams(
            self.current_stitch_profile()
        ):
            self.log(
                f"Configuration validation: {validation_error}",
                level="ERROR",
            )

    def t(self, text: str, **kwargs: Any) -> str:
        return i18n(self.language, text, **kwargs)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        hint = super().minimumSizeHint()
        return QSize(min(hint.width(), 1500), min(hint.height(), 700))

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.ensure_window_frame_inside_available_screen()

    def fit_initial_window_to_available_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1280, 760)
            return
        available = screen.availableGeometry()
        target_width = min(1500, max(720, int(available.width() * 0.88)))
        target_height = min(900, max(560, int(available.height() * 0.84)))
        target_width = min(target_width, available.width())
        target_height = min(target_height, available.height())
        self.resize(target_width, target_height)
        self.move(
            available.x() + (available.width() - target_width) // 2,
            available.y() + (available.height() - target_height) // 2,
        )

    def ensure_window_frame_inside_available_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        frame = self.frameGeometry()
        if frame.width() > available.width() or frame.height() > available.height():
            width_delta = max(0, frame.width() - available.width() + 16)
            height_delta = max(0, frame.height() - available.height() + 16)
            self.resize(
                max(720, self.width() - width_delta),
                max(560, self.height() - height_delta),
            )
            frame = self.frameGeometry()
        target_x = min(
            max(frame.x(), available.x()),
            available.x() + max(0, available.width() - frame.width()),
        )
        target_y = min(
            max(frame.y(), available.y()),
            available.y() + max(0, available.height() - frame.height()),
        )
        if target_x != frame.x() or target_y != frame.y():
            self.move(
                self.x() + target_x - frame.x(),
                self.y() + target_y - frame.y(),
            )

    def create_stitcher(self) -> SurroundStitcher:
        max_width = int(self.performance_config.get("max_input_width", 960))
        if max_width <= 0:
            max_width = None
        use_intrinsics = bool(self.performance_config.get("use_intrinsics", False))
        return SurroundStitcher(self.calibration_config, max_input_width=max_width, use_intrinsics=use_intrinsics)

    def log_source_coordinate_warnings(
        self,
        frames: dict[str, np.ndarray],
    ) -> None:
        profile = self.current_stitch_profile()
        for key, frame in frames.items():
            if frame is None or key not in profile.get("cameras", {}):
                continue
            height, width = frame.shape[:2]
            diagnostics = source_coordinate_diagnostics(
                profile,
                key,
                (width, height),
            )
            for warning in diagnostics["warnings"]:
                if warning in self.source_contract_log_messages:
                    continue
                self.source_contract_log_messages.add(warning)
                self.log(f"Source coordinate warning: {warning}", level="WARNING")
        self.update_preview_status_summary()

    def current_stitch_profile(self) -> dict[str, Any]:
        return active_stitch_profile(self.calibration_config)

    def live_stitcher_options(self) -> tuple[int | None, bool]:
        max_width = int(self.performance_config.get("max_input_width", 960))
        if max_width <= 0:
            max_width = None
        return max_width, bool(self.performance_config.get("use_intrinsics", False))

    def bump_preview_session(self) -> int:
        self.preview_session_id += 1
        self.displayed_stitch_result_id = 0
        self.last_stitch_error_result_id = 0
        self.last_frame_signature = ()
        return self.preview_session_id

    def configure_live_stitch_processor(self) -> None:
        max_width, use_intrinsics = self.live_stitcher_options()
        processor_mode = self.live_stitch_mode
        runtime_config = self.runtime_stitch_config
        if (
            runtime_config.mode == StitchRuntimeMode.NEAR_FIELD
            or runtime_config.use_far_field_custom_layout
        ):
            processor_mode = "template"
        candidate_directory = (
            str(self.live_candidate_directory)
            if processor_mode == "candidate"
            and self.live_candidate_directory is not None
            else None
        )
        try:
            self.stitch_processor.configure(
                self.preview_session_id,
                self.calibration_config,
                max_width,
                use_intrinsics,
                processor_mode=processor_mode,
                candidate_directory=candidate_directory,
                candidate_use_opencl=self.b2_candidate_opencl_enabled(),
                runtime_config=runtime_config,
            )
        except Exception as exc:
            if processor_mode != "candidate":
                raise
            self.log(
                f"Candidate live processor unavailable; falling back to "
                f"template: {exc}",
                level="ERROR",
            )
            self.live_stitch_mode = "template"
            self.stitch_processor.configure(
                self.preview_session_id,
                self.calibration_config,
                max_width,
                use_intrinsics,
                processor_mode="template",
                runtime_config=RuntimeStitchConfig(),
            )
            self.apply_live_stitch_mode_controls()

    def b2_candidate_opencl_enabled(self) -> bool:
        if hasattr(self, "b2_candidate_opencl"):
            return bool(self.b2_candidate_opencl.isChecked())
        return bool(self.performance_config.get("b2_candidate_opencl", False))

    def current_runtime_profile_id(self) -> str:
        return str(
            self.calibration_config.get(
                "stitch_topology",
                "triple_front_panorama",
            )
        )

    def choose_runtime_layout_candidate(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.t("Choose Layout Candidate V2"),
            str(default_layout_candidate_root()),
            self.t("Layout Candidate (*.yaml);;All Files (*)"),
        )
        if not path:
            return
        try:
            self.load_runtime_layout_candidate_path(Path(path))
        except Exception as exc:
            QMessageBox.warning(
                self,
                self.t("Runtime layout candidate load failed"),
                str(exc),
            )

    def load_runtime_layout_candidate_path(self, path: str | Path) -> LayoutRuntimeCandidate:
        candidate = load_layout_candidate_for_runtime(
            Path(path),
            expected_profile_id=self.current_runtime_profile_id(),
        )
        self.runtime_layout_candidate = candidate
        projection = candidate.projection
        if projection.source == ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE:
            if projection.intrinsics_source_path is None:
                raise LayoutCandidateRuntimeError(
                    "Fisheye Rectilinear candidate is missing intrinsics_source_path."
                )
            self.runtime_fisheye_intrinsics_source = load_fisheye_intrinsics_source(
                projection.intrinsics_source_path
            )
            self.layout_tuner_fisheye_intrinsics_source = self.runtime_fisheye_intrinsics_source
        self.runtime_stitch_config = RuntimeStitchConfig(
            mode=self.runtime_stitch_config.mode,
            projection_source=projection.source,
            layout_candidate_path=candidate.path,
            use_far_field_custom_layout=self.runtime_stitch_config.use_far_field_custom_layout,
            far_field_layout_candidate_path=self.runtime_stitch_config.far_field_layout_candidate_path,
            projection_intrinsics_source_path=projection.intrinsics_source_path,
            fisheye_balance=projection.balance,
            fisheye_fov_scale=projection.fov_scale,
            auto_enabled=self.runtime_stitch_config.auto_enabled,
        )
        self.refresh_stitch_runtime_controls()
        self.update_preview_status_summary()
        self.log(f"Loaded runtime layout candidate: {candidate.path}")
        return candidate

    def clear_runtime_layout_candidate(self) -> None:
        self.runtime_layout_candidate = None
        mode = self.runtime_stitch_config.mode
        if mode == StitchRuntimeMode.NEAR_FIELD:
            mode = StitchRuntimeMode.FAR_FIELD
        self.runtime_stitch_config = RuntimeStitchConfig(
            mode=mode,
            projection_source=ProjectionSource.CURRENT_PERSPECTIVE,
            layout_candidate_path=None,
            use_far_field_custom_layout=self.runtime_stitch_config.use_far_field_custom_layout,
            far_field_layout_candidate_path=self.runtime_stitch_config.far_field_layout_candidate_path,
            projection_intrinsics_source_path=None,
            fisheye_balance=self.runtime_stitch_config.fisheye_balance,
            fisheye_fov_scale=self.runtime_stitch_config.fisheye_fov_scale,
            auto_enabled=False,
        )
        self.refresh_stitch_runtime_controls()
        self.update_preview_status_summary()

    def choose_far_field_layout_candidate(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.t("Choose Far-field Layout Candidate"),
            str(default_far_field_layout_candidate_root()),
            self.t("Layout Candidate (*.yaml);;All Files (*)"),
        )
        if not path:
            return
        try:
            self.load_far_field_layout_candidate_path(Path(path))
        except Exception as exc:
            QMessageBox.warning(
                self,
                self.t("Far-field layout candidate load failed"),
                str(exc),
            )

    def load_far_field_layout_candidate_path(
        self,
        path: str | Path,
    ) -> FarFieldLayoutRuntimeCandidate:
        candidate = load_far_field_layout_candidate(
            Path(path),
            expected_profile_id=self.current_runtime_profile_id(),
        )
        self.far_field_layout_candidate = candidate
        self.runtime_stitch_config = RuntimeStitchConfig(
            mode=self.runtime_stitch_config.mode,
            projection_source=self.runtime_stitch_config.projection_source,
            layout_candidate_path=self.runtime_stitch_config.layout_candidate_path,
            use_far_field_custom_layout=True,
            far_field_layout_candidate_path=candidate.path,
            projection_intrinsics_source_path=self.runtime_stitch_config.projection_intrinsics_source_path,
            fisheye_balance=self.runtime_stitch_config.fisheye_balance,
            fisheye_fov_scale=self.runtime_stitch_config.fisheye_fov_scale,
            auto_enabled=self.runtime_stitch_config.auto_enabled,
        )
        if hasattr(self, "far_field_custom_layout_check"):
            self.far_field_custom_layout_check.setChecked(True)
        self.refresh_stitch_runtime_controls()
        self.update_preview_status_summary()
        self.log(f"Loaded Far-field layout candidate: {candidate.path}")
        return candidate

    def clear_far_field_layout_candidate(self) -> None:
        self.far_field_layout_candidate = None
        self.runtime_stitch_config = RuntimeStitchConfig(
            mode=self.runtime_stitch_config.mode,
            projection_source=self.runtime_stitch_config.projection_source,
            layout_candidate_path=self.runtime_stitch_config.layout_candidate_path,
            use_far_field_custom_layout=False,
            far_field_layout_candidate_path=None,
            projection_intrinsics_source_path=self.runtime_stitch_config.projection_intrinsics_source_path,
            fisheye_balance=self.runtime_stitch_config.fisheye_balance,
            fisheye_fov_scale=self.runtime_stitch_config.fisheye_fov_scale,
            auto_enabled=self.runtime_stitch_config.auto_enabled,
        )
        self.refresh_stitch_runtime_controls()
        self.update_preview_status_summary()

    def choose_runtime_fisheye_intrinsics_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.t("Choose Fisheye Intrinsics Source"),
            str(PROJECT_ROOT / "projects" / "calibration_candidates"),
            self.t("Fisheye source (*.yaml);;All Files (*)"),
        )
        if not path:
            return
        try:
            source = load_fisheye_intrinsics_source(Path(path))
        except Exception as exc:
            QMessageBox.warning(
                self,
                self.t("Fisheye intrinsics source load failed"),
                str(exc),
            )
            return
        self.runtime_fisheye_intrinsics_source = source
        self.layout_tuner_fisheye_intrinsics_source = source
        self.runtime_stitch_config = RuntimeStitchConfig(
            mode=self.runtime_stitch_config.mode,
            projection_source=self.runtime_stitch_config.projection_source,
            layout_candidate_path=self.runtime_stitch_config.layout_candidate_path,
            use_far_field_custom_layout=self.runtime_stitch_config.use_far_field_custom_layout,
            far_field_layout_candidate_path=self.runtime_stitch_config.far_field_layout_candidate_path,
            projection_intrinsics_source_path=source.path,
            fisheye_balance=float(getattr(self, "runtime_fisheye_balance", None).value())
            if hasattr(self, "runtime_fisheye_balance")
            else self.runtime_stitch_config.fisheye_balance,
            fisheye_fov_scale=float(getattr(self, "runtime_fisheye_fov_scale", None).value())
            if hasattr(self, "runtime_fisheye_fov_scale")
            else self.runtime_stitch_config.fisheye_fov_scale,
            auto_enabled=self.runtime_stitch_config.auto_enabled,
        )
        self.refresh_stitch_runtime_controls()
        self.update_preview_status_summary()

    def clear_runtime_fisheye_intrinsics_source(self) -> None:
        self.runtime_fisheye_intrinsics_source = None
        self.layout_tuner_fisheye_intrinsics_source = None
        projection = self.runtime_stitch_config.projection_source
        if projection == ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE:
            projection = ProjectionSource.CURRENT_PERSPECTIVE
        self.runtime_stitch_config = RuntimeStitchConfig(
            mode=self.runtime_stitch_config.mode,
            projection_source=projection,
            layout_candidate_path=self.runtime_stitch_config.layout_candidate_path,
            use_far_field_custom_layout=self.runtime_stitch_config.use_far_field_custom_layout,
            far_field_layout_candidate_path=self.runtime_stitch_config.far_field_layout_candidate_path,
            projection_intrinsics_source_path=None,
            fisheye_balance=self.runtime_stitch_config.fisheye_balance,
            fisheye_fov_scale=self.runtime_stitch_config.fisheye_fov_scale,
            auto_enabled=self.runtime_stitch_config.auto_enabled,
        )
        self.refresh_stitch_runtime_controls()
        self.update_preview_status_summary()

    def open_runtime_layout_candidate_folder(self) -> None:
        if self.runtime_layout_candidate is not None:
            directory = self.runtime_layout_candidate.candidate_directory
        else:
            directory = default_layout_candidate_root()
        directory.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def selected_runtime_mode(self) -> StitchRuntimeMode:
        if not hasattr(self, "runtime_mode_combo"):
            return self.runtime_stitch_config.mode
        return StitchRuntimeMode(str(self.runtime_mode_combo.currentData()))

    def selected_projection_source(self) -> ProjectionSource:
        if not hasattr(self, "runtime_projection_combo"):
            return self.runtime_stitch_config.projection_source
        return ProjectionSource(str(self.runtime_projection_combo.currentData()))

    def selected_runtime_fisheye_params(self) -> tuple[float, float]:
        balance = (
            float(self.runtime_fisheye_balance.value())
            if hasattr(self, "runtime_fisheye_balance")
            else self.runtime_stitch_config.fisheye_balance
        )
        fov_scale = (
            float(self.runtime_fisheye_fov_scale.value())
            if hasattr(self, "runtime_fisheye_fov_scale")
            else self.runtime_stitch_config.fisheye_fov_scale
        )
        return balance, fov_scale

    def selected_layout_tuner_projection_source(self) -> ProjectionSource:
        if not hasattr(self, "layout_tuner_projection_combo"):
            return ProjectionSource.CURRENT_PERSPECTIVE
        return ProjectionSource(str(self.layout_tuner_projection_combo.currentData()))

    def selected_layout_tuner_target(self) -> str:
        if not hasattr(self, "layout_tuner_target_combo"):
            return "near_field"
        return str(self.layout_tuner_target_combo.currentData() or "near_field")

    def selected_layout_tuner_fisheye_params(self) -> tuple[float, float]:
        balance = (
            float(self.layout_tuner_fisheye_balance.value())
            if hasattr(self, "layout_tuner_fisheye_balance")
            else 0.6
        )
        fov_scale = (
            float(self.layout_tuner_fisheye_fov_scale.value())
            if hasattr(self, "layout_tuner_fisheye_fov_scale")
            else 1.0
        )
        return balance, fov_scale

    def fisheye_intrinsics_source_status_text(self, source_path: Path) -> str:
        parts = source_path.parts
        if len(parts) >= 2:
            short_path = str(Path("...", parts[-2], parts[-1]))
        else:
            short_path = source_path.name
        return self.t("Loaded fisheye intrinsics source") + f"\n{short_path}"

    def current_layout_tuner_projection_block(self) -> dict[str, Any] | None:
        source = self.layout_tuner_projection_metadata.get(
            "projection_source",
            self.selected_layout_tuner_projection_source().value,
        )
        if source != ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE.value:
            return None
        intrinsics = self.layout_tuner_fisheye_intrinsics_source or self.runtime_fisheye_intrinsics_source
        if intrinsics is None:
            return None
        selected_balance, selected_fov_scale = self.selected_layout_tuner_fisheye_params()
        balance = float(self.layout_tuner_projection_metadata.get("balance", selected_balance))
        fov_scale = float(
            self.layout_tuner_projection_metadata.get("fov_scale", selected_fov_scale)
        )
        return {
            "source": "fisheye_rectilinear",
            "intrinsics_source_path": str(intrinsics.path),
            "balance": float(balance),
            "fov_scale": float(fov_scale),
            "projection_pipeline": "fisheye_rectilinear_then_template_perspective_warp",
        }

    def project_layout_tuner_frames(self, frames: dict[str, np.ndarray]) -> Any:
        if self.selected_layout_tuner_target() == "far_field":
            candidate_directory = self.current_b2_candidate_directory()
            if candidate_directory is None:
                raise RuntimeError(
                    self.t(
                        "Far-field layout tuning requires a B-2 candidate projection. Generate or load a B-2 candidate first."
                    )
                )
            return B2CandidateProjectionProvider(candidate_directory).project(frames)
        projection_source = self.selected_layout_tuner_projection_source()
        if projection_source == ProjectionSource.CURRENT_PERSPECTIVE:
            return CurrentPerspectiveProjectionProvider(self.stitcher).project(frames)
        if projection_source == ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE:
            source = self.layout_tuner_fisheye_intrinsics_source or self.runtime_fisheye_intrinsics_source
            if source is None:
                raise RuntimeError(
                    self.t("Fisheye Rectilinear projection requires a loaded fisheye intrinsics source.")
                )
            balance, fov_scale = self.selected_layout_tuner_fisheye_params()
            return FisheyeRectilinearProjectionProvider(
                self.stitcher,
                source,
                FisheyeRectilinearParams(balance=balance, fov_scale=fov_scale),
            ).project(frames)
        raise RuntimeError(
            self.t("Projection candidate is research-only and not connected to runtime yet.")
        )

    def current_b2_candidate_directory(self) -> Path | None:
        if self.live_candidate_directory is not None:
            return self.live_candidate_directory
        candidate_directory = latest_calibration_candidate(
            topology=str(
                self.calibration_config.get(
                    "stitch_topology",
                    "triple_front_panorama",
                )
            )
        )
        self.live_candidate_directory = candidate_directory
        return candidate_directory

    def apply_stitch_runtime_config(self) -> None:
        mode = self.selected_runtime_mode()
        projection = (
            self.selected_projection_source()
            if mode == StitchRuntimeMode.NEAR_FIELD
            else ProjectionSource.CURRENT_PERSPECTIVE
        )
        if projection == ProjectionSource.EQUIRECTANGULAR_CANDIDATE:
            QMessageBox.information(
                self,
                self.t("Stitch Runtime Mode"),
                self.t("Projection candidate is research-only and not connected to runtime yet."),
            )
            return
        if (
            projection == ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE
            and self.runtime_fisheye_intrinsics_source is None
        ):
            QMessageBox.information(
                self,
                self.t("Stitch Runtime Mode"),
                self.t("Fisheye Rectilinear projection requires a loaded fisheye intrinsics source."),
            )
            return
        if mode == StitchRuntimeMode.NEAR_FIELD and self.runtime_layout_candidate is None:
            QMessageBox.information(
                self,
                self.t("Stitch Runtime Mode"),
                self.t("Near-field mode requires a loaded Layout Candidate V2."),
            )
            return
        use_far_field_custom = (
            mode == StitchRuntimeMode.FAR_FIELD
            and hasattr(self, "far_field_custom_layout_check")
            and self.far_field_custom_layout_check.isChecked()
        )
        if use_far_field_custom and self.far_field_layout_candidate is None:
            QMessageBox.information(
                self,
                self.t("Stitch Runtime Mode"),
                self.t("Far-field custom layout requires a loaded Far-field Layout Candidate."),
            )
            return
        if mode == StitchRuntimeMode.AUTO:
            QMessageBox.information(
                self,
                self.t("Stitch Runtime Mode"),
                self.t("Auto stitch runtime mode is not implemented; falling back to Far-field."),
            )
        if mode == StitchRuntimeMode.NEAR_FIELD and self.live_stitch_mode == "candidate":
            self.live_stitch_mode = "template"
            self.apply_live_stitch_mode_controls()
            self.log(
                self.t(
                    "Near-field runtime uses current perspective warp; experimental candidate stitch was switched back to template."
                )
            )
        balance, fov_scale = self.selected_runtime_fisheye_params()
        self.runtime_stitch_config = RuntimeStitchConfig(
            mode=mode,
            projection_source=projection,
            layout_candidate_path=(
                self.runtime_layout_candidate.path
                if self.runtime_layout_candidate is not None
                else None
            ),
            use_far_field_custom_layout=use_far_field_custom,
            far_field_layout_candidate_path=(
                self.far_field_layout_candidate.path
                if self.far_field_layout_candidate is not None
                else None
            ),
            projection_intrinsics_source_path=(
                self.runtime_fisheye_intrinsics_source.path
                if self.runtime_fisheye_intrinsics_source is not None
                else None
            ),
            fisheye_balance=balance,
            fisheye_fov_scale=fov_scale,
            auto_enabled=False,
        )
        self.last_runtime_warning_log_text = ""
        self.refresh_stitch_runtime_controls()
        self.update_preview_status_summary()
        if self.preview_content_mode == PreviewContentMode.LIVE:
            self.bump_preview_session()
            try:
                self.configure_live_stitch_processor()
            except Exception as exc:
                QMessageBox.warning(self, self.t("Stitch Runtime Mode"), str(exc))
                return
            self.warped.clear()
            self.canvas = None
            self.last_process_time = 0.0
        elif self.preview_content_mode == PreviewContentMode.STILL and self.frames:
            self._process_and_render_frames("Static preview")
        if self.preview_layout_mode != PreviewLayoutMode.STITCHED:
            self.statusBar().showMessage(
                self.t("Runtime mode applied. Open the Canvas tab to see the effect.")
            )
        else:
            self.statusBar().showMessage(self.t("Stitch Runtime Mode"))

    def refresh_stitch_runtime_controls(self, sync_selection: bool = True) -> None:
        if not hasattr(self, "runtime_candidate_status"):
            return
        if sync_selection and hasattr(self, "runtime_mode_combo"):
            index = self.runtime_mode_combo.findData(self.runtime_stitch_config.mode.value)
            if index >= 0:
                self.runtime_mode_combo.blockSignals(True)
                self.runtime_mode_combo.setCurrentIndex(index)
                self.runtime_mode_combo.blockSignals(False)
        if hasattr(self, "runtime_projection_combo"):
            if sync_selection:
                index = self.runtime_projection_combo.findData(
                    self.runtime_stitch_config.projection_source.value
                )
                if index >= 0:
                    self.runtime_projection_combo.blockSignals(True)
                    self.runtime_projection_combo.setCurrentIndex(index)
                    self.runtime_projection_combo.blockSignals(False)
            near_enabled = self.selected_runtime_mode() == StitchRuntimeMode.NEAR_FIELD
            self.runtime_projection_combo.setEnabled(near_enabled)
            if hasattr(self, "runtime_projection_note"):
                far_custom_enabled = (
                    self.selected_runtime_mode() == StitchRuntimeMode.FAR_FIELD
                    and hasattr(self, "far_field_custom_layout_check")
                    and self.far_field_custom_layout_check.isChecked()
                )
                self.runtime_projection_note.setText(
                    self.t("Near-field Projection Source")
                    if near_enabled
                    else self.t("Far-field Custom uses B-2 candidate projection.")
                    if far_custom_enabled
                    else self.t("Far-field uses current runtime projection.")
                )
        if sync_selection and hasattr(self, "runtime_fisheye_balance"):
            self.runtime_fisheye_balance.blockSignals(True)
            self.runtime_fisheye_balance.setValue(float(self.runtime_stitch_config.fisheye_balance))
            self.runtime_fisheye_balance.blockSignals(False)
        if sync_selection and hasattr(self, "runtime_fisheye_fov_scale"):
            self.runtime_fisheye_fov_scale.blockSignals(True)
            self.runtime_fisheye_fov_scale.setValue(float(self.runtime_stitch_config.fisheye_fov_scale))
            self.runtime_fisheye_fov_scale.blockSignals(False)
        if sync_selection and hasattr(self, "far_field_custom_layout_check"):
            self.far_field_custom_layout_check.blockSignals(True)
            self.far_field_custom_layout_check.setChecked(
                bool(self.runtime_stitch_config.use_far_field_custom_layout)
            )
            self.far_field_custom_layout_check.blockSignals(False)
        if hasattr(self, "far_field_layout_candidate_status"):
            if self.far_field_layout_candidate is None:
                self.far_field_layout_candidate_status.setText(
                    self.t("No Far-field Layout Candidate loaded.")
                )
                self.far_field_layout_candidate_status.setToolTip("")
            else:
                lines = [
                    self.t("Loaded Far-field Layout Candidate"),
                    f"schema_version: {self.far_field_layout_candidate.schema_version}",
                    f"profile_id: {self.far_field_layout_candidate.profile_id}",
                    f"projection: {self.far_field_layout_candidate.projection_source}",
                    f"output: {self.far_field_layout_candidate.output_width_px}x{self.far_field_layout_candidate.output_height_px}",
                ]
                self.far_field_layout_candidate_status.setText("\n".join(lines))
                self.far_field_layout_candidate_status.setToolTip(
                    str(self.far_field_layout_candidate.path)
                )
        if hasattr(self, "runtime_fisheye_source_status"):
            if self.runtime_fisheye_intrinsics_source is None:
                self.runtime_fisheye_source_status.setText(
                    self.t("No fisheye intrinsics source loaded.")
                )
                self.runtime_fisheye_source_status.setToolTip("")
            else:
                text = self.fisheye_intrinsics_source_status_text(
                    self.runtime_fisheye_intrinsics_source.path
                )
                self.runtime_fisheye_source_status.setText(text)
                self.runtime_fisheye_source_status.setToolTip(
                    str(self.runtime_fisheye_intrinsics_source.path)
                )
        if hasattr(self, "layout_tuner_fisheye_source_status"):
            if self.layout_tuner_fisheye_intrinsics_source is None:
                self.layout_tuner_fisheye_source_status.setText(
                    self.t("No fisheye intrinsics source loaded.")
                )
                self.layout_tuner_fisheye_source_status.setToolTip("")
            else:
                text = self.fisheye_intrinsics_source_status_text(
                    self.layout_tuner_fisheye_intrinsics_source.path
                )
                self.layout_tuner_fisheye_source_status.setText(text)
                self.layout_tuner_fisheye_source_status.setToolTip(
                    str(self.layout_tuner_fisheye_intrinsics_source.path)
                )
        if self.runtime_layout_candidate is None:
            self.runtime_candidate_status.setText(self.t("No Layout Candidate V2 loaded."))
            self.runtime_candidate_status.setToolTip("")
            return
        candidate = self.runtime_layout_candidate
        lines = [
            self.t("Loaded Layout Candidate V2"),
            f"path: {candidate.path}",
            f"schema_version: {candidate.schema_version}",
            f"profile_id: {candidate.profile_id}",
            f"projection: {candidate.projection.source.value}",
            (
                "left_pair: "
                f"shift={candidate.left_pair.side_shift_px}, "
                f"side={candidate.left_pair.side_visible_fraction:.2f}, "
                f"feather={candidate.left_pair.feather_width_px}"
            ),
            (
                "right_pair: "
                f"shift={candidate.right_pair.side_shift_px}, "
                f"side={candidate.right_pair.side_visible_fraction:.2f}, "
                f"feather={candidate.right_pair.feather_width_px}"
            ),
            (
                "camera_adjust: "
                f"front scale={candidate.camera_adjust['front'].scale:.2f}, "
                f"x={candidate.camera_adjust['front'].x_offset_px}, "
                f"y={candidate.camera_adjust['front'].y_offset_px}"
            ),
        ]
        if candidate.warnings:
            lines.append("warnings: " + "; ".join(candidate.warnings))
        visible_lines = lines[:6]
        if candidate.warnings:
            visible_lines.append("warnings: " + "; ".join(candidate.warnings))
        self.runtime_candidate_status.setText("\n".join(visible_lines))
        self.runtime_candidate_status.setToolTip("\n".join(lines))

    def process_frames_with_runtime_controller(
        self,
        frames: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], np.ndarray | None, list[str]]:
        controller = RuntimeStitchController(
            self.stitcher,
            self.runtime_stitch_config,
            layout_candidate=self.runtime_layout_candidate,
            far_field_layout_candidate=self.far_field_layout_candidate,
        )
        result = controller.process(frames)
        self.last_runtime_stitch_status = result.status
        self.last_runtime_warnings = list(result.warnings)
        self.last_runtime_metrics = result.metrics
        return result.warped, result.canvas, list(result.warnings)

    def runtime_timing_status_suffix(
        self,
        metrics: dict[str, Any] | None,
    ) -> str:
        timing = metrics.get("timing", {}) if isinstance(metrics, dict) else {}
        if not isinstance(timing, dict):
            return ""
        if "near_field_compositor_ms" in timing:
            suffix = (
                f" | near {float(timing.get('near_field_compositor_ms', 0.0)):.1f} ms"
                f" / warp {float(timing.get('warp_all_ms', 0.0)):.1f} ms"
            )
            if "fisheye_remap_ms" in timing:
                suffix += (
                    f" / remap {float(timing.get('fisheye_remap_ms', 0.0)):.1f} ms"
                    f" / tmpl {float(timing.get('template_warp_ms', 0.0)):.1f} ms"
                )
            return suffix
        if "far_field_custom_total_ms" in timing:
            return (
                f" | far custom {float(timing.get('far_field_custom_total_ms', 0.0)):.1f} ms"
                f" / adjust {float(timing.get('camera_adjust_ms', 0.0)):.1f} ms"
                f" / blend {float(timing.get('far_field_composition_ms', 0.0)):.1f} ms"
            )
        if "candidate_total_ms" in timing:
            suffix = (
                f" | B-2 {float(timing.get('candidate_total_ms', 0.0)):.1f} ms"
                f" / remap {float(timing.get('candidate_remap_ms', 0.0)):.1f} ms"
                f" / compose {float(timing.get('candidate_compose_ms', 0.0)):.1f} ms"
            )
            if timing.get("opencl_enabled"):
                suffix += (
                    f" / dl {float(timing.get('candidate_download_ms', 0.0)):.1f} ms"
                    " / OpenCL"
                )
            return suffix
        if "far_field_total_ms" in timing:
            return f" | far {float(timing.get('far_field_total_ms', 0.0)):.1f} ms"
        return ""

    def frame_signature(self, snapshots: dict[str, Any]) -> tuple[tuple[str, int], ...]:
        return tuple(
            sorted(
                (key, int(snapshot.frame_count))
                for key, snapshot in snapshots.items()
                if snapshot.frame_count > 0
            )
        )

    def _build_ui(self) -> None:
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(8, 8, 8, 8)

        language_bar = QHBoxLayout()
        language_bar.addStretch(1)
        language_bar.addWidget(QLabel(self.t("Language")))
        self.language_combo = NoWheelComboBox()
        self.language_combo.addItem(self.t("English"), "en")
        self.language_combo.addItem(self.t("Chinese"), "zh_CN")
        self.language_combo.setCurrentIndex(1 if self.language == "zh_CN" else 0)
        self.language_combo.currentIndexChanged.connect(self.change_language)
        language_bar.addWidget(self.language_combo)
        central_layout.addLayout(language_bar)

        self.root_tabs = QTabWidget()
        self.root_tabs.addTab(self._build_preview_workspace(), self.t("Realtime Monitor"))
        self.root_tabs.addTab(self._build_layout_tuner_workspace(), self.t("Layout Tuning"))
        self.root_tabs.addTab(self._build_calibration_workspace(), self.t("Calibration & Candidates"))
        self.root_tabs.addTab(self._build_projection_research_workspace(), self.t("Projection Research"))
        self.root_tabs.addTab(self._build_project_management_workspace(), self.t("Project Management"))
        self.root_tabs.addTab(self._build_diagnostics_workspace(), self.t("Diagnostics & Logs"))
        self.root_tabs.currentChanged.connect(self.on_root_tab_changed)
        central_layout.addWidget(self.root_tabs, 1)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

    def change_language(self) -> None:
        language = self.language_combo.currentData()
        if not language or language == self.language:
            return
        previous_language = self.language
        self.collect_camera_config_from_widgets()
        self.language = language
        self.camera_config["language"] = language
        self.backup_before_config_write()
        if not self.persist_camera_config_from_widgets():
            self.language = previous_language
            self.camera_config["language"] = previous_language
            self.language_combo.blockSignals(True)
            self.language_combo.setCurrentIndex(
                1 if previous_language == "zh_CN" else 0
            )
            self.language_combo.blockSignals(False)
            return
        self.stop_live_preview()
        self.camera_rows.clear()
        self.camera_views.clear()
        self.warped_views.clear()
        self._build_ui()
        self.refresh_camera_count()
        self.statusBar().showMessage(self.t("Ready"))

    def _build_preview_workspace(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        source_toolbar = QHBoxLayout()
        view_toolbar = QHBoxLayout()
        self.input_path_label = QLabel(str(self.input_dir))
        self.input_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.input_path_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self.choose_input_dir_button = QPushButton(self.t("Choose Image Directory"))
        self.load_images_button = QPushButton(self.t("Load Images"))
        self.start_live_button = QPushButton(self.t("Start Live"))
        self.stop_live_button = QPushButton(self.t("Stop"))
        self.save_results_button = QPushButton(self.t("Save Results"))
        self.grid_view_button = QPushButton(self.t("Multi-camera View"))
        self.stitched_view_button = QPushButton(self.t("Stitched View"))
        self.back_to_grid_button = QPushButton(self.t("Back to Grid"))
        self.preview_mode_label = QLabel()
        self.preview_mode_label.setStyleSheet(
            "QLabel { color: #334155; font-weight: 600; padding: 0 8px; }"
        )
        self.grid_view_button.setCheckable(True)
        self.stitched_view_button.setCheckable(True)
        self.candidate_stitch_button = QPushButton(
            self.t("B-2 Candidate View [Advanced]")
        )
        self.template_stitch_button = QPushButton(
            self.t("Current Profile Template")
        )
        self.candidate_stitch_button.setCheckable(True)
        self.template_stitch_button.setCheckable(True)
        self.stitch_strategy_group = QButtonGroup(self)
        self.stitch_strategy_group.setExclusive(True)
        self.stitch_strategy_group.addButton(self.candidate_stitch_button)
        self.stitch_strategy_group.addButton(self.template_stitch_button)
        self.stitch_strategy_label = QLabel()
        self.stitch_strategy_label.setStyleSheet(
            "QLabel { color: #9a3412; font-weight: 600; padding: 0 8px; }"
        )
        self.choose_input_dir_button.clicked.connect(self.choose_input_dir)
        self.load_images_button.clicked.connect(self.run_still_preview)
        self.start_live_button.clicked.connect(self.start_live_preview)
        self.stop_live_button.clicked.connect(self.stop_live_preview)
        self.save_results_button.clicked.connect(self.save_results)
        self.grid_view_button.clicked.connect(
            lambda: self.set_preview_layout_mode(PreviewLayoutMode.GRID)
        )
        self.stitched_view_button.clicked.connect(
            lambda: self.set_preview_layout_mode(PreviewLayoutMode.STITCHED)
        )
        self.back_to_grid_button.clicked.connect(
            lambda: self.set_preview_layout_mode(PreviewLayoutMode.GRID)
        )
        self.candidate_stitch_button.clicked.connect(
            lambda: self.set_live_stitch_mode("candidate")
        )
        self.template_stitch_button.clicked.connect(
            lambda: self.set_live_stitch_mode("template")
        )
        source_toolbar.addWidget(self.choose_input_dir_button)
        source_toolbar.addWidget(self.load_images_button)
        source_toolbar.addWidget(self.start_live_button)
        source_toolbar.addWidget(self.stop_live_button)
        source_toolbar.addWidget(self.save_results_button)
        source_toolbar.addStretch(1)
        view_toolbar.addWidget(self.back_to_grid_button)
        view_toolbar.addWidget(self.candidate_stitch_button)
        self.template_stitch_button.setVisible(False)
        view_toolbar.addWidget(self.stitch_strategy_label)
        view_toolbar.addWidget(self.preview_mode_label)
        view_toolbar.addWidget(self.input_path_label, 1)
        root.addLayout(source_toolbar)
        root.addLayout(view_toolbar)
        root.addWidget(self._build_stitch_runtime_mode_panel())
        self.preview_status_summary = QLabel()
        self.preview_status_summary.setWordWrap(True)
        self.preview_status_summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.preview_status_summary.setStyleSheet(
            "QLabel { color: #334155; background: #f8fafc; "
            "border: 1px solid #cbd5e1; padding: 5px 8px; }"
        )
        root.addWidget(self.preview_status_summary)
        self.stitched_view_notice = QLabel()
        self.stitched_view_notice.setWordWrap(True)
        self.stitched_view_notice.setStyleSheet(
            "QLabel { color: #92400e; background: #fffbeb; "
            "border: 1px solid #fcd34d; padding: 5px 8px; }"
        )
        root.addWidget(self.stitched_view_notice)

        self.preview_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.camera_grid = QWidget()
        self.camera_grid_layout = QGridLayout(self.camera_grid)
        self.camera_grid_layout.setContentsMargins(0, 0, 0, 0)
        for key in CAMERA_KEYS:
            view = ImageView(key, camera_id=key)
            view.doubleClicked.connect(self.on_raw_view_double_clicked)
            self.camera_views[key] = view
        self._layout_camera_views(self.camera_views, self.camera_grid_layout)

        self.preview_tabs = QTabWidget()
        self.canvas_view = ImageView(self.t("Stitched surround canvas"))
        self.canvas_view.setMinimumSize(320, 220)
        self.warped_page = QWidget()
        self.warped_grid_layout = QGridLayout(self.warped_page)
        self.warped_grid_layout.setContentsMargins(0, 0, 0, 0)
        for key in CAMERA_KEYS:
            self.warped_views[key] = ImageView(f"{key} warped")
        self._layout_camera_views(self.warped_views, self.warped_grid_layout)
        self.canvas_tab_index = self.preview_tabs.addTab(self.canvas_view, self.t("Canvas"))
        self.warped_tab_index = self.preview_tabs.addTab(self.warped_page, self.t("Warped Views"))
        self.preview_tabs.currentChanged.connect(self.on_preview_tab_changed)

        self.preview_splitter.addWidget(self.camera_grid)
        self.preview_splitter.addWidget(self.preview_tabs)
        self.preview_splitter.setStretchFactor(0, 1)
        self.preview_splitter.setStretchFactor(1, 2)
        root.addWidget(self.preview_splitter, 1)

        self.health_table = QTableWidget(0, 5)
        self.health_table.setHorizontalHeaderLabels([
            self.t("Camera"),
            self.t("Enabled"),
            self.t("Status"),
            self.t("Frames"),
            self.t("Source"),
        ])
        self.health_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.health_table.setMaximumHeight(180)
        root.addWidget(self.health_table)
        self.apply_live_stitch_mode_controls()
        return page

    def _build_layout_tuner_workspace(self) -> QWidget:
        return self._build_layout_tuner_page()

    def _build_projection_research_workspace(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(6, 6, 6, 6)
        overview = QGroupBox(self.t("Projection Research Overview"))
        layout = QVBoxLayout(overview)
        note = QLabel(self.t("Projection Research Note"))
        note.setWordWrap(True)
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(note)
        location_note = QLabel(
            self.t(
                "Runtime projection controls remain in Realtime Monitor for current preview switching; layout-specific projection controls are in Layout Tuning."
            )
        )
        location_note.setWordWrap(True)
        location_note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(location_note)
        root.addWidget(overview)
        root.addStretch(1)
        return page

    def _build_project_management_workspace(self) -> QWidget:
        tabs = QTabWidget()
        tabs.addTab(self._build_camera_config_workspace(), self.t("Camera & Runtime Config"))
        tabs.addTab(self._build_project_workspace(), self.t("Project Files"))
        tabs.addTab(self._build_qgc_video_output_workspace(), self.t("QGC Video Output"))
        return tabs

    def _build_diagnostics_workspace(self) -> QWidget:
        return self._build_log_workspace()

    def _build_stitch_runtime_mode_panel(self) -> QGroupBox:
        panel = StitchRuntimeModePanel(
            self.t,
            combo_box_class=NoWheelComboBox,
            double_spin_box_class=NoWheelDoubleSpinBox,
            parent=self,
        )
        self.stitch_runtime_mode_panel = panel
        self.runtime_mode_combo = panel.runtime_mode_combo
        self.runtime_projection_label = panel.runtime_projection_label
        self.runtime_projection_note = panel.runtime_projection_note
        self.runtime_projection_combo = panel.runtime_projection_combo
        self.runtime_fisheye_balance = panel.runtime_fisheye_balance
        self.runtime_fisheye_fov_scale = panel.runtime_fisheye_fov_scale
        self.runtime_load_fisheye_button = panel.runtime_load_fisheye_button
        self.runtime_clear_fisheye_button = panel.runtime_clear_fisheye_button
        self.runtime_fisheye_source_status = panel.runtime_fisheye_source_status
        self.far_field_custom_layout_check = panel.far_field_custom_layout_check
        self.far_field_layout_candidate_status = panel.far_field_layout_candidate_status
        self.far_field_load_candidate_button = panel.far_field_load_candidate_button
        self.far_field_clear_candidate_button = panel.far_field_clear_candidate_button
        self.runtime_candidate_status = panel.runtime_candidate_status
        self.runtime_load_candidate_button = panel.runtime_load_candidate_button
        self.runtime_clear_candidate_button = panel.runtime_clear_candidate_button
        self.runtime_open_candidate_button = panel.runtime_open_candidate_button
        self.runtime_apply_button = panel.runtime_apply_button
        self.runtime_mode_combo.currentIndexChanged.connect(
            lambda _index: self.refresh_stitch_runtime_controls(sync_selection=False)
        )
        self.runtime_load_candidate_button.clicked.connect(
            self.choose_runtime_layout_candidate
        )
        self.runtime_clear_candidate_button.clicked.connect(
            self.clear_runtime_layout_candidate
        )
        self.runtime_open_candidate_button.clicked.connect(
            self.open_runtime_layout_candidate_folder
        )
        self.runtime_load_fisheye_button.clicked.connect(
            self.choose_runtime_fisheye_intrinsics_source
        )
        self.runtime_clear_fisheye_button.clicked.connect(
            self.clear_runtime_fisheye_intrinsics_source
        )
        self.far_field_load_candidate_button.clicked.connect(
            self.choose_far_field_layout_candidate
        )
        self.far_field_clear_candidate_button.clicked.connect(
            self.clear_far_field_layout_candidate
        )
        self.far_field_custom_layout_check.toggled.connect(
            lambda _checked: self.refresh_stitch_runtime_controls(sync_selection=False)
        )
        self.runtime_apply_button.clicked.connect(self.apply_stitch_runtime_config)
        self.refresh_stitch_runtime_controls()
        return panel

    def _build_layout_tuner_page(self) -> QWidget:
        page = QWidget()
        root = QHBoxLayout(page)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        self.layout_tuner_note = QLabel(self.t("Near-field Layout Tuner Note"))
        self.layout_tuner_note.setWordWrap(True)
        self.layout_tuner_note.setStyleSheet(
            "QLabel { color: #92400e; background: #fffbeb; "
            "border: 1px solid #fcd34d; padding: 6px; }"
        )
        controls_layout.addWidget(self.layout_tuner_note)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel(self.t("Layout Tuner Target")))
        self.layout_tuner_target_combo = NoWheelComboBox()
        self.layout_tuner_target_combo.addItem(
            self.t("Near-field Layout"),
            "near_field",
        )
        self.layout_tuner_target_combo.addItem(
            self.t("Far-field Layout"),
            "far_field",
        )
        self.layout_tuner_target_combo.currentIndexChanged.connect(
            self.on_layout_tuner_target_changed
        )
        target_row.addWidget(self.layout_tuner_target_combo, 1)
        controls_layout.addLayout(target_row)

        self.layout_tuner_projection_group = QGroupBox(self.t("Near-field Projection Source"))
        projection_group = self.layout_tuner_projection_group
        projection_form = QFormLayout(projection_group)
        self.layout_tuner_projection_combo = NoWheelComboBox()
        self.layout_tuner_projection_combo.addItem(
            self.t("Current Perspective"),
            ProjectionSource.CURRENT_PERSPECTIVE.value,
        )
        self.layout_tuner_projection_combo.addItem(
            self.t("Fisheye Rectilinear [Experimental]"),
            ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE.value,
        )
        self.layout_tuner_projection_combo.setToolTip(
            self.t("Fisheye Rectilinear is experimental and only affects Near-field preview/runtime.")
        )
        self.layout_tuner_fisheye_balance = NoWheelDoubleSpinBox()
        self.layout_tuner_fisheye_balance.setRange(0.0, 1.0)
        self.layout_tuner_fisheye_balance.setSingleStep(0.05)
        self.layout_tuner_fisheye_balance.setDecimals(2)
        self.layout_tuner_fisheye_balance.setValue(0.6)
        self.layout_tuner_fisheye_fov_scale = NoWheelDoubleSpinBox()
        self.layout_tuner_fisheye_fov_scale.setRange(0.8, 1.2)
        self.layout_tuner_fisheye_fov_scale.setSingleStep(0.05)
        self.layout_tuner_fisheye_fov_scale.setDecimals(2)
        self.layout_tuner_fisheye_fov_scale.setValue(1.0)
        self.layout_tuner_load_fisheye_button = QPushButton(
            self.t("Load Fisheye Intrinsics Source...")
        )
        self.layout_tuner_clear_fisheye_button = QPushButton(
            self.t("Clear Fisheye Intrinsics Source")
        )
        self.layout_tuner_fisheye_source_status = QLabel(
            self.t("No fisheye intrinsics source loaded.")
        )
        self.layout_tuner_fisheye_source_status.setWordWrap(True)
        self.layout_tuner_fisheye_source_status.setMaximumHeight(48)
        self.layout_tuner_fisheye_source_status.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self.layout_tuner_load_fisheye_button.clicked.connect(
            self.choose_runtime_fisheye_intrinsics_source
        )
        self.layout_tuner_clear_fisheye_button.clicked.connect(
            self.clear_runtime_fisheye_intrinsics_source
        )
        projection_form.addRow(self.t("Projection Source"), self.layout_tuner_projection_combo)
        projection_form.addRow(self.t("Fisheye Balance"), self.layout_tuner_fisheye_balance)
        projection_form.addRow(self.t("Fisheye FOV Scale"), self.layout_tuner_fisheye_fov_scale)
        projection_form.addRow(self.layout_tuner_load_fisheye_button, self.layout_tuner_clear_fisheye_button)
        projection_form.addRow(self.layout_tuner_fisheye_source_status)
        controls_layout.addWidget(projection_group)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel(self.t("Preset")))
        self.layout_tuner_preset = NoWheelComboBox()
        for name in LAYOUT_TUNER_PRESETS_V2:
            self.layout_tuner_preset.addItem(self.t(name), name)
        self.layout_tuner_preset.currentIndexChanged.connect(
            self.on_layout_tuner_preset_changed
        )
        preset_row.addWidget(self.layout_tuner_preset, 1)
        controls_layout.addLayout(preset_row)

        params_group = QGroupBox(self.t("Core Layout"))
        params_form = QFormLayout(params_group)
        self.layout_tuner_output_width = self._layout_tuner_int_control(
            1500,
            2200,
            20,
            1800,
            self.t("Final width crop only; no scaling."),
        )
        params_form.addRow(self.t("Width"), self.layout_tuner_output_width["widget"])
        self.layout_tuner_output_height = self._layout_tuner_int_control(
            520,
            700,
            20,
            700,
            self.t("Final height crop only; no scaling."),
        )
        params_form.addRow(self.t("Height"), self.layout_tuner_output_height["widget"])
        controls_layout.addWidget(params_group)

        self.layout_tuner_pair_group = QGroupBox(self.t("Pair Boundary Params"))
        pair_group = self.layout_tuner_pair_group
        pair_layout = QVBoxLayout(pair_group)
        self.layout_tuner_pair_syncing = False
        self.layout_tuner_lock_pairs = QCheckBox(self.t("Lock Left/Right Pair Params"))
        self.layout_tuner_lock_pairs.setChecked(True)
        self.layout_tuner_lock_pairs.stateChanged.connect(
            self._on_layout_tuner_lock_pairs_changed
        )
        pair_layout.addWidget(self.layout_tuner_lock_pairs)
        pair_form = QFormLayout()
        self.layout_tuner_left_shift = self._layout_tuner_int_control(
            0,
            160,
            5,
            40,
            self.t("Moves this side camera inward after warp."),
        )
        self.layout_tuner_left_fraction = self._layout_tuner_float_control(
            0.10,
            0.50,
            0.01,
            0.30,
            self.t("Allowed side-camera edge fraction."),
        )
        self.layout_tuner_left_feather = self._layout_tuner_int_control(
            0,
            48,
            2,
            24,
            self.t("Narrow seam feather; 0 means hard seam."),
        )
        self.layout_tuner_right_shift = self._layout_tuner_int_control(
            0,
            160,
            5,
            40,
            self.t("Moves this side camera inward after warp."),
        )
        self.layout_tuner_right_fraction = self._layout_tuner_float_control(
            0.10,
            0.50,
            0.01,
            0.30,
            self.t("Allowed side-camera edge fraction."),
        )
        self.layout_tuner_right_feather = self._layout_tuner_int_control(
            0,
            48,
            2,
            24,
            self.t("Narrow seam feather; 0 means hard seam."),
        )
        self.layout_tuner_side_shift = self.layout_tuner_left_shift
        self.layout_tuner_side_fraction = self.layout_tuner_left_fraction
        self.layout_tuner_feather = self.layout_tuner_left_feather
        pair_form.addRow(self.t("Left Side Shift"), self.layout_tuner_left_shift["widget"])
        pair_form.addRow(self.t("Left Side Visible"), self.layout_tuner_left_fraction["widget"])
        pair_form.addRow(self.t("Left Feather"), self.layout_tuner_left_feather["widget"])
        pair_form.addRow(self.t("Right Side Shift"), self.layout_tuner_right_shift["widget"])
        pair_form.addRow(self.t("Right Side Visible"), self.layout_tuner_right_fraction["widget"])
        pair_form.addRow(self.t("Right Feather"), self.layout_tuner_right_feather["widget"])
        pair_layout.addLayout(pair_form)
        controls_layout.addWidget(pair_group)
        self._connect_layout_tuner_pair_sync_controls()

        camera_group = QGroupBox(self.t("Post-warp Camera Adjustment"))
        camera_group.setToolTip(self.t("Preview-only x/y/scale after warp; not real extrinsics calibration."))
        camera_layout = QVBoxLayout(camera_group)
        self.layout_tuner_camera_adjust_syncing = False
        self.layout_tuner_lock_side_scale = QCheckBox(self.t("Lock Side Camera Scale"))
        self.layout_tuner_lock_side_scale.setChecked(False)
        self.layout_tuner_lock_side_scale.stateChanged.connect(
            self._on_layout_tuner_lock_side_scale_changed
        )
        camera_layout.addWidget(self.layout_tuner_lock_side_scale)
        self.layout_tuner_camera_adjust_controls: dict[str, dict[str, dict[str, Any]]] = {}
        for camera in ("front_left", "front", "front_right"):
            form = QFormLayout()
            row = QWidget()
            row_layout = QVBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(QLabel(self.t(camera)))
            camera_controls = {
                "x": self._layout_tuner_int_control(
                    -120,
                    120,
                    2,
                    0,
                    self.t("Horizontal preview offset after warp."),
                ),
                "y": self._layout_tuner_int_control(
                    -80,
                    80,
                    2,
                    0,
                    self.t("Vertical preview offset after warp."),
                ),
                "scale": self._layout_tuner_float_control(
                    0.85,
                    1.15,
                    0.01,
                    1.00,
                    self.t("Preview-only scale around valid-image center."),
                ),
            }
            self.layout_tuner_camera_adjust_controls[camera] = camera_controls
            form.addRow(self.t("X Offset"), camera_controls["x"]["widget"])
            form.addRow(self.t("Y Offset"), camera_controls["y"]["widget"])
            form.addRow(self.t("Scale"), camera_controls["scale"]["widget"])
            row_layout.addLayout(form)
            camera_layout.addWidget(row)
        camera_buttons = QGridLayout()
        self.layout_tuner_reset_camera_adjust_button = QPushButton(self.t("Reset Camera Adjust"))
        self.layout_tuner_reset_front_button = QPushButton(self.t("Reset Front Only"))
        self.layout_tuner_reset_sides_button = QPushButton(self.t("Reset Side Cameras"))
        self.layout_tuner_reset_camera_adjust_button.clicked.connect(
            self.reset_layout_tuner_camera_adjust
        )
        self.layout_tuner_reset_front_button.clicked.connect(
            self.reset_layout_tuner_front_adjust
        )
        self.layout_tuner_reset_sides_button.clicked.connect(
            self.reset_layout_tuner_side_adjust
        )
        camera_buttons.addWidget(self.layout_tuner_reset_camera_adjust_button, 0, 0)
        camera_buttons.addWidget(self.layout_tuner_reset_front_button, 0, 1)
        camera_buttons.addWidget(self.layout_tuner_reset_sides_button, 1, 0, 1, 2)
        camera_layout.addLayout(camera_buttons)
        controls_layout.addWidget(camera_group)
        self._connect_layout_tuner_camera_adjust_controls()

        self.layout_tuner_advanced_group = QGroupBox(self.t("Advanced Vertical Safety"))
        advanced_group = self.layout_tuner_advanced_group
        advanced_form = QFormLayout(advanced_group)
        self.layout_tuner_vertical_safe = self._layout_tuner_float_control(
            0.70,
            1.00,
            0.01,
            1.00,
            self.t("Suppress only side cameras near top/bottom bands."),
        )
        self.layout_tuner_vertical_fade = self._layout_tuner_int_control(
            0,
            96,
            8,
            0,
            self.t("Smooth fade width for side top/bottom suppression."),
        )
        advanced_form.addRow(self.t("Vertical Safe Ratio"), self.layout_tuner_vertical_safe["widget"])
        advanced_form.addRow(self.t("Side Vertical Fade"), self.layout_tuner_vertical_fade["widget"])
        controls_layout.addWidget(advanced_group)

        action_grid = QGridLayout()
        self.layout_tuner_capture_button = QPushButton(self.t("Capture Preview Frame"))
        self.layout_tuner_refresh_button = QPushButton(self.t("Refresh Preview"))
        self.layout_tuner_reset_button = QPushButton(self.t("Reset to Baseline"))
        self.layout_tuner_save_button = QPushButton(
            self.t("Save Near-field Layout Candidate")
        )
        self.layout_tuner_export_button = QPushButton(self.t("Export Preview PNG"))
        self.layout_tuner_open_button = QPushButton(self.t("Open Candidate Folder"))
        self.layout_tuner_capture_button.clicked.connect(
            self.capture_layout_tuner_preview_frame
        )
        self.layout_tuner_refresh_button.clicked.connect(
            self.refresh_layout_tuner_preview
        )
        self.layout_tuner_reset_button.clicked.connect(
            self.reset_layout_tuner_to_baseline
        )
        self.layout_tuner_save_button.clicked.connect(
            self.save_layout_tuner_candidate
        )
        self.layout_tuner_export_button.clicked.connect(
            self.export_layout_tuner_preview_png
        )
        self.layout_tuner_open_button.clicked.connect(
            self.open_layout_tuner_candidate_folder
        )
        action_grid.addWidget(self.layout_tuner_capture_button, 0, 0)
        action_grid.addWidget(self.layout_tuner_refresh_button, 0, 1)
        action_grid.addWidget(self.layout_tuner_reset_button, 1, 0)
        action_grid.addWidget(self.layout_tuner_save_button, 1, 1)
        action_grid.addWidget(self.layout_tuner_export_button, 2, 0)
        action_grid.addWidget(self.layout_tuner_open_button, 2, 1)
        controls_layout.addLayout(action_grid)

        self.layout_tuner_metrics = QTextEdit()
        self.layout_tuner_metrics.setReadOnly(True)
        self.layout_tuner_metrics.setMaximumHeight(170)
        controls_layout.addWidget(self.layout_tuner_metrics)
        controls_layout.addStretch(1)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        controls_scroll.setMinimumWidth(420)
        controls_scroll.setMaximumWidth(450)
        controls_scroll.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Expanding,
        )
        controls_scroll.setWidget(controls)

        self.layout_tuner_view = ImageView(self.t("Capture preview frame to tune layout."))
        self.layout_tuner_view.setMinimumSize(320, 220)
        root.addWidget(controls_scroll)
        root.addWidget(self.layout_tuner_view, 1)
        self.reset_layout_tuner_to_baseline(schedule=False)
        self.on_layout_tuner_target_changed(0)
        return page

    def _layout_tuner_int_control(
        self,
        minimum: int,
        maximum: int,
        step: int,
        value: int,
        tooltip: str,
    ) -> dict[str, Any]:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        slider = NoWheelSlider(Qt.Orientation.Horizontal)
        slider.setRange(int(minimum), int(maximum))
        slider.setSingleStep(int(step))
        slider.setPageStep(int(step))
        slider.setValue(int(value))
        spin = NoWheelSpinBox()
        spin.setRange(int(minimum), int(maximum))
        spin.setSingleStep(int(step))
        spin.setValue(int(value))
        slider.setToolTip(tooltip)
        spin.setToolTip(tooltip)
        slider.valueChanged.connect(
            lambda new_value, s=spin: self._sync_layout_tuner_int(
                s,
                int(new_value),
            )
        )
        spin.valueChanged.connect(
            lambda new_value, s=slider: self._sync_layout_tuner_int(
                s,
                int(new_value),
            )
        )
        layout.addWidget(slider, 1)
        layout.addWidget(spin)
        return {"widget": container, "slider": slider, "spin": spin}

    def _layout_tuner_float_control(
        self,
        minimum: float,
        maximum: float,
        step: float,
        value: float,
        tooltip: str,
    ) -> dict[str, Any]:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        scale = 100
        slider = NoWheelSlider(Qt.Orientation.Horizontal)
        slider.setRange(int(round(minimum * scale)), int(round(maximum * scale)))
        slider.setSingleStep(max(1, int(round(step * scale))))
        slider.setPageStep(max(1, int(round(step * scale * 5))))
        slider.setValue(int(round(value * scale)))
        spin = NoWheelDoubleSpinBox()
        spin.setRange(float(minimum), float(maximum))
        spin.setSingleStep(float(step))
        spin.setDecimals(2)
        spin.setValue(float(value))
        slider.setToolTip(tooltip)
        spin.setToolTip(tooltip)
        slider.valueChanged.connect(
            lambda new_value, s=spin: self._sync_layout_tuner_float(
                s,
                float(new_value) / scale,
            )
        )
        spin.valueChanged.connect(
            lambda new_value, s=slider: self._sync_layout_tuner_slider_float(
                s,
                float(new_value),
                scale,
            )
        )
        layout.addWidget(slider, 1)
        layout.addWidget(spin)
        return {"widget": container, "slider": slider, "spin": spin}

    def _sync_layout_tuner_int(self, target: Any, value: int) -> None:
        target.blockSignals(True)
        target.setValue(int(value))
        target.blockSignals(False)
        self.schedule_layout_tuner_preview()

    def _sync_layout_tuner_float(
        self,
        target: QDoubleSpinBox,
        value: float,
    ) -> None:
        target.blockSignals(True)
        target.setValue(float(value))
        target.blockSignals(False)
        self.schedule_layout_tuner_preview()

    def _sync_layout_tuner_slider_float(
        self,
        target: QSlider,
        value: float,
        scale: int,
    ) -> None:
        target.blockSignals(True)
        target.setValue(int(round(float(value) * scale)))
        target.blockSignals(False)
        self.schedule_layout_tuner_preview()

    def _connect_layout_tuner_pair_sync_controls(self) -> None:
        pairs = [
            ("left", "shift", self.layout_tuner_left_shift),
            ("left", "fraction", self.layout_tuner_left_fraction),
            ("left", "feather", self.layout_tuner_left_feather),
            ("right", "shift", self.layout_tuner_right_shift),
            ("right", "fraction", self.layout_tuner_right_fraction),
            ("right", "feather", self.layout_tuner_right_feather),
        ]
        for side, field, control in pairs:
            control["spin"].valueChanged.connect(
                lambda _value, s=side, f=field: self._on_layout_tuner_pair_changed(s, f)
            )

    def _connect_layout_tuner_camera_adjust_controls(self) -> None:
        for camera, controls in self.layout_tuner_camera_adjust_controls.items():
            controls["scale"]["spin"].valueChanged.connect(
                lambda _value, c=camera: self._on_layout_tuner_camera_adjust_changed(c, "scale")
            )

    def _on_layout_tuner_lock_pairs_changed(self, _state: int) -> None:
        if not self.layout_tuner_lock_pairs.isChecked():
            self.schedule_layout_tuner_preview()
            return
        for field in ("shift", "fraction", "feather"):
            self._on_layout_tuner_pair_changed("left", field)
        self.schedule_layout_tuner_preview()

    def _on_layout_tuner_lock_side_scale_changed(self, _state: int) -> None:
        if not self.layout_tuner_lock_side_scale.isChecked():
            return
        self._on_layout_tuner_camera_adjust_changed("front_left", "scale")

    def _on_layout_tuner_pair_changed(self, side: str, field: str) -> None:
        if self.layout_tuner_pair_syncing:
            return
        if not self.layout_tuner_lock_pairs.isChecked():
            return
        other = "right" if side == "left" else "left"
        source = self._layout_tuner_pair_control(side, field)
        target = self._layout_tuner_pair_control(other, field)
        self.layout_tuner_pair_syncing = True
        try:
            if field == "fraction":
                self._set_layout_tuner_float_control(target, float(source["spin"].value()))
            else:
                self._set_layout_tuner_int_control(target, int(source["spin"].value()))
        finally:
            self.layout_tuner_pair_syncing = False
        self.schedule_layout_tuner_preview()

    def _on_layout_tuner_camera_adjust_changed(self, camera: str, field: str) -> None:
        if field != "scale" or self.layout_tuner_camera_adjust_syncing:
            return
        if not self.layout_tuner_lock_side_scale.isChecked():
            return
        if camera not in {"front_left", "front_right"}:
            return
        other = "front_right" if camera == "front_left" else "front_left"
        source = self.layout_tuner_camera_adjust_controls[camera]["scale"]
        target = self.layout_tuner_camera_adjust_controls[other]["scale"]
        self.layout_tuner_camera_adjust_syncing = True
        try:
            self._set_layout_tuner_float_control(target, float(source["spin"].value()))
        finally:
            self.layout_tuner_camera_adjust_syncing = False
        self.schedule_layout_tuner_preview()

    def _layout_tuner_pair_control(self, side: str, field: str) -> dict[str, Any]:
        return getattr(self, f"layout_tuner_{side}_{field}")

    def _set_layout_tuner_int_control(self, control: dict[str, Any], value: int) -> None:
        for widget in (control["slider"], control["spin"]):
            widget.blockSignals(True)
            widget.setValue(int(value))
            widget.blockSignals(False)

    def _set_layout_tuner_float_control(self, control: dict[str, Any], value: float) -> None:
        control["slider"].blockSignals(True)
        control["spin"].blockSignals(True)
        control["slider"].setValue(int(round(float(value) * 100)))
        control["spin"].setValue(float(value))
        control["spin"].blockSignals(False)
        control["slider"].blockSignals(False)

    def current_layout_tuner_params(self) -> LayoutPreviewParamsV2:
        left_pair = PairLayoutParams(
            side_shift_px=int(self.layout_tuner_left_shift["spin"].value()),
            side_visible_fraction=float(self.layout_tuner_left_fraction["spin"].value()),
            feather_width_px=int(self.layout_tuner_left_feather["spin"].value()),
        )
        if self.layout_tuner_lock_pairs.isChecked():
            right_pair = left_pair
        else:
            right_pair = PairLayoutParams(
                side_shift_px=int(self.layout_tuner_right_shift["spin"].value()),
                side_visible_fraction=float(self.layout_tuner_right_fraction["spin"].value()),
                feather_width_px=int(self.layout_tuner_right_feather["spin"].value()),
            )
        camera_adjust = {}
        for camera, controls in self.layout_tuner_camera_adjust_controls.items():
            camera_adjust[camera] = CameraAdjustParams(
                x_offset_px=int(controls["x"]["spin"].value()),
                y_offset_px=int(controls["y"]["spin"].value()),
                scale=float(controls["scale"]["spin"].value()),
            )
        return LayoutPreviewParamsV2(
            left_pair=left_pair,
            right_pair=right_pair,
            output_width_px=int(self.layout_tuner_output_width["spin"].value()),
            camera_adjust=camera_adjust,
            output_height_px=int(self.layout_tuner_output_height["spin"].value()),
            vertical_safe_ratio=float(self.layout_tuner_vertical_safe["spin"].value()),
            side_vertical_fade_px=int(self.layout_tuner_vertical_fade["spin"].value()),
        )

    def apply_layout_tuner_params(
        self,
        params: LayoutPreviewParamsV2 | LayoutTunerParams,
        schedule: bool = True,
    ) -> None:
        tuner_params = normalize_layout_tuner_params(params)
        controls = [
            (self.layout_tuner_left_shift, int(tuner_params.left_pair.side_shift_px)),
            (self.layout_tuner_left_feather, int(tuner_params.left_pair.feather_width_px)),
            (self.layout_tuner_right_shift, int(tuner_params.right_pair.side_shift_px)),
            (self.layout_tuner_right_feather, int(tuner_params.right_pair.feather_width_px)),
            (self.layout_tuner_output_width, int(tuner_params.output_width_px)),
            (self.layout_tuner_output_height, int(tuner_params.output_height_px)),
            (self.layout_tuner_vertical_fade, int(tuner_params.side_vertical_fade_px)),
        ]
        for control, value in controls:
            self._set_layout_tuner_int_control(control, value)
        float_controls = [
            (self.layout_tuner_left_fraction, float(tuner_params.left_pair.side_visible_fraction)),
            (self.layout_tuner_right_fraction, float(tuner_params.right_pair.side_visible_fraction)),
            (self.layout_tuner_vertical_safe, float(tuner_params.vertical_safe_ratio)),
        ]
        for control, value in float_controls:
            self._set_layout_tuner_float_control(control, value)
        for camera, params_for_camera in (tuner_params.camera_adjust or {}).items():
            controls_for_camera = self.layout_tuner_camera_adjust_controls.get(camera)
            if not controls_for_camera:
                continue
            self._set_layout_tuner_int_control(
                controls_for_camera["x"],
                int(params_for_camera.x_offset_px),
            )
            self._set_layout_tuner_int_control(
                controls_for_camera["y"],
                int(params_for_camera.y_offset_px),
            )
            self._set_layout_tuner_float_control(
                controls_for_camera["scale"],
                float(params_for_camera.scale),
            )
        same_pairs = tuner_params.left_pair == tuner_params.right_pair
        self.layout_tuner_lock_pairs.blockSignals(True)
        self.layout_tuner_lock_pairs.setChecked(bool(same_pairs))
        self.layout_tuner_lock_pairs.blockSignals(False)
        if schedule:
            self.schedule_layout_tuner_preview()

    def on_layout_tuner_preset_changed(self, _index: int) -> None:
        if not hasattr(self, "layout_tuner_preset"):
            return
        name = str(self.layout_tuner_preset.currentData() or "Balanced")
        self.apply_layout_tuner_params(
            LAYOUT_TUNER_PRESETS_V2.get(name, LAYOUT_TUNER_PRESETS_V2["Balanced"])
        )

    def on_layout_tuner_target_changed(self, _index: int) -> None:
        far_field = self.selected_layout_tuner_target() == "far_field"
        if hasattr(self, "layout_tuner_projection_group"):
            self.layout_tuner_projection_group.setVisible(not far_field)
        if hasattr(self, "layout_tuner_pair_group"):
            self.layout_tuner_pair_group.setVisible(not far_field)
        if hasattr(self, "layout_tuner_advanced_group"):
            self.layout_tuner_advanced_group.setVisible(not far_field)
        if hasattr(self, "layout_tuner_save_button"):
            self.layout_tuner_save_button.setText(
                self.t("Save Far-field Layout Candidate")
                if far_field
                else self.t("Save Near-field Layout Candidate")
            )
        if hasattr(self, "layout_tuner_note"):
            self.layout_tuner_note.setText(
                self.t("Far-field Layout Tuner Note")
                if far_field
                else self.t("Near-field Layout Tuner Note")
            )
        self.layout_tuner_preview_result = None
        self.schedule_layout_tuner_preview()

    def reset_layout_tuner_to_baseline(self, schedule: bool = True) -> None:
        if hasattr(self, "layout_tuner_preset"):
            self.layout_tuner_preset.blockSignals(True)
            self.layout_tuner_preset.setCurrentIndex(0)
            self.layout_tuner_preset.blockSignals(False)
        self.apply_layout_tuner_params(
            LAYOUT_TUNER_PRESETS_V2["Balanced"],
            schedule=schedule,
        )

    def reset_layout_tuner_camera_adjust(self) -> None:
        for controls in self.layout_tuner_camera_adjust_controls.values():
            self._set_layout_tuner_int_control(controls["x"], 0)
            self._set_layout_tuner_int_control(controls["y"], 0)
            self._set_layout_tuner_float_control(controls["scale"], 1.0)
        self.schedule_layout_tuner_preview()

    def reset_layout_tuner_front_adjust(self) -> None:
        controls = self.layout_tuner_camera_adjust_controls.get("front")
        if controls:
            self._set_layout_tuner_int_control(controls["x"], 0)
            self._set_layout_tuner_int_control(controls["y"], 0)
            self._set_layout_tuner_float_control(controls["scale"], 1.0)
        self.schedule_layout_tuner_preview()

    def reset_layout_tuner_side_adjust(self) -> None:
        for camera in ("front_left", "front_right"):
            controls = self.layout_tuner_camera_adjust_controls.get(camera)
            if not controls:
                continue
            self._set_layout_tuner_int_control(controls["x"], 0)
            self._set_layout_tuner_int_control(controls["y"], 0)
            self._set_layout_tuner_float_control(controls["scale"], 1.0)
        self.schedule_layout_tuner_preview()

    def schedule_layout_tuner_preview(self) -> None:
        if not hasattr(self, "layout_tuner_preview_timer"):
            return
        self.layout_tuner_preview_timer.start()

    def capture_layout_tuner_preview_frame(self) -> None:
        captured_at = time.time()
        frame_ages: dict[str, float | None] = {}
        active = set(active_topology_camera_keys(self.calibration_config))
        if self.preview_content_mode == PreviewContentMode.LIVE:
            latest_frames, snapshots = self.stream_manager.latest_frames()
            frames = {
                key: frame
                for key, frame in latest_frames.items()
                if key in active and frame is not None
            }
            for key, snapshot in snapshots.items():
                if key in active and snapshot.frame_timestamp > 0:
                    frame_ages[key] = max(0.0, captured_at - snapshot.frame_timestamp)
        else:
            frames = {
                key: frame.copy()
                for key, frame in self.frames.items()
                if key in active and frame is not None
            }
        if not frames and self.warped:
            if self.selected_layout_tuner_target() == "far_field":
                QMessageBox.information(
                    self,
                    self.t("Layout Tuner"),
                    self.t(
                        "Far-field layout tuning requires a B-2 candidate projection. Generate or load a B-2 candidate first."
                    ),
                )
                return
            self.layout_tuner_warped = {
                key: image.copy()
                for key, image in self.warped.items()
                if key in active
            }
            self.layout_tuner_valid_masks = {
                key: np.any(image != 0, axis=2)
                for key, image in self.layout_tuner_warped.items()
            }
            self.layout_tuner_projection_metadata = {
                "projection_source": ProjectionSource.CURRENT_PERSPECTIVE.value,
                "valid_mask_source": "nonzero_pixels_runtime_compatible",
            }
            self.layout_tuner_source_info = {
                "mode": "current_warped_cache",
                "frame_info": {"warning": "Used existing warped cache because no raw frames were available."},
                "projection": dict(self.layout_tuner_projection_metadata),
            }
        else:
            required = {"front_left", "front", "front_right"}
            missing = sorted(required - set(frames))
            if missing:
                QMessageBox.information(
                    self,
                    self.t("Layout Tuner"),
                    self.t(
                        "Missing frames for layout tuner: {cameras}",
                        cameras=", ".join(missing),
                    ),
                )
                return
            try:
                self.log_source_coordinate_warnings(frames)
                projection = self.project_layout_tuner_frames(frames)
                self.layout_tuner_warped = projection.warped_images
                self.layout_tuner_valid_masks = projection.valid_masks
                self.layout_tuner_projection_metadata = projection.metadata
            except Exception as exc:
                self.log(
                    self.t("Layout tuner capture failed: {error}", error=exc)
                    + f"\n{traceback.format_exc()}",
                    level="ERROR",
                )
                QMessageBox.warning(self, self.t("Layout Tuner"), str(exc))
                return
            self.layout_tuner_source_info = {
                "mode": (
                    "live_latest"
                    if self.preview_content_mode == PreviewContentMode.LIVE
                    else "snapshot_preview"
                ),
                "frame_info": {
                    "cameras": sorted(frames),
                    "frame_age_seconds": frame_ages,
                    "warped_keys": sorted(self.layout_tuner_warped),
                    "projection_source": self.layout_tuner_projection_metadata.get(
                        "projection_source",
                        ProjectionSource.CURRENT_PERSPECTIVE.value,
                    ),
                },
                "projection": dict(self.layout_tuner_projection_metadata),
            }
        self.refresh_layout_tuner_preview()

    def refresh_layout_tuner_preview(self) -> None:
        if not hasattr(self, "layout_tuner_view"):
            return
        if not self.layout_tuner_warped:
            self.layout_tuner_view.set_placeholder(
                self.t("Capture preview frame to tune layout.")
            )
            self.layout_tuner_metrics.setPlainText(
                self.t("No cached warped images. Click Capture Preview Frame.")
            )
            return
        start = time.perf_counter()
        try:
            tuner_params = self.current_layout_tuner_params()
            if self.selected_layout_tuner_target() == "far_field":
                projection = ProjectionResult(
                    warped_images=self.layout_tuner_warped,
                    valid_masks=self.layout_tuner_valid_masks or {
                        key: np.any(image != 0, axis=2)
                        for key, image in self.layout_tuner_warped.items()
                    },
                    metadata={
                        **dict(self.layout_tuner_projection_metadata),
                        "projection_source": self.layout_tuner_projection_metadata.get(
                            "projection_source",
                            B2_FAR_FIELD_PROJECTION_SOURCE,
                        ),
                        "provider_name": "LayoutTunerCachedProjection",
                        "valid_mask_source": self.layout_tuner_projection_metadata.get(
                            "valid_mask_source",
                            "layout_tuner_cached_masks",
                        ),
                        "warning_reasons": self.layout_tuner_projection_metadata.get(
                            "warning_reasons",
                            [],
                        ),
                    },
                    timings={"projection_total_ms": 0.0},
                )
                preview_candidate = FarFieldLayoutRuntimeCandidate(
                    path=Path("layout_tuner_preview"),
                    schema_version=1,
                    profile_id=self.current_runtime_profile_id(),
                    camera_adjust=tuner_params.camera_adjust or {},
                    output_width_px=int(tuner_params.output_width_px),
                    output_height_px=int(tuner_params.output_height_px),
                    feather_width_override_px=None,
                    calibration_hash=None,
                    current_calibration_hash=None,
                    warnings=(),
                    raw={},
                )
                result = render_far_field_custom_from_projection(
                    projection,
                    preview_candidate,
                    self.stitcher,
                )
            else:
                result = render_front_priority_layout_preview(
                    self.layout_tuner_warped,
                    layout_tuner_pair_candidates(self.current_stitch_profile()),
                    tuner_params,
                    valid_masks=self.layout_tuner_valid_masks or None,
                )
        except Exception as exc:
            self.log(
                self.t("Layout tuner render failed: {error}", error=exc)
                + f"\n{traceback.format_exc()}",
                level="ERROR",
            )
            self.layout_tuner_view.set_placeholder(str(exc))
            return
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.layout_tuner_preview_result = result
        self.layout_tuner_view.set_image(
            result.image,
            "Far-field layout tuner preview"
            if self.selected_layout_tuner_target() == "far_field"
            else "Front-priority layout tuner preview",
        )
        metrics = result.metrics
        lines = [
            "target: " + self.selected_layout_tuner_target(),
            f"render_ms: {elapsed_ms:.1f}",
            "projection_source: "
            + str(
                self.layout_tuner_projection_metadata.get(
                    "projection_source",
                    ProjectionSource.CURRENT_PERSPECTIVE.value,
                )
            ),
        ]
        if hasattr(result, "layout_id"):
            lines.insert(1, f"layout_id: {result.layout_id}")
        if hasattr(result, "vertical_safety_enabled"):
            lines.append(f"vertical_safety_enabled: {result.vertical_safety_enabled}")
        for key in (
            "front_preserved_ratio",
            "side_visible_ratio",
            "side_invasion_ratio",
            "side_suppressed_ratio",
            "side_suppressed_top_bottom_ratio",
            "black_pixel_ratio",
            "valid_pixel_ratio",
            "duplicate_risk_proxy",
            "duplicate_risk_proxy_after",
        ):
            if key in metrics:
                lines.append(f"{key}: {float(metrics[key]):.4f}")
        warnings = metrics.get("warning_reasons", [])
        if warnings:
            lines.append("warnings: " + ", ".join(str(item) for item in warnings))
        self.layout_tuner_metrics.setPlainText("\n".join(lines))
        self.statusBar().showMessage(
            self.t("Layout tuner preview rendered in {ms:.1f} ms", ms=elapsed_ms)
        )

    def save_layout_tuner_candidate(self) -> None:
        if self.layout_tuner_preview_result is None:
            self.refresh_layout_tuner_preview()
        if self.layout_tuner_preview_result is None:
            QMessageBox.information(
                self,
                self.t("Layout Tuner"),
                self.t("Capture and render a preview before saving."),
            )
            return
        try:
            tuner_params = self.current_layout_tuner_params()
            if self.selected_layout_tuner_target() == "far_field":
                directory = save_far_field_layout_candidate(
                    default_far_field_layout_candidate_root(),
                    str(self.calibration_config.get("stitch_topology", "triple_front_panorama")),
                    tuner_params.camera_adjust or {},
                    int(tuner_params.output_width_px),
                    int(tuner_params.output_height_px),
                    self.layout_tuner_preview_result,
                    source=self.layout_tuner_source_info,
                )
            else:
                directory = save_layout_tuner_candidate(
                    default_layout_candidate_root(),
                    str(self.calibration_config.get("stitch_topology", "triple_front_panorama")),
                    tuner_params,
                    self.layout_tuner_preview_result,
                    source=self.layout_tuner_source_info,
                    projection=self.current_layout_tuner_projection_block(),
                )
        except Exception as exc:
            self.log(
                self.t("Layout tuner candidate save failed: {error}", error=exc)
                + f"\n{traceback.format_exc()}",
                level="ERROR",
            )
            QMessageBox.warning(self, self.t("Layout Tuner"), str(exc))
            return
        self.layout_tuner_candidate_dir = directory
        self.statusBar().showMessage(
            self.t("Layout candidate saved: {path}", path=directory)
        )
        self.log(self.t("Layout candidate saved: {path}", path=directory))

    def export_layout_tuner_preview_png(self) -> None:
        if self.layout_tuner_preview_result is None:
            self.refresh_layout_tuner_preview()
        if self.layout_tuner_preview_result is None:
            QMessageBox.information(
                self,
                self.t("Layout Tuner"),
                self.t("Capture and render a preview before export."),
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.t("Export layout tuner preview"),
            str(PROJECT_ROOT / "layout_tuner_preview.png"),
            "PNG (*.png);;All Files (*)",
        )
        if not path:
            return
        try:
            save_image(path, self.layout_tuner_preview_result.image)
        except Exception as exc:
            QMessageBox.warning(self, self.t("Layout Tuner"), str(exc))
            return
        self.statusBar().showMessage(
            self.t("Layout tuner preview exported: {path}", path=path)
        )

    def open_layout_tuner_candidate_folder(self) -> None:
        directory = self.layout_tuner_candidate_dir or default_layout_candidate_root()
        directory.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def _build_camera_config_workspace(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)

        top = QHBoxLayout()
        top.addWidget(QLabel(self.t("Active camera count")))
        self.camera_count = NoWheelSpinBox()
        topology_keys = [
            key
            for key in active_topology_camera_keys(self.calibration_config)
            if key in CAMERA_KEYS
        ][:4]
        self.camera_count.setRange(1, max(1, len(topology_keys)))
        configured_cameras = self.camera_config.get("cameras", {})
        enabled_count = sum(
            bool(configured_cameras.get(key, {}).get("enabled", True))
            for key in topology_keys
        )
        configured_count = enabled_count or int(
            self.camera_config.get("active_camera_count", len(topology_keys) or 1)
        )
        self.camera_count.setValue(max(1, min(4, configured_count)))
        self.camera_count.valueChanged.connect(self.on_camera_count_changed)
        self.save_camera_config_button = QPushButton(self.t("Save Camera Config"))
        self.reload_config_button = QPushButton(self.t("Reload Config"))
        self.save_camera_config_button.clicked.connect(self.save_camera_config)
        self.reload_config_button.clicked.connect(self.reload_configs)
        top.addWidget(self.camera_count)
        top.addStretch(1)
        top.addWidget(self.reload_config_button)
        top.addWidget(self.save_camera_config_button)
        root.addLayout(top)

        group = QGroupBox(self.t("Camera Sources"))
        grid = QGridLayout(group)
        headers = [self.t("Use"), self.t("Camera"), self.t("Name"), self.t("Type"), self.t("Source"), ""]
        for column, header in enumerate(headers):
            grid.addWidget(QLabel(header), 0, column)

        cameras = self.camera_config.get("cameras", {})
        for row_index, key in enumerate(CAMERA_KEYS, start=1):
            row = CameraRow(key, cameras.get(key, {}))
            row.browse.setText(self.t("Browse"))
            row.browse.clicked.connect(lambda _=False, camera_key=key: self.browse_camera_source(camera_key))
            row.enabled.toggled.connect(self.on_camera_enabled_changed)
            row.name.textChanged.connect(self.mark_camera_config_dirty)
            row.source_type.currentTextChanged.connect(self.mark_camera_config_dirty)
            row.source.textChanged.connect(self.mark_camera_config_dirty)
            self.camera_rows[key] = row
            grid.addWidget(row.enabled, row_index, 0)
            grid.addWidget(QLabel(key), row_index, 1)
            grid.addWidget(row.name, row_index, 2)
            grid.addWidget(row.source_type, row_index, 3)
            grid.addWidget(row.source, row_index, 4)
            grid.addWidget(row.browse, row_index, 5)
        root.addWidget(group)

        output_group = QGroupBox(self.t("Canvas"))
        form = QFormLayout(output_group)
        stitch_profile = self.current_stitch_profile()
        self.canvas_width = NoWheelSpinBox()
        self.canvas_width.setRange(320, 8192)
        self.canvas_width.setValue(int(stitch_profile.get("canvas", {}).get("width", 2440)))
        self.canvas_height = NoWheelSpinBox()
        self.canvas_height.setRange(240, 8192)
        self.canvas_height.setValue(int(stitch_profile.get("canvas", {}).get("height", 1800)))
        self.canvas_width.valueChanged.connect(self.mark_camera_config_dirty)
        self.canvas_height.valueChanged.connect(self.mark_camera_config_dirty)
        form.addRow(self.t("Width"), self.canvas_width)
        form.addRow(self.t("Height"), self.canvas_height)
        root.addWidget(output_group)

        perf_group = QGroupBox(self.t("Performance"))
        perf_form = QFormLayout(perf_group)
        self.process_fps = NoWheelSpinBox()
        self.process_fps.setRange(1, 60)
        self.process_fps.setValue(int(self.performance_config.get("process_fps", 5)))
        self.preview_fps = NoWheelSpinBox()
        self.preview_fps.setRange(1, 60)
        self.preview_fps.setValue(int(self.performance_config.get("preview_fps", 2)))
        self.max_input_width = NoWheelSpinBox()
        self.max_input_width.setRange(0, 4096)
        self.max_input_width.setSpecialValueText(self.t("Original"))
        self.max_input_width.setValue(int(self.performance_config.get("max_input_width", 960)))
        self.refresh_warped_preview = QCheckBox()
        self.refresh_warped_preview.setChecked(bool(self.performance_config.get("refresh_warped_preview", False)))
        self.use_intrinsics_live = QCheckBox()
        self.use_intrinsics_live.setChecked(bool(self.performance_config.get("use_intrinsics", False)))
        self.b2_candidate_opencl = QCheckBox()
        self.b2_candidate_opencl.setChecked(bool(self.performance_config.get("b2_candidate_opencl", False)))
        self.b2_candidate_opencl.setToolTip(
            "实验性：仅加速 B-2 Candidate View / QGC 当前画面输出；不修改 calibration.yaml。"
        )
        self.process_fps.valueChanged.connect(self.mark_camera_config_dirty)
        self.preview_fps.valueChanged.connect(self.mark_camera_config_dirty)
        self.max_input_width.valueChanged.connect(self.mark_camera_config_dirty)
        self.refresh_warped_preview.toggled.connect(self.mark_camera_config_dirty)
        self.use_intrinsics_live.toggled.connect(self.mark_camera_config_dirty)
        self.b2_candidate_opencl.toggled.connect(self.mark_camera_config_dirty)
        perf_form.addRow(self.t("Stitch FPS"), self.process_fps)
        perf_form.addRow(self.t("Preview FPS"), self.preview_fps)
        perf_form.addRow(self.t("Max input width"), self.max_input_width)
        perf_form.addRow(self.t("Refresh warped previews"), self.refresh_warped_preview)
        perf_form.addRow(self.t("Use undistort live"), self.use_intrinsics_live)
        perf_form.addRow("B-2 OpenCL [Experimental]", self.b2_candidate_opencl)
        root.addWidget(perf_group)
        root.addStretch(1)
        return page

    def _build_project_workspace(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)

        group = QGroupBox(self.t("Project Management"))
        layout = QVBoxLayout(group)
        self.project_status = QLabel(self.t("No project file loaded. Current configs are stored in configs/*.yaml."))
        self.project_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.project_status)

        buttons = QHBoxLayout()
        save_project_button = QPushButton(self.t("Save Project As"))
        open_project_button = QPushButton(self.t("Open Project"))
        backup_button = QPushButton(self.t("Backup Current Configs"))
        export_runtime_button = QPushButton(self.t("Export Runtime Config"))
        self.project_package_export_button = QPushButton(self.t("Export Project Package"))
        self.project_package_import_button = QPushButton(self.t("Import Project Package"))
        self.project_package_validate_button = QPushButton(self.t("Validate Project Package"))
        export_runtime_button.setText(
            f"{self.t('Export Runtime Config')} [{self.t('Experimental')}]"
        )
        export_runtime_button.setToolTip(
            "高级/实验性入口：用于未来服务或 QGC 桥接，"
            "不会自动部署或应用到运行设备。"
        )
        save_project_button.clicked.connect(self.save_project_as)
        open_project_button.clicked.connect(self.open_project_file)
        backup_button.clicked.connect(self.backup_current_configs)
        export_runtime_button.clicked.connect(self.export_runtime_config_file)
        self.project_package_export_button.clicked.connect(self.export_project_package_file)
        self.project_package_import_button.clicked.connect(self.import_project_package_file)
        self.project_package_validate_button.clicked.connect(self.validate_project_package_file)
        buttons.addWidget(save_project_button)
        buttons.addWidget(open_project_button)
        buttons.addWidget(backup_button)
        buttons.addWidget(self.project_package_export_button)
        buttons.addWidget(self.project_package_import_button)
        buttons.addWidget(self.project_package_validate_button)
        buttons.addWidget(export_runtime_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        root.addWidget(group)
        notes = QTextEdit()
        notes.setReadOnly(True)
        notes.setText(self.t(
            "Project files bundle calibration.yaml, cameras.yaml, and network.yaml into one portable .dsvs.yaml file.\n\n"
            "Use backups before large calibration or seam edits. Runtime export is the compact configuration intended for a future service/QGC bridge."
        ) + "\n\n" + self.t(
            "Project Package exports configs and active candidates into a folder with package-relative paths. Import validates first; activation backs up current configs before replacing them."
        ))
        root.addWidget(notes, 1)
        return page

    @staticmethod
    def _default_qgc_output_url(kind: str) -> str:
        if kind == "udp_mpegts":
            return "udp://127.0.0.1:5600?pkt_size=1316"
        return "rtsp://127.0.0.1:8554/deepshark"

    def _qgc_video_output_config(self) -> dict[str, Any]:
        raw = self.camera_config.get("qgc_video_output", {})
        if not isinstance(raw, dict):
            raw = {}
        kind = str(raw.get("kind", "udp_mpegts"))
        if kind not in {"rtsp", "udp_mpegts"}:
            kind = "rtsp"
        return {
            "kind": kind,
            "url": str(raw.get("url", self._default_qgc_output_url(kind))),
            "fps": float(raw.get("fps", 15.0)),
            "bitrate": str(raw.get("bitrate", "6000k")),
            "rtsp_transport": str(raw.get("rtsp_transport", "tcp")),
            "ffmpeg_path": str(raw.get("ffmpeg_path", "ffmpeg")),
            "output_width": int(raw.get("output_width", 0) or 0),
            "output_height": int(raw.get("output_height", 0) or 0),
        }

    def _build_qgc_video_output_workspace(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)

        group = QGroupBox(self.t("QGC Video Output"))
        form = QFormLayout(group)
        config = self._qgc_video_output_config()

        self.qgc_output_kind = NoWheelComboBox()
        self.qgc_output_kind.addItem(self.t("UDP Local Low Latency"), "udp_mpegts")
        self.qgc_output_kind.addItem(self.t("RTSP Standard Service"), "rtsp")
        self.qgc_output_kind.setCurrentIndex(
            max(0, self.qgc_output_kind.findData(config["kind"]))
        )
        self.qgc_output_url = QLineEdit(config["url"])
        self.qgc_output_url.setPlaceholderText(
            self._default_qgc_output_url(config["kind"])
        )
        self.qgc_output_url.setToolTip(
            self.t(
                "Use udp://127.0.0.1:5600 for local low-latency bridge, or rtsp://127.0.0.1:8554/deepshark when using MediaMTX/RTSP service."
            )
        )
        self.qgc_output_fps = NoWheelDoubleSpinBox()
        self.qgc_output_fps.setRange(1.0, 60.0)
        self.qgc_output_fps.setSingleStep(1.0)
        self.qgc_output_fps.setDecimals(1)
        self.qgc_output_fps.setValue(float(config["fps"]))
        self.qgc_output_bitrate = QLineEdit(config["bitrate"])
        self.qgc_rtsp_transport = NoWheelComboBox()
        self.qgc_rtsp_transport.addItem("TCP", "tcp")
        self.qgc_rtsp_transport.addItem("UDP", "udp")
        self.qgc_rtsp_transport.setCurrentIndex(
            max(0, self.qgc_rtsp_transport.findData(config["rtsp_transport"]))
        )
        self.qgc_ffmpeg_path = QLineEdit(config["ffmpeg_path"])

        self.qgc_output_kind.currentIndexChanged.connect(
            self.on_qgc_output_kind_changed
        )
        self.qgc_output_kind.currentIndexChanged.connect(self.mark_camera_config_dirty)
        self.qgc_output_url.textChanged.connect(self.mark_camera_config_dirty)
        self.qgc_output_fps.valueChanged.connect(self.mark_camera_config_dirty)
        self.qgc_output_bitrate.textChanged.connect(self.mark_camera_config_dirty)
        self.qgc_rtsp_transport.currentIndexChanged.connect(self.mark_camera_config_dirty)
        self.qgc_ffmpeg_path.textChanged.connect(self.mark_camera_config_dirty)

        form.addRow(self.t("Output Type"), self.qgc_output_kind)
        form.addRow(self.t("Output URL"), self.qgc_output_url)
        form.addRow(self.t("Output FPS"), self.qgc_output_fps)
        form.addRow(self.t("Output Bitrate"), self.qgc_output_bitrate)
        form.addRow(self.t("RTSP Transport"), self.qgc_rtsp_transport)
        form.addRow(self.t("FFmpeg Path"), self.qgc_ffmpeg_path)
        root.addWidget(group)

        note = QLabel(
            self.t(
                "RTSP output publishes to an RTSP server. QGC should open the same stream URL."
            )
            + "\n"
            + self.t(
                "UDP MPEG-TS sends directly to a UDP port and is usually better for same-machine low-latency QGC preview."
            )
            + "\n"
            + self.t(
                "QGC output settings are saved in cameras.yaml and do not modify calibration.yaml."
            )
        )
        note.setWordWrap(True)
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(note)

        save_button = QPushButton(self.t("Save QGC Output Settings"))
        save_button.clicked.connect(self.save_qgc_video_output_settings)
        self.qgc_output_start_button = QPushButton(self.t("Start QGC Output Service"))
        self.qgc_output_start_button.setToolTip(
            self.t(
                "Start streaming the current GUI stitched canvas to QGC."
            )
        )
        self.qgc_output_stop_button = QPushButton(self.t("Stop QGC Output Service"))
        self.qgc_output_stop_button.setToolTip(
            self.t("Stop streaming the current GUI stitched canvas to QGC.")
        )
        self.qgc_output_stop_button.setEnabled(False)
        self.qgc_output_start_button.clicked.connect(self.start_qgc_output_service)
        self.qgc_output_stop_button.clicked.connect(self.stop_qgc_output_service)
        row = QHBoxLayout()
        row.addWidget(save_button)
        row.addWidget(self.qgc_output_start_button)
        row.addWidget(self.qgc_output_stop_button)
        row.addStretch(1)
        root.addLayout(row)
        self.qgc_output_service_status = QLabel(
            self.t("QGC output service is stopped.")
        )
        self.qgc_output_service_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        root.addWidget(self.qgc_output_service_status)
        root.addStretch(1)
        return page

    def _build_log_workspace(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        toolbar = QHBoxLayout()
        copy_button = QPushButton(self.t("Copy Log"))
        clear_button = QPushButton(self.t("Clear View"))
        copy_button.clicked.connect(self.copy_application_log)
        clear_button.clicked.connect(self.clear_application_log_view)
        toolbar.addWidget(copy_button)
        toolbar.addWidget(clear_button)
        toolbar.addWidget(QLabel(self.t("Log file")))
        log_path_label = QLabel(str(self.log_path))
        log_path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        log_path_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        log_path_label.setToolTip(str(self.log_path))
        toolbar.addWidget(log_path_label, 1)
        root.addLayout(toolbar)

        self.application_log = QTextEdit()
        self.application_log.setReadOnly(True)
        self.application_log.setLineWrapMode(
            QTextEdit.LineWrapMode.NoWrap
        )
        if self.log_path.exists():
            existing_lines = self.log_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()[-500:]
            self.application_log.setPlainText("\n".join(existing_lines))
        root.addWidget(self.application_log, 1)
        return page

    def copy_application_log(self) -> None:
        QApplication.clipboard().setText(self.application_log.toPlainText())
        self.statusBar().showMessage(self.t("Copy Log"))

    def clear_application_log_view(self) -> None:
        self.application_log.clear()

    def _build_calibration_workspace(self) -> QWidget:
        self.calibration_scroll_area = QScrollArea()
        self.calibration_scroll_area.setWidgetResizable(True)
        self.calibration_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.calibration_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.calibration_scroll_area.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )
        page = QWidget()
        page.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        root = QVBoxLayout(page)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)
        self.calibration_legacy_header = QWidget()
        calibration_header_layout = QVBoxLayout(
            self.calibration_legacy_header
        )
        calibration_header_layout.setContentsMargins(0, 0, 0, 0)
        calibration_header_layout.setSpacing(6)

        diagnostics_group = QGroupBox(self.t("Topology diagnostics"))
        diagnostics_layout = QVBoxLayout(diagnostics_group)
        self.topology_diagnostics_label = QLabel()
        self.topology_diagnostics_label.setWordWrap(True)
        self.topology_diagnostics_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        diagnostics_layout.addWidget(self.topology_diagnostics_label)
        calibration_header_layout.addWidget(diagnostics_group)

        board_group = QGroupBox(self.t("Chessboard"))
        board_form = QFormLayout(board_group)
        board = self.calibration_config.get("calibration_board", {})
        self.board_columns = NoWheelSpinBox()
        self.board_columns.setRange(3, 40)
        self.board_columns.setValue(int(board.get("total_columns", 12)))
        self.board_rows = NoWheelSpinBox()
        self.board_rows.setRange(3, 40)
        self.board_rows.setValue(int(board.get("total_rows", 9)))
        self.square_size = NoWheelSpinBox()
        self.square_size.setRange(1, 500)
        self.square_size.setValue(int(float(board.get("square_size_mm", 25.0))))
        board_form.addRow(self.t("Total square columns"), self.board_columns)
        board_form.addRow(self.t("Total square rows"), self.board_rows)
        board_form.addRow(self.t("Square size mm"), self.square_size)
        calibration_header_layout.addWidget(board_group)

        calibration_toolbar_primary = QHBoxLayout()
        calibration_toolbar_secondary = QHBoxLayout()
        self.calibration_camera = NoWheelComboBox()
        self.calibration_camera.addItems(active_topology_camera_keys(self.calibration_config))
        load_image = QPushButton(self.t("Load Calibration Image"))
        detect_board = QPushButton(self.t("Detect Chessboard"))
        calibrate_folder = QPushButton(self.t("Calibrate From Folder"))
        undistort_preview = QPushButton(self.t("Preview Undistort"))
        export_report = QPushButton(self.t("Export Quality Report"))
        save_board = QPushButton(self.t("Save Board Settings"))
        load_image.clicked.connect(self.load_calibration_image)
        detect_board.clicked.connect(self.detect_chessboard)
        calibrate_folder.clicked.connect(self.calibrate_from_folder)
        undistort_preview.clicked.connect(self.preview_undistort)
        export_report.clicked.connect(self.export_calibration_report)
        save_board.clicked.connect(self.save_board_settings)
        calibration_toolbar_primary.addWidget(QLabel(self.t("Camera")))
        calibration_toolbar_primary.addWidget(self.calibration_camera)
        calibration_toolbar_primary.addWidget(load_image)
        calibration_toolbar_primary.addWidget(detect_board)
        calibration_toolbar_primary.addWidget(calibrate_folder)
        calibration_toolbar_primary.addStretch(1)
        calibration_toolbar_secondary.addWidget(undistort_preview)
        calibration_toolbar_secondary.addWidget(export_report)
        calibration_toolbar_secondary.addWidget(save_board)
        calibration_toolbar_secondary.addStretch(1)
        calibration_header_layout.addLayout(calibration_toolbar_primary)
        calibration_header_layout.addLayout(calibration_toolbar_secondary)
        root.addWidget(self.calibration_legacy_header)

        self.calibration_tabs = QTabWidget()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.calibration_image_view = PointEditorView()
        stitch_profile = self.current_stitch_profile()
        self.calibration_image_view.set_source_coordinate_contract(
            str(stitch_profile.get("source_coordinate_space", "")),
            stitch_profile.get("source_reference_size"),
        )
        self.calibration_image_view.pointMoved.connect(self.on_source_point_moved)
        splitter.addWidget(self.calibration_image_view)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.point_table = QTableWidget(4, 5)
        self.point_table.setHorizontalHeaderLabels([
            self.t("Index"),
            self.t("Source X"),
            self.t("Source Y"),
            self.t("Target X"),
            self.t("Target Y"),
        ])
        self.point_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        right_layout.addWidget(QLabel(self.t("Perspective points for selected camera")))
        right_layout.addWidget(self.point_table)
        apply_points = QPushButton(self.t("Apply Point Edits To Config"))
        preview_warp = QPushButton(self.t("Preview Warp"))
        apply_points.clicked.connect(self.apply_point_edits)
        preview_warp.clicked.connect(self.preview_calibration_warp)
        right_layout.addWidget(apply_points)
        right_layout.addWidget(preview_warp)
        self.calibration_warp_view = ImageView(self.t("Warp preview"))
        self.calibration_warp_view.setMinimumHeight(120)
        right_layout.addWidget(self.calibration_warp_view)
        self.calibration_log = QTextEdit()
        self.calibration_log.setReadOnly(True)
        right_layout.addWidget(self.calibration_log, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        self.calibration_tabs.addTab(
            splitter,
            f"{self.t('Advanced')} · {self.t('Perspective')}",
        )

        seam_page = QWidget()
        seam_layout = QVBoxLayout(seam_page)
        seam_toolbar = QHBoxLayout()
        refresh_seam = QPushButton(self.t("Refresh Seam Canvas"))
        save_seam = QPushButton(self.t("Save Seam Points"))
        center_seams = QPushButton(self.t("Center Seams in Overlaps"))
        self.feather_width = NoWheelSpinBox()
        self.feather_width.setRange(1, 1000)
        self.feather_width.setValue(self.current_feather_width())
        refresh_seam.clicked.connect(self.refresh_seam_editor)
        save_seam.clicked.connect(self.save_seam_points)
        center_seams.clicked.connect(self.center_seams_in_overlaps)
        seam_toolbar.addWidget(refresh_seam)
        seam_toolbar.addWidget(save_seam)
        seam_toolbar.addWidget(center_seams)
        seam_toolbar.addWidget(QLabel(self.t("Feather width")))
        seam_toolbar.addWidget(self.feather_width)
        seam_toolbar.addStretch(1)
        seam_layout.addLayout(seam_toolbar)
        seam_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.seam_editor = SeamEditorView()
        self.seam_editor.seamPointMoved.connect(self.on_seam_point_moved)
        seam_splitter.addWidget(self.seam_editor)
        seam_right = QWidget()
        seam_right_layout = QVBoxLayout(seam_right)
        self.seam_table = QTableWidget(0, 5)
        self.seam_table.setHorizontalHeaderLabels([self.t("Seam"), self.t("Point"), "X", "Y", self.t("Role")])
        self.seam_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        seam_right_layout.addWidget(QLabel(self.t("Stitch seam endpoints")))
        seam_right_layout.addWidget(self.seam_table, 1)
        seam_splitter.addWidget(seam_right)
        seam_splitter.setStretchFactor(0, 2)
        seam_splitter.setStretchFactor(1, 1)
        seam_layout.addWidget(seam_splitter, 1)

        image_set_group = QGroupBox(self.t("Calibration Image Set"))
        image_set_layout = QVBoxLayout(image_set_group)
        image_set_toolbar = QHBoxLayout()
        self.calibration_folder_label = QLabel(str(PROJECT_ROOT / "samples" / "calibration"))
        self.calibration_folder_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.calibration_folder_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self.calibration_folder_label.setToolTip(
            str(PROJECT_ROOT / "samples" / "calibration")
        )
        choose_calibration_folder = QPushButton(self.t("Choose Folder"))
        capture_frame = QPushButton(self.t("Capture Current Frame"))
        self.save_calibration_snapshot_button = QPushButton(
            self.t("Save Calibration Snapshot")
        )
        scan_folder = QPushButton(self.t("Scan Chessboard"))
        choose_calibration_folder.clicked.connect(self.choose_calibration_folder)
        capture_frame.clicked.connect(self.capture_calibration_frame)
        self.save_calibration_snapshot_button.clicked.connect(
            self.save_current_calibration_snapshot
        )
        scan_folder.clicked.connect(self.scan_calibration_folder)
        image_set_toolbar.addWidget(choose_calibration_folder)
        image_set_toolbar.addWidget(capture_frame)
        image_set_toolbar.addWidget(self.save_calibration_snapshot_button)
        image_set_toolbar.addWidget(scan_folder)
        image_set_toolbar.addWidget(self.calibration_folder_label, 1)
        image_set_layout.addLayout(image_set_toolbar)
        self.calibration_image_table = QTableWidget(0, 5)
        self.calibration_image_table.setHorizontalHeaderLabels([
            self.t("File"),
            self.t("Camera"),
            self.t("Readable"),
            self.t("Chessboard"),
            self.t("Corners"),
        ])
        self.calibration_image_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        image_set_layout.addWidget(self.calibration_image_table)
        seam_layout.addWidget(image_set_group)
        self.calibration_tabs.addTab(
            seam_page,
            f"{self.t('Advanced')} · {self.t('Seams & Image Set')}",
        )

        geometry_page = QWidget()
        geometry_layout = QVBoxLayout(geometry_page)
        geometry_toolbar = QHBoxLayout()
        pairwise_toolbar = QHBoxLayout()
        refresh_geometry = QPushButton(
            self.t("Refresh Geometry Diagnostics")
        )
        self.geometry_pair_mode = NoWheelComboBox()
        self.geometry_pair_mode.addItem(self.t("50% Alpha"), ALPHA_50)
        self.geometry_pair_mode.addItem(self.t("No Feather"), NO_FEATHER)
        self.run_pairwise_button = QPushButton(
            self.t("Run Pairwise Auto Detection")
        )
        self.manual_pairwise_button = QPushButton(
            self.t("Manual Correspondences")
        )
        self.pairwise_pair_combo = NoWheelComboBox()
        refresh_geometry.clicked.connect(self.refresh_geometry_diagnostics)
        self.run_pairwise_button.clicked.connect(
            self.run_pairwise_candidate_diagnostics
        )
        self.manual_pairwise_button.clicked.connect(
            self.open_manual_pairwise_dialog
        )
        geometry_toolbar.addWidget(refresh_geometry)
        geometry_toolbar.addWidget(self.geometry_pair_mode)
        geometry_toolbar.addStretch(1)
        pairwise_toolbar.addWidget(self.run_pairwise_button)
        pairwise_toolbar.addWidget(self.pairwise_pair_combo)
        pairwise_toolbar.addWidget(self.manual_pairwise_button)
        pairwise_toolbar.addStretch(1)
        geometry_layout.addLayout(geometry_toolbar)
        geometry_layout.addLayout(pairwise_toolbar)

        self.diagnostic_output_tabs = QTabWidget()
        warp_output_page = QWidget()
        warp_output_layout = QVBoxLayout(warp_output_page)
        self.geometry_diagnostics_summary = QTextEdit()
        self.geometry_diagnostics_summary.setReadOnly(True)
        self.geometry_diagnostics_summary.setMaximumHeight(160)
        warp_output_layout.addWidget(self.geometry_diagnostics_summary)
        self.geometry_diagnostics_tabs = QTabWidget()
        warp_output_layout.addWidget(self.geometry_diagnostics_tabs, 1)
        self.diagnostic_output_tabs.addTab(
            warp_output_page,
            self.t("Geometry Diagnostics"),
        )

        pairwise_output_page = QWidget()
        pairwise_output_layout = QVBoxLayout(pairwise_output_page)
        self.pairwise_diagnostics_summary = QTextEdit()
        self.pairwise_diagnostics_summary.setReadOnly(True)
        self.pairwise_diagnostics_summary.setMaximumHeight(160)
        pairwise_output_layout.addWidget(self.pairwise_diagnostics_summary)
        self.pairwise_diagnostics_tabs = QTabWidget()
        pairwise_output_layout.addWidget(self.pairwise_diagnostics_tabs, 1)
        self.pairwise_output_tab_index = self.diagnostic_output_tabs.addTab(
            pairwise_output_page,
            self.t("Pairwise Candidates"),
        )
        geometry_layout.addWidget(self.diagnostic_output_tabs, 1)
        self.calibration_tabs.addTab(
            geometry_page,
            f"{self.t('Advanced')}/{self.t('Experimental')} · "
            f"{self.t('Geometry Diagnostics')}",
        )
        session_details_page = self._build_calibration_session_page()
        self.calibration_tabs.addTab(
            session_details_page,
            f"{self.t('Advanced')} · {self.t('Sample Session Details')}",
        )
        self.calibration_tabs.insertTab(
            0,
            self._build_calibration_wizard_page(),
            self.t("Calibration Wizard"),
        )
        self.calibration_tabs.setCurrentIndex(0)
        self.calibration_tabs.currentChanged.connect(
            self.on_calibration_tab_changed
        )
        self.on_calibration_tab_changed(0)

        root.addWidget(self.calibration_tabs, 1)

        self.calibration_camera.currentTextChanged.connect(self.populate_point_table)
        self.point_table.itemChanged.connect(self.on_point_table_changed)
        self.seam_table.itemChanged.connect(self.on_seam_table_changed)
        self.calibration_image: np.ndarray | None = None
        self._updating_point_table = False
        self._updating_seam_table = False
        self.calibration_folder = PROJECT_ROOT / "samples" / "calibration"
        self.pairwise_diagnostic_results: list[
            PairwiseCandidateResult
        ] = []
        self.refresh_pairwise_pair_choices()
        self.populate_point_table()
        self.populate_seam_table()
        self.refresh_seam_editor()
        self.update_topology_diagnostics()
        self.calibration_scroll_area.setWidget(page)
        return self.calibration_scroll_area

    def _build_calibration_session_page(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)

        setup_group = QGroupBox(self.t("Fisheye Sample Session"))
        setup_layout = QFormLayout(setup_group)
        self.session_board_type = NoWheelComboBox()
        self.session_board_type.addItem("Chessboard", "chessboard")
        self.session_board_type.addItem("ArUco Grid", "aruco_grid")
        self.session_board_type.addItem("Charuco", "charuco")
        self.session_board_columns = NoWheelSpinBox()
        self.session_board_columns.setRange(2, 40)
        self.session_board_rows = NoWheelSpinBox()
        self.session_board_rows.setRange(2, 40)
        board = self.calibration_config.get("calibration_board", {})
        self.session_board_columns.setValue(
            max(2, int(board.get("total_columns", 12)) - 1)
        )
        self.session_board_rows.setValue(
            max(2, int(board.get("total_rows", 9)) - 1)
        )
        self.session_square_length = NoWheelDoubleSpinBox()
        self.session_square_length.setRange(0.1, 1000.0)
        self.session_square_length.setDecimals(3)
        self.session_square_length.setSuffix(" mm")
        self.session_square_length.setValue(
            float(board.get("square_size_mm", 25.0))
        )
        self.session_marker_length = NoWheelDoubleSpinBox()
        self.session_marker_length.setRange(0.1, 1000.0)
        self.session_marker_length.setDecimals(3)
        self.session_marker_length.setSuffix(" mm")
        self.session_marker_length.setValue(18.0)
        self.session_marker_separation = NoWheelDoubleSpinBox()
        self.session_marker_separation.setRange(0.0, 1000.0)
        self.session_marker_separation.setDecimals(3)
        self.session_marker_separation.setSuffix(" mm")
        self.session_marker_separation.setValue(5.0)
        self.session_dictionary = NoWheelComboBox()
        self.session_dictionary.addItems(supported_aruco_dictionaries())
        self.session_board_confirmed = QCheckBox(
            self.t("Board definition confirmed")
        )
        self.session_columns_label = QLabel(self.t("Inner corner columns"))
        self.session_rows_label = QLabel(self.t("Inner corner rows"))
        self.session_square_label = QLabel(self.t("Square size"))
        self.session_marker_label = QLabel(self.t("Marker length"))
        self.session_separation_label = QLabel(
            self.t("Marker separation")
        )
        setup_layout.addRow(self.t("Board type"), self.session_board_type)
        setup_layout.addRow(
            self.session_columns_label,
            self.session_board_columns,
        )
        setup_layout.addRow(
            self.session_rows_label,
            self.session_board_rows,
        )
        setup_layout.addRow(
            self.session_square_label,
            self.session_square_length,
        )
        setup_layout.addRow(
            self.session_marker_label,
            self.session_marker_length,
        )
        setup_layout.addRow(
            self.session_separation_label,
            self.session_marker_separation,
        )
        setup_layout.addRow(
            self.t("ArUco dictionary"),
            self.session_dictionary,
        )
        setup_layout.addRow(self.session_board_confirmed)

        quality_group = QGroupBox(self.t("Quality gates"))
        quality_form = QFormLayout(quality_group)
        self.session_min_area_percent = NoWheelDoubleSpinBox()
        self.session_min_area_percent.setRange(0.1, 50.0)
        self.session_min_area_percent.setDecimals(2)
        self.session_min_area_percent.setSuffix(" %")
        self.session_min_area_percent.setValue(1.5)
        self.session_blur_threshold = NoWheelDoubleSpinBox()
        self.session_blur_threshold.setRange(0.0, 100000.0)
        self.session_blur_threshold.setDecimals(1)
        self.session_blur_threshold.setValue(60.0)
        self.session_sync_threshold_ms = NoWheelSpinBox()
        self.session_sync_threshold_ms.setRange(1, 5000)
        self.session_sync_threshold_ms.setSuffix(" ms")
        self.session_sync_threshold_ms.setValue(100)
        quality_form.addRow(
            self.t("Minimum board area"),
            self.session_min_area_percent,
        )
        quality_form.addRow(
            self.t("Minimum Laplacian variance"),
            self.session_blur_threshold,
        )
        quality_form.addRow(
            self.t("Maximum pair time delta"),
            self.session_sync_threshold_ms,
        )

        setup_column = QVBoxLayout()
        setup_column.addWidget(setup_group)
        setup_column.addWidget(quality_group)
        root.addLayout(setup_column)

        session_toolbar = QHBoxLayout()
        self.create_calibration_session_button = QPushButton(
            self.t("Create Session")
        )
        self.load_calibration_session_button = QPushButton(
            self.t("Load Session")
        )
        self.calibration_session_path_label = QLabel(
            self.t("No active session")
        )
        self.calibration_session_path_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self.create_calibration_session_button.clicked.connect(
            self.create_calibration_sample_session
        )
        self.load_calibration_session_button.clicked.connect(
            self.load_calibration_sample_session
        )
        session_toolbar.addWidget(self.create_calibration_session_button)
        session_toolbar.addWidget(self.load_calibration_session_button)
        session_toolbar.addWidget(self.calibration_session_path_label, 1)
        root.addLayout(session_toolbar)

        capture_group = QGroupBox(self.t("Sample capture"))
        capture_layout = QVBoxLayout(capture_group)
        capture_primary = QHBoxLayout()
        capture_board = QHBoxLayout()
        self.session_capture_target = NoWheelComboBox()
        for camera in ("front_left", "front", "front_right"):
            self.session_capture_target.addItem(
                f"{self.t('Intrinsic')}: {camera}",
                f"intrinsic:{camera}",
            )
        for left, right in (
            ("front_left", "front"),
            ("front", "front_right"),
        ):
            self.session_capture_target.addItem(
                f"{self.t('Pair')}: {left} <-> {right}",
                f"pair:{left}__{right}",
            )
        self.session_physical_board_id = QLineEdit("board_1")
        self.session_pair_board_confirmed = QCheckBox(
            self.t("Same physical board confirmed")
        )
        self.capture_calibration_session_button = QPushButton(
            self.t("Capture Sample")
        )
        self.capture_calibration_session_button.clicked.connect(
            self.capture_calibration_session_sample
        )
        self.session_capture_target.currentIndexChanged.connect(
            self.update_session_capture_controls
        )
        capture_primary.addWidget(self.session_capture_target)
        capture_primary.addWidget(self.capture_calibration_session_button)
        capture_primary.addStretch(1)
        capture_board.addWidget(QLabel(self.t("Physical board ID")))
        capture_board.addWidget(self.session_physical_board_id)
        capture_board.addWidget(self.session_pair_board_confirmed)
        capture_board.addStretch(1)
        capture_layout.addLayout(capture_primary)
        capture_layout.addLayout(capture_board)
        root.addWidget(capture_group)

        self.calibration_session_status_table = QTableWidget(0, 6)
        self.calibration_session_status_table.setHorizontalHeaderLabels(
            [
                self.t("Target"),
                self.t("Accepted"),
                self.t("Goal"),
                self.t("Rejected"),
                self.t("Missing coverage"),
                self.t("Last rejection"),
            ]
        )
        self.calibration_session_status_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        root.addWidget(self.calibration_session_status_table, 1)
        self.calibration_session_log = QTextEdit()
        self.calibration_session_log.setReadOnly(True)
        self.calibration_session_log.setMaximumHeight(120)
        root.addWidget(self.calibration_session_log)

        historical = self.calibration_config.get("camera_intrinsics", {})
        reference_lines = [
            "Historical intrinsics are reference-only and are not imported "
            "into this 1920x1080 session."
        ]
        for camera in ("front_left", "front", "front_right"):
            item = historical.get(camera)
            if item:
                reference_lines.append(
                    f"{camera}: {item.get('image_size')}, "
                    f"images={item.get('image_count')}, "
                    f"RMS={float(item.get('rms', 0)):.3f}"
                )
            else:
                reference_lines.append(f"{camera}: no historical intrinsics")
        reference_label = QLabel("\n".join(reference_lines))
        reference_label.setWordWrap(True)
        root.addWidget(reference_label)

        self.session_board_type.currentIndexChanged.connect(
            self.update_session_board_controls
        )
        self.update_session_board_controls()
        self.update_session_capture_controls()
        if self.calibration_session is not None:
            self.apply_calibration_session_to_controls(
                self.calibration_session
            )
        self.refresh_calibration_session_status()
        return page

    def _build_calibration_wizard_page(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        self.wizard_step_title = QLabel()
        self.wizard_step_title.setStyleSheet(
            "QLabel { font-size: 16px; font-weight: 700; color: #1f2937; }"
        )
        root.addWidget(self.wizard_step_title)
        self.calibration_wizard_stack = QStackedWidget()
        self.calibration_wizard_stack.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Ignored,
        )
        self.calibration_wizard_stack.setMinimumHeight(440)
        root.addWidget(self.calibration_wizard_stack, 1)

        prepare_page = QWidget()
        prepare_layout = QVBoxLayout(prepare_page)
        prepare_text = QLabel(
            "开始前请完成以下检查。本步骤只确认现场条件，不会修改相机或运行配置。\n\n"
            "• 固定三台相机，整个采集过程中不要移动支架。\n"
            "• 保持 1920×1080，不改焦距、电子防抖或画面裁切。\n"
            "• 标定板必须平整，避免反光、运动模糊和剧烈光照变化。"
        )
        prepare_text.setWordWrap(True)
        prepare_layout.addWidget(prepare_text)
        self.wizard_camera_fixed_check = QCheckBox(
            "三台相机和支架已经固定"
        )
        self.wizard_resolution_check = QCheckBox(
            "已确认三路原始画面均为 1920×1080"
        )
        self.wizard_board_ready_check = QCheckBox(
            "标定板平整、清晰，现场光线稳定"
        )
        prepare_layout.addWidget(self.wizard_camera_fixed_check)
        prepare_layout.addWidget(self.wizard_resolution_check)
        prepare_layout.addWidget(self.wizard_board_ready_check)
        prepare_layout.addStretch(1)
        self.calibration_wizard_stack.addWidget(prepare_page)

        board_page = QWidget()
        board_layout = QVBoxLayout(board_page)
        board_intro = QLabel(
            "请选择实际使用的标定板并填写实测参数。程序不会根据画面猜测尺寸或字典。"
        )
        board_intro.setWordWrap(True)
        board_layout.addWidget(board_intro)
        self.wizard_board_hint = QLabel(
            "实验室棋盘默认：总格 12×9，对应内角点 11×8，"
            "方格边长 25mm。请按实物核对后再确认。"
        )
        self.wizard_board_hint.setWordWrap(True)
        self.wizard_board_hint.setStyleSheet(
            "QLabel { color: #92400e; padding: 4px; }"
        )
        board_layout.addWidget(self.wizard_board_hint)
        board_form = QFormLayout()
        self.wizard_board_type = NoWheelComboBox()
        self.wizard_board_type.addItem("Chessboard", "chessboard")
        self.wizard_board_type.addItem("ArUco Grid", "aruco_grid")
        self.wizard_board_type.addItem("Charuco", "charuco")
        self.wizard_board_columns = NoWheelSpinBox()
        self.wizard_board_columns.setRange(2, 40)
        self.wizard_board_columns.setValue(11)
        self.wizard_board_rows = NoWheelSpinBox()
        self.wizard_board_rows.setRange(2, 40)
        self.wizard_board_rows.setValue(8)
        self.wizard_square_length = NoWheelDoubleSpinBox()
        self.wizard_square_length.setRange(0.1, 1000.0)
        self.wizard_square_length.setDecimals(3)
        self.wizard_square_length.setSuffix(" mm")
        self.wizard_square_length.setValue(25.0)
        self.wizard_marker_length = NoWheelDoubleSpinBox()
        self.wizard_marker_length.setRange(0.1, 1000.0)
        self.wizard_marker_length.setDecimals(3)
        self.wizard_marker_length.setSuffix(" mm")
        self.wizard_marker_length.setValue(18.0)
        self.wizard_marker_separation = NoWheelDoubleSpinBox()
        self.wizard_marker_separation.setRange(0.0, 1000.0)
        self.wizard_marker_separation.setDecimals(3)
        self.wizard_marker_separation.setSuffix(" mm")
        self.wizard_marker_separation.setValue(5.0)
        self.wizard_dictionary = NoWheelComboBox()
        self.wizard_dictionary.addItems(supported_aruco_dictionaries())
        self.wizard_board_confirmed = QCheckBox(
            "我已核对标定板类型、行列数、尺寸和字典"
        )
        self.wizard_columns_label = QLabel("内角点列数")
        self.wizard_rows_label = QLabel("内角点行数")
        self.wizard_square_label = QLabel("方格边长")
        self.wizard_marker_label = QLabel("Charuco Marker 边长")
        self.wizard_separation_label = QLabel("Marker 间距")
        board_form.addRow("标定板类型", self.wizard_board_type)
        board_form.addRow(
            self.wizard_columns_label,
            self.wizard_board_columns,
        )
        board_form.addRow(
            self.wizard_rows_label,
            self.wizard_board_rows,
        )
        board_form.addRow(
            self.wizard_square_label,
            self.wizard_square_length,
        )
        board_form.addRow(
            self.wizard_marker_label,
            self.wizard_marker_length,
        )
        board_form.addRow(
            self.wizard_separation_label,
            self.wizard_marker_separation,
        )
        board_form.addRow("ArUco 字典", self.wizard_dictionary)
        board_form.addRow(self.wizard_board_confirmed)
        board_layout.addLayout(board_form)
        board_actions = QHBoxLayout()
        self.wizard_create_session_button = QPushButton("创建新采集会话")
        self.wizard_load_session_button = QPushButton("继续已有会话")
        self.wizard_create_session_button.clicked.connect(
            self.wizard_create_session
        )
        self.wizard_load_session_button.clicked.connect(
            self.wizard_load_session
        )
        board_actions.addWidget(self.wizard_create_session_button)
        board_actions.addWidget(self.wizard_load_session_button)
        board_actions.addStretch(1)
        board_layout.addLayout(board_actions)
        self.wizard_session_label = QLabel("尚未创建或加载采集会话")
        self.wizard_session_label.setWordWrap(True)
        board_layout.addWidget(self.wizard_session_label)
        board_layout.addStretch(1)
        self.wizard_board_type.currentIndexChanged.connect(
            self.update_wizard_board_controls
        )
        self.calibration_wizard_stack.addWidget(board_page)

        intrinsic_page = QWidget()
        intrinsic_layout = QVBoxLayout(intrinsic_page)
        intrinsic_intro = QLabel(
            "按 front_left → front → front_right 的顺序采集。一次只需要一台相机看见板。\n"
            "这里采集的是“每台相机自己的镜头畸变参数”。"
            "把板移动到画面中心、四边和四角，同时改变距离和倾斜方向。\n"
            "若样本被拒绝，请按进度框中的“建议”调整后再次主动采集。"
        )
        intrinsic_intro.setWordWrap(True)
        intrinsic_layout.addWidget(intrinsic_intro)
        intrinsic_controls = QHBoxLayout()
        self.wizard_intrinsic_camera = NoWheelComboBox()
        for camera in ("front_left", "front", "front_right"):
            self.wizard_intrinsic_camera.addItem(camera, camera)
        self.wizard_capture_intrinsic_button = QPushButton("采集当前相机样本")
        self.wizard_next_camera_button = QPushButton("下一台相机")
        self.wizard_capture_intrinsic_button.clicked.connect(
            self.wizard_capture_intrinsic
        )
        self.wizard_next_camera_button.clicked.connect(
            self.wizard_advance_intrinsic_camera
        )
        intrinsic_controls.addWidget(self.wizard_intrinsic_camera)
        intrinsic_controls.addWidget(self.wizard_capture_intrinsic_button)
        intrinsic_controls.addWidget(self.wizard_next_camera_button)
        intrinsic_controls.addStretch(1)
        intrinsic_layout.addLayout(intrinsic_controls)
        self.wizard_intrinsic_progress = QTextEdit()
        self.wizard_intrinsic_progress.setReadOnly(True)
        intrinsic_layout.addWidget(self.wizard_intrinsic_progress, 1)
        self.calibration_wizard_stack.addWidget(intrinsic_page)

        pair_page = QWidget()
        pair_layout = QVBoxLayout(pair_page)
        pair_intro = QLabel(
            "先采 front_left ↔ front，再采 front ↔ front_right。\n"
            "Pair 用于求出“相邻两台相机之间的位置和朝向关系”。"
            "每次必须让同一块物理标定板同时出现在所选两路中。两个 pair "
            "可以使用不同标定板，也不要求两块板之间位置固定。\n"
            "采集前让板停稳约 1 秒，并覆盖重叠区域的不同位置、距离和倾角。"
        )
        pair_intro.setWordWrap(True)
        pair_layout.addWidget(pair_intro)
        pair_controls = QHBoxLayout()
        self.wizard_pair = NoWheelComboBox()
        self.wizard_pair.addItem(
            "front_left ↔ front",
            "front_left__front",
        )
        self.wizard_pair.addItem(
            "front ↔ front_right",
            "front__front_right",
        )
        self.wizard_pair.currentIndexChanged.connect(
            self.wizard_inspect_pair_if_visible
        )
        self.wizard_pair_board_id = QLineEdit("board_1")
        self.wizard_pair_board_confirmed = QCheckBox(
            "确认两路看到的是同一块物理板"
        )
        pair_controls.addWidget(self.wizard_pair)
        pair_controls.addWidget(QLabel("标定板编号"))
        pair_controls.addWidget(self.wizard_pair_board_id)
        pair_controls.addWidget(self.wizard_pair_board_confirmed)
        pair_layout.addLayout(pair_controls)
        pair_actions = QHBoxLayout()
        self.wizard_inspect_pair_button = QPushButton(
            self.t("Check Current Pair")
        )
        self.wizard_capture_pair_button = QPushButton("保存这组 Pair 样本")
        self.wizard_next_pair_button = QPushButton("下一个 Pair")
        self.wizard_inspect_pair_button.clicked.connect(
            self.wizard_inspect_pair
        )
        self.wizard_capture_pair_button.clicked.connect(
            self.wizard_capture_pair
        )
        self.wizard_next_pair_button.clicked.connect(
            self.wizard_advance_pair
        )
        pair_actions.addWidget(self.wizard_inspect_pair_button)
        pair_actions.addWidget(self.wizard_capture_pair_button)
        pair_actions.addWidget(self.wizard_next_pair_button)
        pair_actions.addStretch(1)
        pair_layout.addLayout(pair_actions)
        self.wizard_pair_detection_status = QLabel(
            "尚未检查当前两路画面。检查不会保存图片。"
        )
        self.wizard_pair_detection_status.setWordWrap(True)
        pair_layout.addWidget(self.wizard_pair_detection_status)
        self.wizard_pair_progress = QTextEdit()
        self.wizard_pair_progress.setReadOnly(True)
        pair_layout.addWidget(self.wizard_pair_progress, 1)
        self.calibration_wizard_stack.addWidget(pair_page)

        review_page = QWidget()
        review_layout = QVBoxLayout(review_page)
        self.wizard_readiness_label = QLabel()
        self.wizard_readiness_label.setWordWrap(True)
        self.wizard_readiness_label.setStyleSheet(
            "QLabel { font-size: 16px; font-weight: 700; padding: 8px; }"
        )
        review_layout.addWidget(self.wizard_readiness_label)
        self.wizard_b2_mode_label = QLabel()
        self.wizard_b2_mode_label.setWordWrap(True)
        review_layout.addWidget(self.wizard_b2_mode_label)
        self.wizard_review_summary = QTextEdit()
        self.wizard_review_summary.setReadOnly(True)
        review_layout.addWidget(self.wizard_review_summary, 1)
        self.wizard_details_group = QGroupBox("详细信息")
        self.wizard_details_group.setCheckable(True)
        self.wizard_details_group.setChecked(False)
        details_layout = QVBoxLayout(self.wizard_details_group)
        self.wizard_details_text = QTextEdit()
        self.wizard_details_text.setReadOnly(True)
        self.wizard_details_text.setVisible(False)
        self.wizard_details_group.toggled.connect(
            self.wizard_details_text.setVisible
        )
        details_layout.addWidget(self.wizard_details_text)
        review_layout.addWidget(self.wizard_details_group)
        note = QLabel(
            "本阶段只确认样本是否足够。B-2 将读取本 session 生成候选标定，"
            "当前不会修改正式拼接参数。"
        )
        note.setWordWrap(True)
        review_layout.addWidget(note)
        self.calibration_wizard_stack.addWidget(review_page)

        navigation = QHBoxLayout()
        self.wizard_back_button = QPushButton(self.t("Previous"))
        self.wizard_next_button = QPushButton(self.t("Next"))
        self.wizard_back_button.clicked.connect(self.wizard_previous_step)
        self.wizard_next_button.clicked.connect(self.wizard_next_step)
        navigation.addStretch(1)
        navigation.addWidget(self.wizard_back_button)
        navigation.addWidget(self.wizard_next_button)
        root.addLayout(navigation)

        self.update_wizard_board_controls()
        self.set_calibration_wizard_step(0)
        self.refresh_calibration_wizard()
        return page

    def set_preview_layout_mode(
        self,
        mode: PreviewLayoutMode,
        camera_id: str | None = None,
    ) -> None:
        if mode == PreviewLayoutMode.FOCUS:
            target = camera_id or self.focused_camera_id
            if target not in self.active_camera_keys():
                mode = PreviewLayoutMode.GRID
                target = None
            self.focused_camera_id = target
        else:
            self.focused_camera_id = None
        self.preview_layout_mode = mode
        self.apply_preview_view_state()

    def set_preview_content_mode(self, mode: PreviewContentMode) -> None:
        self.preview_content_mode = mode
        self.apply_preview_view_state()

    def on_raw_view_double_clicked(self, camera_id: str) -> None:
        if (
            self.preview_layout_mode == PreviewLayoutMode.FOCUS
            and self.focused_camera_id == camera_id
        ):
            self.set_preview_layout_mode(PreviewLayoutMode.GRID)
            return
        self.set_preview_layout_mode(PreviewLayoutMode.FOCUS, camera_id)

    def on_root_tab_changed(self, index: int) -> None:
        if self._applying_view_state:
            return
        if index == 0:
            self.apply_preview_view_state()

    def on_preview_tab_changed(self, index: int) -> None:
        if self._applying_view_state:
            return
        if (
            self.preview_layout_mode == PreviewLayoutMode.STITCHED
            and index != self.canvas_tab_index
        ):
            self.set_preview_layout_mode(PreviewLayoutMode.GRID)

    def on_calibration_tab_changed(self, index: int) -> None:
        if hasattr(self, "calibration_legacy_header"):
            self.calibration_legacy_header.setVisible(index != 0)
        if hasattr(self, "calibration_tabs"):
            self.calibration_tabs.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                (
                    QSizePolicy.Policy.Ignored
                    if index == 0
                    else QSizePolicy.Policy.Preferred
                ),
            )
            self.calibration_tabs.setMinimumHeight(
                520 if index == 0 else 0
            )
            self.calibration_tabs.setMaximumHeight(
                560 if index == 0 else 16777215
            )
            self.calibration_tabs.updateGeometry()
        if hasattr(self, "calibration_scroll_area"):
            self.calibration_scroll_area.verticalScrollBar().setValue(0)

    def camera_display_name(self, camera_id: str) -> str:
        row = self.camera_rows.get(camera_id)
        if row is not None:
            return row.name.text().strip() or camera_id
        camera = self.camera_config.get("cameras", {}).get(camera_id, {})
        return str(camera.get("display_name", camera_id))

    def preview_mode_text(self) -> str:
        if self.preview_content_mode == PreviewContentMode.STOPPED:
            return self.t("Live preview stopped")
        if self.preview_content_mode == PreviewContentMode.STILL:
            return self.t("Static preview")
        if (
            self.preview_layout_mode == PreviewLayoutMode.FOCUS
            and self.focused_camera_id
        ):
            return self.t(
                "Single-camera view: {camera}",
                camera=self.camera_display_name(self.focused_camera_id),
            )
        if self.preview_layout_mode == PreviewLayoutMode.STITCHED:
            return self.t("Stitched View")
        return self.t("Multi-camera monitoring")

    def current_candidate_metadata(self) -> dict[str, Any]:
        path = self.live_candidate_directory
        if path is None:
            self._candidate_ui_cache_path = None
            self._candidate_ui_cache = {}
            return {}
        resolved = Path(path)
        if self._candidate_ui_cache_path == resolved:
            return self._candidate_ui_cache
        try:
            candidate, _report = load_calibration_candidate(resolved)
        except Exception:
            candidate = {}
        self._candidate_ui_cache_path = resolved
        self._candidate_ui_cache = candidate
        return candidate

    def calibration_candidate_state(self) -> str:
        if self.live_stitch_mode != "candidate":
            origin = self.current_stitch_profile().get(
                "calibration_origin",
                {},
            )
            if any(
                "initial_template" in str(value)
                for value in origin.values()
            ):
                return "initial_template"
            return "applied"
        if self.live_candidate_directory is None:
            return "none"
        candidate = self.current_candidate_metadata()
        if candidate.get("applied"):
            return "applied"
        if candidate.get("recommended"):
            return "recommended"
        if candidate.get("experimental", True):
            return "experimental"
        return "none"

    def preview_warning_messages(self) -> list[str]:
        warnings: list[str] = []
        if self.source_contract_log_messages:
            warnings.append("source 坐标存在警告，请查看日志和拓扑诊断")
        seam_errors = validate_overlap_seams(self.current_stitch_profile())
        if seam_errors:
            warnings.append("当前 profile 的 overlap/seam 校验未通过")
        if self.last_stitch_ui_error:
            warnings.append(f"最近一次拼接失败：{self.last_stitch_ui_error}")
        warnings.extend(self.last_runtime_warnings)
        warnings.extend(self.runtime_stitch_config.runtime_warnings())
        candidate = self.current_candidate_metadata()
        if self.calibration_candidate_state() == "experimental":
            warnings.append("当前候选为 experimental，仅供诊断")
        readiness = candidate.get("readiness", {})
        quality_issues = readiness.get("quality_issues", [])
        if quality_issues:
            warnings.append(f"候选仍有 {len(quality_issues)} 项采样质量风险")
        panorama = candidate.get("virtual_panorama") or {}
        if (
            self.live_stitch_mode == "candidate"
            and "rotation_only" in str(panorama.get("projection", ""))
        ):
            warnings.append("三台相机非共光心：远景优先，近景可能重影")
        if self.calibration_session is not None:
            readiness = self.calibration_session.readiness()
            if readiness["blocking_reasons"]:
                warnings.append("当前采集 session 尚未满足 B-2 最低门槛")
            elif readiness["quality_issues"]:
                warnings.append("当前采集 session 的 Pair/姿态覆盖仍需补充")
        return list(dict.fromkeys(warnings))

    def stitched_view_notice_text(self) -> str:
        if self.preview_content_mode == PreviewContentMode.STOPPED:
            return "实时预览已停止。当前布局偏好已保留，启动后才会继续刷新。"
        if self.preview_content_mode == PreviewContentMode.STILL:
            return "静态拼接画面：用于检查图片，不代表实时流状态。"
        if self.runtime_stitch_config.mode == StitchRuntimeMode.NEAR_FIELD:
            return (
                "近景优先拼接：使用只读 Layout Candidate V2、当前 perspective warp、"
                "front-priority layout。适合近景物体靠近接缝时手动对比。"
            )
        if self.runtime_stitch_config.mode == StitchRuntimeMode.AUTO:
            return "Auto 拼接算法仍是占位；当前回退 Far-field，不会自动切换。"
        if self.live_stitch_mode == "candidate":
            return (
                "实验性候选（rotation-only）：远景优先，近景可能重影。"
                "检查近处人物或物体时，建议切换到多路视图或单路查看。"
            )
        return (
            "初始模板拼接：当前结果不是已正式应用的候选标定。"
            "近景目标建议切换到多路视图或单路查看。"
        )

    def update_preview_status_summary(self) -> None:
        if not hasattr(self, "preview_status_summary"):
            return
        active_keys = self.active_camera_keys()[:4]
        status_counts = {
            STREAM_LIVE: 0,
            STREAM_CONNECTING: 0,
            STREAM_FAILED: 0,
            STREAM_STOPPING: 0,
            STREAM_STOPPED: 0,
        }
        for key in active_keys:
            snapshot = self.stream_snapshots.get(key)
            status = snapshot.status if snapshot is not None else STREAM_STOPPED
            if self.preview_content_mode == PreviewContentMode.STOPPED:
                status = STREAM_STOPPED
            status_counts[status] = status_counts.get(status, 0) + 1
        topology = str(
            self.calibration_config.get("stitch_topology", "未设置")
        )
        layout = {
            PreviewLayoutMode.GRID: "Grid",
            PreviewLayoutMode.FOCUS: (
                f"Focus({self.focused_camera_id or '-'})"
            ),
            PreviewLayoutMode.STITCHED: "Stitched",
        }[self.preview_layout_mode]
        content = {
            PreviewContentMode.STOPPED: "Stopped",
            PreviewContentMode.LIVE: "Live",
            PreviewContentMode.STILL: "Still",
        }[self.preview_content_mode]
        runtime = self.runtime_stitch_config.mode.value
        warnings = self.preview_warning_messages()
        summary = (
            f"Topology: {topology}  |  相机: {len(active_keys)} 路 "
            f"(Live {status_counts[STREAM_LIVE]} / "
            f"Connecting {status_counts[STREAM_CONNECTING]} / "
            f"Failed {status_counts[STREAM_FAILED]} / "
            f"Stopped {status_counts[STREAM_STOPPED]})  |  "
            f"布局: {layout}  |  内容: {content}  |  "
            f"拼接算法: {runtime}  |  "
            f"标定状态: {self.calibration_candidate_state()}"
        )
        if warnings:
            summary += f"  |  警告: {len(warnings)} 项"
        else:
            summary += "  |  警告: 无"
        self.preview_status_summary.setText(summary)
        self.preview_status_summary.setToolTip("\n".join(warnings))
        self.preview_status_summary.setStyleSheet(
            "QLabel { color: #7c2d12; background: #fff7ed; "
            "border: 1px solid #fdba74; padding: 5px 8px; }"
            if warnings
            else
            "QLabel { color: #334155; background: #f8fafc; "
            "border: 1px solid #cbd5e1; padding: 5px 8px; }"
        )

    def canvas_status_text(self) -> str:
        if self.preview_content_mode == PreviewContentMode.LIVE:
            if self.runtime_stitch_config.mode == StitchRuntimeMode.NEAR_FIELD:
                return self.t("Near-field stitched view")
            if self.runtime_stitch_config.mode == StitchRuntimeMode.FAR_FIELD:
                return self.t("Far-field stitched view")
            if self.live_stitch_mode == "candidate":
                return self.t("Experimental candidate live view")
            return self.t("Live stitched view")
        if self.preview_content_mode == PreviewContentMode.STILL:
            return self.t("Static stitched view")
        return self.t("Live preview stopped")

    def render_canvas_view(self) -> None:
        if self.canvas is None or not hasattr(self, "canvas_view"):
            return
        self.canvas_view.set_overlay(self.canvas_status_text())
        self.canvas_view.set_image(self.canvas, self.t("stitched canvas"))

    def apply_live_stitch_mode_controls(self) -> None:
        if not hasattr(self, "candidate_stitch_button"):
            return
        candidate_available = self.live_candidate_directory is not None
        if self.live_stitch_mode == "candidate" and not candidate_available:
            self.live_stitch_mode = "template"
        self.candidate_stitch_button.blockSignals(True)
        self.template_stitch_button.blockSignals(True)
        self.candidate_stitch_button.setEnabled(candidate_available)
        self.candidate_stitch_button.setToolTip(
            (
                "B-2 候选实时视图，仅供高级诊断；不会写入正式 calibration.yaml。"
                if candidate_available
                else "当前 topology 没有可用候选，请先完成采样和 B-2 候选求解。"
            )
        )
        self.template_stitch_button.setToolTip(
            "初始模板/正式 profile 的当前几何配置；不代表候选标定已应用。"
        )
        self.candidate_stitch_button.setChecked(
            self.live_stitch_mode == "candidate"
        )
        self.template_stitch_button.setChecked(
            self.live_stitch_mode == "template"
        )
        self.candidate_stitch_button.blockSignals(False)
        self.template_stitch_button.blockSignals(False)
        if self.live_stitch_mode == "candidate":
            self.stitch_strategy_label.setText(
                self.t("B-2 candidate view")
            )
            self.stitch_strategy_label.setToolTip(
                str(self.live_candidate_directory)
            )
        else:
            self.stitch_strategy_label.setText(
                self.t("Current Profile Template")
            )
            self.stitch_strategy_label.setToolTip("")

    def set_live_stitch_mode(self, mode: str) -> None:
        if mode not in {"candidate", "template"}:
            raise ValueError(f"Unknown live stitch mode: {mode}")
        if mode == "candidate" and self.live_candidate_directory is None:
            QMessageBox.information(
                self,
                self.t("Stitched View"),
                "当前没有可用的三路候选标定。\n\n"
                "可能原因：尚未完成采样、B-2 求解失败，或候选与当前 "
                "topology 不匹配。\n"
                "下一步：使用初始模板查看画面，或到“标定向导”检查进度。",
            )
            self.apply_live_stitch_mode_controls()
            return
        if mode == self.live_stitch_mode:
            self.apply_live_stitch_mode_controls()
            return
        self.live_stitch_mode = mode
        self.apply_live_stitch_mode_controls()
        self.log(
            "Live stitch strategy changed: "
            f"mode={mode}, candidate={self.live_candidate_directory}"
        )
        if self.preview_content_mode != PreviewContentMode.LIVE:
            self.apply_preview_view_state()
            return
        self.bump_preview_session()
        self.configure_live_stitch_processor()
        self.warped.clear()
        self.canvas = None
        self.canvas_view.set_placeholder(self.canvas_status_text())
        for view in self.warped_views.values():
            view.set_placeholder(self.canvas_status_text())
        self.last_process_time = 0.0
        self.apply_preview_view_state()

    def apply_preview_view_state(self) -> None:
        if self._applying_view_state or not hasattr(self, "preview_splitter"):
            return

        self._applying_view_state = True
        try:
            active_keys = self.active_camera_keys()[:4]
            if (
                self.preview_layout_mode == PreviewLayoutMode.FOCUS
                and self.focused_camera_id not in active_keys
            ):
                self.preview_layout_mode = PreviewLayoutMode.GRID
                self.focused_camera_id = None

            focus_key = (
                self.focused_camera_id
                if self.preview_layout_mode == PreviewLayoutMode.FOCUS
                else None
            )
            self._layout_camera_views(
                self.camera_views,
                self.camera_grid_layout,
                active_keys,
                focus_key=focus_key,
            )
            self._layout_camera_views(
                self.warped_views,
                self.warped_grid_layout,
                active_keys,
            )

            for key, view in self.camera_views.items():
                should_show = key in active_keys
                if focus_key is not None:
                    should_show = key == focus_key
                view.setVisible(should_show)
            for key, view in self.warped_views.items():
                view.setVisible(key in active_keys)

            is_focus = self.preview_layout_mode == PreviewLayoutMode.FOCUS
            is_stitched = self.preview_layout_mode == PreviewLayoutMode.STITCHED
            self.camera_grid.setVisible(True)
            self.preview_tabs.setVisible(not is_focus)
            self.grid_view_button.setChecked(
                self.preview_layout_mode == PreviewLayoutMode.GRID
            )
            self.stitched_view_button.setChecked(is_stitched)
            self.back_to_grid_button.setVisible(is_focus)
            self.preview_mode_label.setText(self.preview_mode_text())
            self.apply_live_stitch_mode_controls()
            self.refresh_stitch_runtime_controls()
            self.canvas_view.set_overlay(self.canvas_status_text())
            self.stitched_view_notice.setVisible(is_stitched)
            self.stitched_view_notice.setText(
                self.stitched_view_notice_text()
            )
            self.update_preview_status_summary()

            if is_focus:
                self.preview_splitter.setSizes([1, 0])
            elif is_stitched:
                self.preview_tabs.setCurrentIndex(self.canvas_tab_index)
                self.preview_splitter.setSizes([1, 4])
            else:
                self.preview_splitter.setSizes([1, 2])
        finally:
            self._applying_view_state = False

        self.refresh_raw_preview(force=True)
        if self.preview_content_mode == PreviewContentMode.STOPPED:
            self.canvas_view.set_placeholder(self.t("Live preview stopped"))
        elif self.should_render_canvas() and self.canvas is not None:
            self.render_canvas_view()
        self.refresh_warped_views()

    def _layout_camera_views(
        self,
        views: dict[str, ImageView],
        layout: QGridLayout,
        camera_keys: list[str] | None = None,
        focus_key: str | None = None,
    ) -> None:
        while layout.count():
            layout.takeAt(0)

        keys = (camera_keys or CAMERA_KEYS)[:4]
        if focus_key in keys:
            layout.addWidget(views[focus_key], 0, 0, 2, 2)
            return
        for index, key in enumerate(keys):
            layout.addWidget(views[key], index // 2, index % 2)

    def active_camera_keys(self) -> list[str]:
        topology_keys = [
            key
            for key in active_topology_camera_keys(self.calibration_config)
            if key in CAMERA_KEYS
        ]
        active = topology_keys[:4]
        if self.camera_rows:
            active = [key for key in active if self.camera_rows[key].enabled.isChecked()]
        return active

    def refresh_camera_count(self) -> None:
        topology_keys = [
            key
            for key in active_topology_camera_keys(self.calibration_config)
            if key in CAMERA_KEYS
        ][:4]
        if hasattr(self, "camera_count") and self.camera_rows:
            enabled_count = sum(
                self.camera_rows[key].enabled.isChecked()
                for key in topology_keys
            )
            self.camera_count.blockSignals(True)
            self.camera_count.setRange(1, max(1, len(topology_keys)))
            self.camera_count.setValue(max(1, enabled_count))
            self.camera_count.blockSignals(False)
        self.apply_preview_view_state()
        self.update_health_table()

    def mark_camera_config_dirty(self, *_args: Any) -> None:
        if not self._syncing_camera_widgets:
            self._camera_config_dirty = True

    def on_camera_enabled_changed(self, _checked: bool) -> None:
        if self._syncing_camera_widgets:
            return
        self._camera_config_dirty = True
        self.refresh_camera_count()

    def on_camera_count_changed(self, value: int) -> None:
        if self._syncing_camera_widgets:
            return
        topology_keys = [
            key
            for key in active_topology_camera_keys(self.calibration_config)
            if key in CAMERA_KEYS
        ][:4]
        selected = set(topology_keys[: max(0, int(value))])
        self._syncing_camera_widgets = True
        try:
            for key, row in self.camera_rows.items():
                row.enabled.setChecked(key in selected)
        finally:
            self._syncing_camera_widgets = False
        self._camera_config_dirty = True
        self.refresh_camera_count()

    def choose_input_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, self.t("Choose image directory"), str(self.input_dir))
        if directory:
            self.input_dir = Path(directory)
            self.input_path_label.setText(str(self.input_dir))
            self.statusBar().showMessage(self.t("Selected {path}", path=self.input_dir))

    def browse_camera_source(self, camera_key: str) -> None:
        row = self.camera_rows[camera_key]
        source_type = row.source_type.currentText()
        if source_type == "image_dir":
            path = QFileDialog.getExistingDirectory(self, self.t("Choose image directory"), str(self.input_dir))
        elif source_type == "video_file":
            path, _ = QFileDialog.getOpenFileName(self, self.t("Choose video file"), str(PROJECT_ROOT), "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)")
        else:
            path = ""
        if path:
            row.source.setText(path)

    def live_stream_configs(self) -> list[CameraStreamConfig]:
        configs: list[CameraStreamConfig] = []
        for key in self.active_camera_keys():
            row = self.camera_rows[key]
            source_type = row.source_type.currentText()
            source_text = row.source.text().strip()
            source: str | int = source_text
            if source_type == "usb":
                try:
                    source = int(source_text or "0")
                except ValueError:
                    self.log(f"{key}: invalid USB index '{source_text}'")
                    source = ""
            configs.append(CameraStreamConfig(
                key=key,
                source_type=source_type,
                source=source,
                enabled=row.enabled.isChecked(),
            ))
        return configs

    def run_still_preview(self) -> None:
        self.stop_live_preview()
        self.frames = load_images_from_directory(self.input_dir)
        self.set_preview_content_mode(PreviewContentMode.STILL)
        self._process_and_render_frames("Loaded image directory")

    def start_live_preview(self) -> None:
        self.collect_camera_config_from_widgets()
        self.performance_config = self.camera_config.get("performance", {})
        self.stop_live_preview()
        self.bump_preview_session()
        self.configure_live_stitch_processor()
        self.frame_counts.clear()
        self.stream_error_log_counts.clear()
        self.stream_manager.start(self.live_stream_configs())
        self.log(
            "Starting live preview: "
            f"topology={self.calibration_config.get('stitch_topology', '')}, "
            f"cameras={self.active_camera_keys()}, "
            f"stitch_mode={self.live_stitch_mode}, "
            f"runtime_mode={self.runtime_stitch_config.mode.value}"
        )
        self.stream_snapshots = self.stream_manager.snapshots(include_frames=False)
        for key, snapshot in self.stream_snapshots.items():
            if snapshot.status == STREAM_FAILED:
                self.log(f"{key}: {snapshot.last_error}")
                self.stream_error_log_counts[key] = snapshot.failed_read_count
            elif snapshot.status not in (STREAM_IDLE,):
                self.log(f"{key}: stream worker started")

        if self.stream_manager.has_active_workers():
            if (
                self.preview_layout_mode == PreviewLayoutMode.FOCUS
                and self.focused_camera_id not in self.active_camera_keys()
            ):
                self.set_preview_layout_mode(PreviewLayoutMode.GRID)
            self.set_preview_content_mode(PreviewContentMode.LIVE)
            self.last_process_time = 0.0
            self.last_preview_time = 0.0
            self.last_health_time = 0.0
            self.last_canvas_render_time = 0.0
            self.last_qgc_output_status_time = 0.0
            self.preview_timer.start()
            self.statusBar().showMessage(self.t("Live preview running"))
        else:
            self.run_still_preview()

    def stop_live_preview(self) -> None:
        self.preview_timer.stop()
        self.bump_preview_session()
        self.stitch_processor.invalidate_session(self.preview_session_id)
        self.stream_manager.stop()
        self.stream_snapshots = self.stream_manager.snapshots(include_frames=False)
        self.frames.clear()
        self.warped.clear()
        self.canvas = None
        self.set_preview_content_mode(PreviewContentMode.STOPPED)
        for key, view in self.camera_views.items():
            view.set_placeholder(
                f"{self.camera_display_name(key)}\n{self.t('Live preview stopped')}"
            )
        for key, view in self.warped_views.items():
            view.set_placeholder(
                f"{self.camera_display_name(key)}\n{self.t('Live preview stopped')}"
            )
        if hasattr(self, "canvas_view"):
            self.canvas_view.set_placeholder(self.t("Live preview stopped"))
        self.update_health_table()
        self.statusBar().showMessage(self.t("Preview stopped"))

    def update_live_preview(self) -> None:
        now = time.perf_counter()
        preview_interval = self.effective_preview_interval()
        process_interval = 1.0 / max(1, int(self.performance_config.get("process_fps", 5)))
        should_refresh_preview = (now - self.last_preview_time) >= preview_interval
        should_submit_stitch = (now - self.last_process_time) >= process_interval

        if should_refresh_preview or should_submit_stitch:
            latest_frames, snapshots = self.stream_manager.latest_frames()
            self.stream_snapshots = snapshots
            for key, snapshot in snapshots.items():
                self.frame_counts[key] = snapshot.frame_count
                previous_failures = self.stream_error_log_counts.get(key, 0)
                if (
                    snapshot.status == STREAM_FAILED
                    and snapshot.last_error
                    and snapshot.failed_read_count != previous_failures
                ):
                    self.log(f"{key}: {snapshot.last_error}")
                    self.stream_error_log_counts[key] = snapshot.failed_read_count
            if latest_frames:
                self.frames.update(latest_frames)
                self.log_source_coordinate_warnings(latest_frames)

            signature = self.frame_signature(snapshots)
            has_new_frames = bool(signature) and signature != self.last_frame_signature
            if should_refresh_preview:
                self.last_preview_time = now
                self.refresh_raw_preview()
                self.update_health_table()
            if should_submit_stitch and self.frames and has_new_frames:
                self.last_process_time = now
                self.last_frame_signature = signature
                active = set(self.active_camera_keys())
                frames = {key: frame for key, frame in self.frames.items() if key in active}
                if frames:
                    self.stitch_processor.submit_latest(self.preview_session_id, frames)

        self.apply_latest_stitch_result()
        if not self.stream_manager.has_active_workers():
            self.preview_timer.stop()

    def effective_preview_interval(self) -> float:
        configured_interval = 1.0 / max(1, int(self.performance_config.get("preview_fps", 2)))
        if self.qgc_output_active:
            return max(configured_interval, 1.0 / 5.0)
        return configured_interval

    def should_render_canvas_now(self, *, force: bool = False) -> bool:
        if not self.should_render_canvas():
            return False
        if force or not self.qgc_output_active:
            self.last_canvas_render_time = time.perf_counter()
            return True
        now = time.perf_counter()
        if now - self.last_canvas_render_time >= 1.0 / 5.0:
            self.last_canvas_render_time = now
            return True
        return False

    def refresh_raw_preview(self, force: bool = False) -> None:
        if hasattr(self, "root_tabs") and self.root_tabs.currentIndex() != 0:
            return
        if (
            self.preview_layout_mode == PreviewLayoutMode.STITCHED
            and not force
        ):
            now = time.perf_counter()
            if now - self.last_stitched_raw_preview_time < 1.0:
                return
            self.last_stitched_raw_preview_time = now

        active_keys = self.active_camera_keys()[:4]
        if self.preview_layout_mode == PreviewLayoutMode.FOCUS:
            render_keys = [self.focused_camera_id] if self.focused_camera_id else []
        else:
            render_keys = active_keys

        for key in render_keys:
            view = self.camera_views[key]
            display_name = self.camera_display_name(key)
            if self.preview_content_mode == PreviewContentMode.STOPPED:
                view.set_placeholder(
                    f"{display_name}\n{self.t('Live preview stopped')}\n"
                    "下一步：点击“开始实时预览”连接已启用相机。"
                )
                continue
            if self.preview_content_mode == PreviewContentMode.STILL:
                if key in self.frames:
                    view.set_overlay(display_name, self.t("Static image"))
                    view.set_image(self.frames[key], display_name)
                else:
                    view.set_placeholder(f"{display_name}\n{self.t('No frame')}")
                continue

            snapshot = self.stream_snapshots.get(key)
            if snapshot is not None:
                if snapshot.status == STREAM_CONNECTING:
                    view.set_placeholder(
                        f"{display_name}\n正在连接视频流\n"
                        "可能原因：RTSP 正在握手或设备响应较慢。\n"
                        "下一步：稍候；若长时间不变，请检查地址、网络和设备。"
                    )
                    continue
                if snapshot.status == STREAM_FAILED:
                    error = (snapshot.last_error or self.t("Failed")).splitlines()[0][:180]
                    lines = [
                        display_name,
                        self.t("Video stream unavailable"),
                        self.t("Last error: {error}", error=error),
                        "可能原因：地址错误、网络中断、设备离线或认证失败。",
                        "下一步：检查相机配置和网络；其他正常相机可继续使用。",
                    ]
                    if (
                        self.preview_layout_mode == PreviewLayoutMode.FOCUS
                        and self.focused_camera_id == key
                    ):
                        lines.append(self.t("Use Back to Grid to leave this view."))
                    view.set_placeholder("\n".join(lines))
                    continue
            if key in self.frames:
                status = self.t(snapshot.status) if snapshot is not None else self.t("Live")
                view.set_overlay(display_name, status)
                view.set_image(self.frames[key], display_name)
            else:
                status = self.t(snapshot.status) if snapshot is not None else self.t("No frame")
                view.set_placeholder(
                    f"{display_name}\n{status}\n"
                    "下一步：确认该相机已启用且输入源可用。"
                )

    def should_render_canvas(self) -> bool:
        if not hasattr(self, "preview_tabs"):
            return False
        if self.preview_layout_mode == PreviewLayoutMode.STITCHED:
            return True
        return (
            hasattr(self, "root_tabs")
            and self.root_tabs.currentIndex() == 0
            and self.preview_tabs.isVisible()
            and self.preview_tabs.currentIndex() == self.canvas_tab_index
        )

    def should_render_warped(self) -> bool:
        return (
            bool(self.performance_config.get("refresh_warped_preview", False))
            and hasattr(self, "root_tabs")
            and self.root_tabs.currentIndex() == 0
            and self.preview_tabs.isVisible()
            and self.preview_tabs.currentIndex() == self.warped_tab_index
        )

    def refresh_warped_views(self) -> None:
        if not self.should_render_warped():
            return
        for key in self.active_camera_keys()[:4]:
            view = self.warped_views[key]
            display_name = self.camera_display_name(key)
            if self.preview_content_mode == PreviewContentMode.STOPPED:
                view.set_placeholder(
                    f"{display_name}\n{self.t('Live preview stopped')}"
                )
            elif key in self.warped:
                status = (
                    self.t("Static image")
                    if self.preview_content_mode == PreviewContentMode.STILL
                    else self.t("Live")
                )
                view.set_overlay(display_name, status)
                view.set_image(self.warped[key], f"{display_name} warped")
            else:
                view.set_placeholder(f"{display_name}\n{self.t('No warped frame')}")

    def apply_latest_stitch_result(self) -> None:
        result = self.stitch_processor.take_latest_result()
        if result is None:
            return
        if result.session_id != self.preview_session_id:
            return
        if result.result_id <= self.displayed_stitch_result_id:
            return

        self.displayed_stitch_result_id = result.result_id
        self.last_stitch_ms = result.elapsed_ms
        if result.error:
            self.last_stitch_ui_error = str(result.error).splitlines()[0][:220]
            if result.result_id != self.last_stitch_error_result_id:
                self.last_stitch_error_result_id = result.result_id
                self.log(f"Stitch failed: {result.error}", level="ERROR")
            if self.should_render_canvas():
                self.canvas_view.set_placeholder(
                    "拼接结果暂不可用\n"
                    f"发生了什么：{self.last_stitch_ui_error}\n"
                    "下一步：检查状态摘要、日志和 topology/profile；"
                    "可先切换 Grid 或 Focus 检查各路原始画面。"
                )
            self.update_preview_status_summary()
            return

        self.last_stitch_ui_error = ""
        self.last_runtime_stitch_status = result.runtime_status
        self.last_runtime_warnings = list(result.runtime_warnings)
        self.last_runtime_metrics = result.runtime_metrics
        warning_text = "; ".join(self.last_runtime_warnings)
        if warning_text and warning_text != self.last_runtime_warning_log_text:
            self.last_runtime_warning_log_text = warning_text
            self.log(warning_text, level="WARNING")
        self.warped = result.warped
        self.canvas = result.canvas
        self.write_qgc_output_frame(self.canvas)
        self.refresh_warped_views()
        if self.canvas is not None and self.should_render_canvas_now():
            self.render_canvas_view()
        message = self.t(
            "{status}: {count} active frames | stitch {ms:.1f} ms",
            status=self.t("Live frame"),
            count=len(self.frames),
            ms=self.last_stitch_ms,
        )
        self.statusBar().showMessage(
            message + self.runtime_timing_status_suffix(self.last_runtime_metrics)
        )

    def _process_and_render_frames(self, status_prefix: str) -> None:
        start_time = time.perf_counter()
        active = set(self.active_camera_keys())
        self.frames = {key: frame for key, frame in self.frames.items() if key in active}
        if not self.frames:
            QMessageBox.warning(self, self.t("No images"), self.t("No matching camera images were found."))
            return
        self.log_source_coordinate_warnings(self.frames)

        try:
            self.warped, self.canvas, runtime_warnings = (
                self.process_frames_with_runtime_controller(self.frames)
            )
        except Exception as exc:
            QMessageBox.critical(self, self.t("Stitch failed"), str(exc))
            return
        if runtime_warnings:
            self.log("; ".join(runtime_warnings), level="WARNING")

        now = time.perf_counter()
        preview_interval = self.effective_preview_interval()
        should_refresh_preview = (now - self.last_preview_time) >= preview_interval or status_prefix != "Live frame"
        if should_refresh_preview:
            self.last_preview_time = now
            self.refresh_raw_preview(force=True)
            self.refresh_warped_views()
        if self.canvas is not None and self.should_render_canvas_now(force=status_prefix != "Live frame"):
            self.render_canvas_view()
            if hasattr(self, "seam_editor") and status_prefix != "Live frame":
                self.refresh_seam_editor()
        self.write_qgc_output_frame(self.canvas)
        self.last_stitch_ms = (time.perf_counter() - start_time) * 1000.0
        if now - self.last_health_time >= 1.0 or status_prefix != "Live frame":
            self.last_health_time = now
            self.update_health_table()
        message = self.t(
            "{status}: {count} active frames | stitch {ms:.1f} ms",
            status=self.t(status_prefix),
            count=len(self.frames),
            ms=self.last_stitch_ms,
        )
        self.statusBar().showMessage(
            message + self.runtime_timing_status_suffix(self.last_runtime_metrics)
        )

    def update_health_table(self) -> None:
        keys = CAMERA_KEYS
        self.health_table.setRowCount(len(keys))
        active = set(self.active_camera_keys()) if self.camera_rows else set(keys)
        for row_index, key in enumerate(keys):
            source = ""
            if self.camera_rows:
                source = self.camera_rows[key].source.text() or self.camera_rows[key].source_type.currentText()
            snapshot = self.stream_snapshots.get(key)
            if snapshot is not None:
                age = time.time() - snapshot.frame_timestamp if snapshot.frame_timestamp else -1
                age_text = f", age {age:.1f}s" if age >= 0 else ""
                error_text = f", fail {snapshot.failed_read_count}" if snapshot.failed_read_count else ""
                status = f"{self.t(snapshot.status)}{age_text}{error_text}"
            else:
                status = self.t("Frame") if key in self.frames else self.t("Idle")
            values = [
                key,
                self.t("yes") if key in active else self.t("no"),
                status,
                str(self.frame_counts.get(key, 0)),
                source,
            ]
            for column, value in enumerate(values):
                self.health_table.setItem(row_index, column, QTableWidgetItem(value))
        self.update_preview_status_summary()

    def save_results(self) -> None:
        if self.canvas is None:
            QMessageBox.information(self, self.t("No result"), self.t("Run preview first."))
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_image(self.output_dir / "canvas.jpg", self.canvas)
        for name, image in self.warped.items():
            save_image(self.output_dir / f"{name}_warped.jpg", image)
        self.statusBar().showMessage(self.t("Saved results to {path}", path=self.output_dir))

    def collect_camera_config_from_widgets(self) -> None:
        if not self.camera_rows:
            return
        topology_keys = [
            key
            for key in active_topology_camera_keys(self.calibration_config)
            if key in CAMERA_KEYS
        ][:4]
        self.camera_config["active_camera_count"] = sum(
            self.camera_rows[key].enabled.isChecked()
            for key in topology_keys
        )
        self.camera_config["camera_order"] = CAMERA_KEYS
        self.camera_config["language"] = self.language
        self.camera_config["performance"] = {
            "process_fps": int(self.process_fps.value()),
            "preview_fps": int(self.preview_fps.value()),
            "max_input_width": int(self.max_input_width.value()),
            "refresh_warped_preview": self.refresh_warped_preview.isChecked(),
            "use_intrinsics": self.use_intrinsics_live.isChecked(),
            "b2_candidate_opencl": self.b2_candidate_opencl.isChecked(),
        }
        self.camera_config["cameras"] = {key: row.to_config() for key, row in self.camera_rows.items()}
        if hasattr(self, "qgc_output_kind"):
            self.camera_config["qgc_video_output"] = (
                self.qgc_video_output_config_from_widgets()
            )

    def backup_before_config_write(self) -> None:
        try:
            path = backup_configs()
        except Exception as exc:
            self.log(self.t("Auto backup failed: {error}", error=str(exc)))
            return
        self.log(self.t("Auto backup created: {path}", path=path))

    def persist_camera_config_from_widgets(self) -> bool:
        self.collect_camera_config_from_widgets()
        try:
            self._camera_config_revision = save_config(
                "cameras.yaml",
                self.camera_config,
                expected_revision=self._camera_config_revision,
            )
        except ConfigConflictError:
            self.show_config_conflict()
            return False
        self.performance_config = self.camera_config.get("performance", {})
        self._camera_config_dirty = False
        return True

    def persist_calibration_config(self) -> bool:
        try:
            self._calibration_config_revision = save_config(
                "calibration.yaml",
                self.calibration_config,
                expected_revision=self._calibration_config_revision,
            )
        except ConfigConflictError:
            self.show_config_conflict()
            return False
        return True

    def show_config_conflict(self) -> None:
        self.log(
            "Configuration changed on disk; refusing stale save.",
            level="ERROR",
        )
        QMessageBox.warning(
            self,
            self.t("Configuration changed on disk"),
            self.t(
                "Configuration changed on disk. Reload before saving to avoid overwriting newer settings."
            ),
        )

    def save_camera_config(self) -> None:
        if (
            config_revision("cameras.yaml") != self._camera_config_revision
            or config_revision("calibration.yaml")
            != self._calibration_config_revision
        ):
            self.show_config_conflict()
            return
        self.backup_before_config_write()
        if not self.persist_camera_config_from_widgets():
            return

        stitch_profile = self.current_stitch_profile()
        stitch_profile.setdefault("canvas", {})["width"] = int(self.canvas_width.value())
        stitch_profile.setdefault("canvas", {})["height"] = int(self.canvas_height.value())
        if not self.persist_calibration_config():
            return
        self.performance_config = self.camera_config.get("performance", {})
        self.stitcher = self.create_stitcher()
        self.bump_preview_session()
        self.configure_live_stitch_processor()
        self.refresh_seam_editor()
        self.update_topology_diagnostics()
        self.statusBar().showMessage(self.t("Saved configs to {path}", path=CONFIG_DIR))

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_qgc_output_service()
        self.stop_live_preview()
        self.stitch_processor.shutdown()
        super().closeEvent(event)

    def reload_configs(self) -> None:
        self.reload_runtime_state()
        QMessageBox.information(
            self,
            self.t("Reloaded"),
            self.t("Configuration and editor controls were reloaded."),
        )

    def reload_runtime_state(self) -> None:
        self.calibration_config = load_config("calibration.yaml")
        self.camera_config = load_config("cameras.yaml")
        self._calibration_config_revision = config_revision("calibration.yaml")
        self._camera_config_revision = config_revision("cameras.yaml")
        self.performance_config = self.camera_config.get("performance", {})
        self.live_candidate_directory = latest_calibration_candidate(
            topology=str(
                self.calibration_config.get(
                    "stitch_topology",
                    "triple_front_panorama",
                )
            )
        )
        self._candidate_ui_cache_path = None
        self._candidate_ui_cache = {}
        if self.live_candidate_directory is None:
            self.live_stitch_mode = "template"
        if self.runtime_layout_candidate is not None:
            try:
                self.runtime_layout_candidate = load_layout_candidate_for_runtime(
                    self.runtime_layout_candidate.path,
                    expected_profile_id=self.current_runtime_profile_id(),
                )
            except Exception as exc:
                self.log(
                    f"Runtime layout candidate unavailable after reload: {exc}",
                    level="WARNING",
                )
                self.runtime_layout_candidate = None
                self.runtime_stitch_config = RuntimeStitchConfig()
        self.apply_live_stitch_mode_controls()
        self.refresh_stitch_runtime_controls()
        self.sync_camera_config_widgets()
        self.stitcher = self.create_stitcher()
        self.bump_preview_session()
        self.configure_live_stitch_processor()
        self.populate_point_table()
        self.populate_seam_table()
        self.refresh_seam_editor()
        self.update_topology_diagnostics()
        self.refresh_pairwise_pair_choices()

    def sync_camera_config_widgets(self) -> None:
        if not self.camera_rows:
            return
        self._syncing_camera_widgets = True
        try:
            cameras = self.camera_config.get("cameras", {})
            for key, row in self.camera_rows.items():
                row.apply_config(cameras.get(key, {}))

            self.process_fps.setValue(
                int(self.performance_config.get("process_fps", 5))
            )
            self.preview_fps.setValue(
                int(self.performance_config.get("preview_fps", 2))
            )
            self.max_input_width.setValue(
                int(self.performance_config.get("max_input_width", 960))
            )
            self.refresh_warped_preview.setChecked(
                bool(self.performance_config.get("refresh_warped_preview", False))
            )
            self.use_intrinsics_live.setChecked(
                bool(self.performance_config.get("use_intrinsics", False))
            )
            if hasattr(self, "b2_candidate_opencl"):
                self.b2_candidate_opencl.setChecked(
                    bool(self.performance_config.get("b2_candidate_opencl", False))
                )

            stitch_profile = self.current_stitch_profile()
            self.canvas_width.setValue(
                int(stitch_profile.get("canvas", {}).get("width", 2440))
            )
            self.canvas_height.setValue(
                int(stitch_profile.get("canvas", {}).get("height", 1800))
            )
            if hasattr(self, "feather_width"):
                self.feather_width.setValue(self.current_feather_width())
            if hasattr(self, "qgc_output_kind"):
                self.sync_qgc_video_output_widgets()
        finally:
            self._syncing_camera_widgets = False
        self._camera_config_dirty = False
        self.refresh_camera_count()

    def qgc_video_output_config_from_widgets(self) -> dict[str, Any]:
        kind = self.qgc_output_kind.currentData() or "udp_mpegts"
        kind = str(kind)
        url = self.qgc_output_url.text().strip() or self._default_qgc_output_url(kind)
        config = {
            "kind": kind,
            "url": url,
            "fps": float(self.qgc_output_fps.value()),
            "bitrate": self.qgc_output_bitrate.text().strip() or "6000k",
            "rtsp_transport": str(self.qgc_rtsp_transport.currentData() or "tcp"),
            "ffmpeg_path": self.qgc_ffmpeg_path.text().strip() or "ffmpeg",
        }
        raw = self.camera_config.get("qgc_video_output", {})
        if isinstance(raw, dict):
            for key in ("output_width", "output_height"):
                value = int(raw.get(key, 0) or 0)
                if value > 0:
                    config[key] = value
        return config

    def sync_qgc_video_output_widgets(self) -> None:
        config = self._qgc_video_output_config()
        self.qgc_output_kind.setCurrentIndex(
            max(0, self.qgc_output_kind.findData(config["kind"]))
        )
        self.qgc_output_url.setText(config["url"])
        self.qgc_output_url.setPlaceholderText(
            self._default_qgc_output_url(config["kind"])
        )
        self.qgc_output_fps.setValue(float(config["fps"]))
        self.qgc_output_bitrate.setText(config["bitrate"])
        self.qgc_rtsp_transport.setCurrentIndex(
            max(0, self.qgc_rtsp_transport.findData(config["rtsp_transport"]))
        )
        self.qgc_ffmpeg_path.setText(config["ffmpeg_path"])
        self.update_qgc_output_mode_controls()

    def on_qgc_output_kind_changed(self, _index: int) -> None:
        kind = str(self.qgc_output_kind.currentData() or "udp_mpegts")
        self.qgc_output_url.setPlaceholderText(self._default_qgc_output_url(kind))
        current_url = self.qgc_output_url.text().strip()
        default_urls = {
            self._default_qgc_output_url("rtsp"),
            self._default_qgc_output_url("udp_mpegts"),
            "",
        }
        if current_url in default_urls:
            self.qgc_output_url.setText(self._default_qgc_output_url(kind))
        self.update_qgc_output_mode_controls()

    def update_qgc_output_mode_controls(self) -> None:
        if not hasattr(self, "qgc_rtsp_transport"):
            return
        kind = str(self.qgc_output_kind.currentData() or "udp_mpegts")
        self.qgc_rtsp_transport.setEnabled(kind == "rtsp")

    def save_qgc_video_output_settings(self) -> None:
        if config_revision("cameras.yaml") != self._camera_config_revision:
            self.show_config_conflict()
            return
        self.camera_config["qgc_video_output"] = (
            self.qgc_video_output_config_from_widgets()
        )
        self.backup_before_config_write()
        if not self.persist_camera_config_from_widgets():
            return
        self.statusBar().showMessage(self.t("QGC output settings saved."))
        self.log(self.t("QGC output settings saved."))

    def start_qgc_output_service(self) -> None:
        if self.qgc_output_active:
            QMessageBox.information(
                self,
                self.t("QGC output service"),
                self.t("QGC output service already running."),
            )
            return
        output = self.qgc_video_output_config_from_widgets()
        self.qgc_video_sink = FfmpegVideoSink(
            VideoOutputConfig(
                kind=output["kind"],
                url=output["url"],
                fps=float(output["fps"]),
                bitrate=output["bitrate"],
                rtsp_transport=output["rtsp_transport"],
                ffmpeg_path=output["ffmpeg_path"],
                output_width=output.get("output_width") or None,
                output_height=output.get("output_height") or None,
            )
        )
        self.qgc_output_active = True
        self.qgc_output_frames_written = 0
        self.qgc_output_last_error = ""
        self.last_qgc_output_status_time = 0.0
        status = self.t(
            "QGC output service started: {url}",
            url=output["url"],
        )
        if hasattr(self, "qgc_output_service_status"):
            self.qgc_output_service_status.setText(status)
        self.qgc_output_start_button.setEnabled(False)
        self.qgc_output_stop_button.setEnabled(True)
        self.statusBar().showMessage(status)
        self.log(status)
        self.write_qgc_output_frame(self.canvas)

    def stop_qgc_output_service(self) -> None:
        sink = self.qgc_video_sink
        self.qgc_video_sink = None
        self.qgc_output_active = False
        if sink is not None:
            sink.close()
        message = self.t("QGC output service stopped.")
        if hasattr(self, "qgc_output_service_status"):
            self.qgc_output_service_status.setText(message)
        if hasattr(self, "qgc_output_start_button"):
            self.qgc_output_start_button.setEnabled(True)
        if hasattr(self, "qgc_output_stop_button"):
            self.qgc_output_stop_button.setEnabled(False)
        self.statusBar().showMessage(message)
        self.log(message)

    def write_qgc_output_frame(self, canvas: np.ndarray | None) -> None:
        if not self.qgc_output_active or self.qgc_video_sink is None or canvas is None:
            return
        try:
            self.qgc_video_sink.write(canvas)
        except Exception as exc:
            self.qgc_output_last_error = str(exc)
            self.log(f"QGC output service error: {exc}", level="ERROR")
            self.stop_qgc_output_service()
            return
        self.qgc_output_frames_written += 1
        now = time.perf_counter()
        if (
            hasattr(self, "qgc_output_service_status")
            and (
                self.qgc_output_frames_written == 1
                or now - self.last_qgc_output_status_time >= 1.0
            )
        ):
            self.last_qgc_output_status_time = now
            output = self.qgc_video_output_config_from_widgets()
            timing_suffix = self.runtime_timing_status_suffix(self.last_runtime_metrics)
            self.qgc_output_service_status.setText(
                f"{self.t('QGC output service started: {url}', url=output['url'])}\n"
                f"Frames: {self.qgc_output_frames_written} | stitch {self.last_stitch_ms:.1f} ms"
                f"{timing_suffix}"
            )

    def save_project_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save DeepShark View Studio project",
            str(PROJECT_ROOT / "projects" / "project.dsvs.yaml"),
            "DeepShark Project (*.yaml *.yml)",
        )
        if not path:
            return
        try:
            saved_path = save_project(path)
        except Exception as exc:
            QMessageBox.critical(self, "Save project failed", str(exc))
            return
        self.project_status.setText(f"Saved project: {saved_path}")
        self.log(f"Saved project: {saved_path}")

    def open_project_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open DeepShark View Studio project",
            str(PROJECT_ROOT / "projects"),
            "DeepShark Project (*.yaml *.yml);;All Files (*)",
        )
        if not path:
            return
        try:
            payload = load_project(path)
            self.reload_runtime_state()
        except Exception as exc:
            QMessageBox.critical(self, "Open project failed", str(exc))
            return
        self.project_status.setText(f"Loaded project: {path}")
        self.log(f"Loaded project version {payload.get('version')} from {path}")

    def backup_current_configs(self) -> None:
        try:
            path = backup_configs()
        except Exception as exc:
            QMessageBox.critical(self, "Backup failed", str(exc))
            return
        self.project_status.setText(f"Backup created: {path}")
        self.log(f"Backup created: {path}")

    def export_runtime_config_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export runtime config",
            str(PROJECT_ROOT / "projects" / "runtime_config.yaml"),
            "YAML (*.yaml *.yml)",
        )
        if not path:
            return
        try:
            saved_path = export_runtime_config(path)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.project_status.setText(f"Runtime config exported: {saved_path}")
        self.log(f"Runtime config exported: {saved_path}")

    def export_project_package_file(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            self.t("Choose project package export folder"),
            str(PROJECT_ROOT / "projects"),
        )
        if not directory:
            return
        try:
            result = export_project_package(
                directory,
                project_name=str(
                    self.calibration_config.get(
                        "stitch_topology",
                        "triple_front_panorama",
                    )
                ),
                far_field_candidate_path=(
                    self.far_field_layout_candidate.path
                    if self.far_field_layout_candidate is not None
                    else None
                ),
                near_field_candidate_path=(
                    self.runtime_layout_candidate.path
                    if self.runtime_layout_candidate is not None
                    else None
                ),
                b2_candidate_path=self.current_b2_candidate_directory(),
                fisheye_intrinsics_path=self._current_fisheye_intrinsics_path(),
                default_stitch_mode=self.runtime_stitch_config.mode.value,
                use_far_field_custom=self.runtime_stitch_config.use_far_field_custom_layout,
                near_field_projection_source=self.runtime_stitch_config.projection_source.value,
            )
        except Exception as exc:
            QMessageBox.critical(self, self.t("Export project package failed"), str(exc))
            return
        self.project_status.setText(
            self.t("Project package exported: {path}", path=result.package_root)
        )
        self.log(f"Project package exported: {result.package_root}")
        if result.warnings:
            self.log(
                "Project package export warnings: " + "; ".join(result.warnings),
                level="WARNING",
            )

    def import_project_package_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.t("Open project package manifest"),
            str(PROJECT_ROOT / "projects"),
            self.t("Project Package (*.yaml *.yml);;All Files (*)"),
        )
        if not path:
            return
        try:
            validation = validate_project_package(path)
        except Exception as exc:
            QMessageBox.critical(self, self.t("Import project package failed"), str(exc))
            return
        if not validation.valid:
            message = "; ".join(validation.errors)
            QMessageBox.critical(
                self,
                self.t("Import project package failed"),
                self.t("Project package invalid: {errors}", errors=message),
            )
            self.project_status.setText(
                self.t("Project package invalid: {errors}", errors=message)
            )
            return
        answer = QMessageBox.question(
            self,
            self.t("Activate project package?"),
            self.t(
                "The package is valid. Activate it now?\n\nThis will back up current configs, then replace active configs with the package configs. Candidate files remain inside the package and calibration.yaml is not edited during export/validation."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self.project_status.setText(
                self.t("Project package validated but not activated: {path}", path=validation.package_root)
            )
            self.log(f"Project package validated but not activated: {validation.package_root}")
            return
        try:
            result = activate_project_package(path)
            self.reload_runtime_state()
        except Exception as exc:
            QMessageBox.critical(self, self.t("Import project package failed"), str(exc))
            return
        self.project_status.setText(
            self.t("Project package activated: {path}", path=result.package_root)
        )
        self.log(f"Project package activated: {result.package_root}")

    def validate_project_package_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.t("Validate project package manifest"),
            str(PROJECT_ROOT / "projects"),
            self.t("Project Package (*.yaml *.yml);;All Files (*)"),
        )
        if not path:
            return
        try:
            validation = validate_project_package(path)
        except Exception as exc:
            QMessageBox.critical(self, self.t("Validate project package failed"), str(exc))
            return
        if validation.valid:
            self.project_status.setText(
                self.t("Project package valid: {path}", path=validation.package_root)
            )
            self.log(f"Project package valid: {validation.package_root}")
        else:
            message = "; ".join(validation.errors)
            self.project_status.setText(
                self.t("Project package invalid: {errors}", errors=message)
            )
            self.log(f"Project package invalid: {message}", level="ERROR")

    def _current_fisheye_intrinsics_path(self) -> Path | None:
        source = (
            self.runtime_fisheye_intrinsics_source
            or self.layout_tuner_fisheye_intrinsics_source
        )
        if source is None:
            return None
        return source.path

    def save_board_settings(self) -> None:
        board = self.calibration_config.setdefault("calibration_board", {})
        board["total_columns"] = int(self.board_columns.value())
        board["total_rows"] = int(self.board_rows.value())
        board["square_size_mm"] = float(self.square_size.value())
        if not self.persist_calibration_config():
            return
        self.log("Saved chessboard settings.")

    def load_calibration_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose calibration image", str(PROJECT_ROOT), "Images (*.jpg *.jpeg *.png *.bmp);;All Files (*)")
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            QMessageBox.warning(self, "Load failed", f"Could not read image: {path}")
            return
        self.calibration_image = image
        self.calibration_image_view.set_editor_image(image)
        self.calibration_image_view.set_points(self.current_source_points())
        self.log_source_coordinate_warnings(
            {self.calibration_camera.currentText(): image}
        )
        self.log(f"Loaded calibration image: {path}")

    def detect_chessboard(self) -> None:
        if self.calibration_image is None:
            QMessageBox.information(self, self.t("No image"), self.t("Load a calibration image first."))
            return
        self.save_board_settings()
        spec = spec_from_config(self.calibration_config)
        annotated, ok, count = draw_chessboard_detection(self.calibration_image, spec)
        self.calibration_image_view.set_editor_image(annotated)
        self.calibration_image_view.set_points(self.current_source_points())
        self.log(
            f"Chessboard detection: {'OK' if ok else 'FAILED'}; "
            f"inner pattern={spec.inner_columns}x{spec.inner_rows}; corners={count}; square={spec.square_size_mm}mm"
        )
        if not ok:
            self.log(
                "标定板检测失败：未找到完整内角点。可能原因是板被遮挡、"
                "画面模糊、反光，或行列参数与实物不一致。"
                "下一步：停稳标定板、保证整板清晰可见，并核对 11×8 内角点。",
                level="WARNING",
            )

    def calibrate_from_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Choose calibration image folder", str(PROJECT_ROOT))
        if not directory:
            return
        images = []
        for path in sorted(Path(directory).glob("*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            image = cv2.imread(str(path))
            if image is not None:
                images.append(image)
        if not images:
            QMessageBox.warning(self, "No images", "No readable calibration images were found.")
            return
        self.save_board_settings()
        spec = spec_from_config(self.calibration_config)
        try:
            result = calibrate_camera_from_images(images, spec)
        except Exception as exc:
            QMessageBox.critical(self, "Calibration failed", str(exc))
            return
        camera_key = self.calibration_camera.currentText()
        self.calibration_config.setdefault("camera_intrinsics", {})[camera_key] = result
        if not self.persist_calibration_config():
            return
        report_path = PROJECT_ROOT / "reports" / f"{camera_key}_calibration_report.md"
        write_calibration_report(report_path, camera_key, result, spec)
        self.log(
            f"Calibrated {camera_key}: RMS={result['rms']:.4f}, "
            f"mean reprojection={result.get('mean_reprojection_error', 0):.4f}px, "
            f"usable images={result['image_count']}. Report: {report_path}"
        )

    def preview_undistort(self) -> None:
        if self.calibration_image is None:
            QMessageBox.information(self, self.t("No image"), self.t("Load a calibration image first."))
            return
        camera_key = self.calibration_camera.currentText()
        intrinsics = self.calibration_config.get("camera_intrinsics", {}).get(camera_key)
        if not intrinsics:
            QMessageBox.information(self, self.t("No intrinsics"), self.t("No saved camera intrinsics for {camera}.", camera=camera_key))
            return
        try:
            undistorted = undistort_image(self.calibration_image, intrinsics)
        except Exception as exc:
            QMessageBox.warning(self, "Undistort failed", str(exc))
            return
        self.calibration_warp_view.set_image(undistorted, f"{camera_key} undistort preview")
        self.log(f"Previewed undistortion for {camera_key}.")

    def export_calibration_report(self) -> None:
        camera_key = self.calibration_camera.currentText()
        result = self.calibration_config.get("camera_intrinsics", {}).get(camera_key)
        if not result:
            QMessageBox.information(self, self.t("No calibration"), self.t("No calibration result saved for {camera}.", camera=camera_key))
            return
        default_path = PROJECT_ROOT / "reports" / f"{camera_key}_calibration_report.md"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export calibration quality report",
            str(default_path),
            "Markdown (*.md);;All Files (*)",
        )
        if not path:
            return
        spec = spec_from_config(self.calibration_config)
        try:
            write_calibration_report(path, camera_key, result, spec)
        except Exception as exc:
            QMessageBox.critical(self, "Report export failed", str(exc))
            return
        self.log(f"Exported calibration report: {path}")

    def populate_point_table(self) -> None:
        camera_key = self.calibration_camera.currentText() if hasattr(self, "calibration_camera") else CAMERA_KEYS[0]
        camera = self.current_stitch_profile().get("cameras", {}).get(camera_key, {})
        source_points = camera.get("source_points", [[0, 0]] * 4)
        target_points = camera.get("target_points", [[0, 0]] * 4)
        self._updating_point_table = True
        self.point_table.setRowCount(4)
        for row in range(4):
            values = [
                str(row + 1),
                str(source_points[row][0]),
                str(source_points[row][1]),
                str(target_points[row][0]),
                str(target_points[row][1]),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.point_table.setItem(row, column, item)
        self._updating_point_table = False
        if hasattr(self, "calibration_image_view"):
            profile = self.current_stitch_profile()
            self.calibration_image_view.set_source_coordinate_contract(
                str(profile.get("source_coordinate_space", "")),
                profile.get("source_reference_size"),
            )
            self.calibration_image_view.set_points(source_points)

    def current_source_points(self) -> list[list[float]]:
        camera_key = self.calibration_camera.currentText()
        camera = self.current_stitch_profile().get("cameras", {}).get(camera_key, {})
        return [[float(x), float(y)] for x, y in camera.get("source_points", [[0, 0]] * 4)]

    def current_target_points(self) -> list[list[float]]:
        camera_key = self.calibration_camera.currentText()
        camera = self.current_stitch_profile().get("cameras", {}).get(camera_key, {})
        return [[float(x), float(y)] for x, y in camera.get("target_points", [[0, 0]] * 4)]

    def on_source_point_moved(self, index: int, x: float, y: float) -> None:
        self._updating_point_table = True
        self.point_table.item(index, 1).setText(f"{x:.3f}")
        self.point_table.item(index, 2).setText(f"{y:.3f}")
        self._updating_point_table = False
        self.statusBar().showMessage(f"Point {index + 1}: {x:.1f}, {y:.1f}")

    def on_point_table_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_point_table or item.column() not in (1, 2):
            return
        try:
            points = []
            for row in range(4):
                points.append([
                    float(self.point_table.item(row, 1).text()),
                    float(self.point_table.item(row, 2).text()),
                ])
        except Exception:
            return
        self.calibration_image_view.set_points(points)

    def apply_point_edits(self) -> None:
        camera_key = self.calibration_camera.currentText()
        source_points = []
        target_points = []
        try:
            for row in range(4):
                sx = float(self.point_table.item(row, 1).text())
                sy = float(self.point_table.item(row, 2).text())
                tx = float(self.point_table.item(row, 3).text())
                ty = float(self.point_table.item(row, 4).text())
                source_points.append([sx, sy])
                target_points.append([tx, ty])
        except Exception as exc:
            QMessageBox.warning(self, "Invalid points", str(exc))
            return

        camera = self.current_stitch_profile().setdefault("cameras", {}).setdefault(camera_key, {})
        camera["source_points"] = source_points
        camera["target_points"] = target_points
        origin = self.current_stitch_profile().setdefault("calibration_origin", {})
        origin["source_points"] = "field_adjusted"
        origin["target_points"] = "field_adjusted"
        actual_size = None
        if self.calibration_image is not None:
            actual_size = (
                int(self.calibration_image.shape[1]),
                int(self.calibration_image.shape[0]),
            )
        diagnostics = source_coordinate_diagnostics(
            self.current_stitch_profile(),
            camera_key,
            actual_size,
        )
        for warning in diagnostics["warnings"]:
            self.log(
                f"Source coordinate warning: {warning}",
                level="WARNING",
            )
        if not self.persist_calibration_config():
            return
        self.stitcher = self.create_stitcher()
        self.bump_preview_session()
        self.configure_live_stitch_processor()
        self.calibration_image_view.set_points(source_points)
        self.update_topology_diagnostics()
        self.log(f"Saved perspective points for {camera_key}.")

    def preview_calibration_warp(self) -> None:
        if self.calibration_image is None:
            QMessageBox.information(self, self.t("No image"), self.t("Load a calibration image first."))
            return
        camera_key = self.calibration_camera.currentText()
        try:
            source_points = []
            target_points = []
            for row in range(4):
                source_points.append([
                    float(self.point_table.item(row, 1).text()),
                    float(self.point_table.item(row, 2).text()),
                ])
                target_points.append([
                    float(self.point_table.item(row, 3).text()),
                    float(self.point_table.item(row, 4).text()),
                ])
            matrix = cv2.getPerspectiveTransform(
                np.asarray(source_points, dtype=np.float32),
                np.asarray(target_points, dtype=np.float32),
            )
            stitch_profile = self.current_stitch_profile()
            width = int(stitch_profile.get("canvas", {}).get("width", 2440))
            height = int(stitch_profile.get("canvas", {}).get("height", 1800))
            warped = cv2.warpPerspective(self.calibration_image, matrix, (width, height))
        except Exception as exc:
            QMessageBox.warning(self, "Preview failed", str(exc))
            return
        self.calibration_warp_view.set_image(warped, f"{camera_key} warp preview")
        self.log(f"Previewed warp for {camera_key}.")

    def current_stitch_points(self) -> dict[str, list[list[float]]]:
        return {
            name: [[float(x), float(y)] for x, y in endpoints]
            for name, endpoints in self.current_stitch_profile().get("stitch_points", {}).items()
        }

    def current_feather_width(self) -> int:
        profile = self.current_stitch_profile()
        composition = profile.get("composition", {})
        if "feather_width" in composition:
            return int(composition["feather_width"])
        overlaps = profile.get("overlaps", [])
        if overlaps and "feather_width" in overlaps[0]:
            return int(overlaps[0]["feather_width"])
        return 120

    def update_topology_diagnostics(self) -> None:
        if not hasattr(self, "topology_diagnostics_label"):
            return
        diagnostics = topology_diagnostics(self.calibration_config)
        canvas = diagnostics.get("canvas", {})
        origin = diagnostics.get("calibration_origin", {})
        lines = [
            f"Topology: {diagnostics.get('topology', '')}",
            f"Camera order: {' -> '.join(diagnostics.get('camera_order', []))}",
            f"Canvas: {canvas.get('width', 0)} x {canvas.get('height', 0)}",
            (
                "Source contract: "
                f"{diagnostics.get('source_coordinate_space', '')}, "
                f"reference={diagnostics.get('source_reference_size')}, "
                f"version={diagnostics.get('source_contract_version')}"
            ),
            (
                "Point origin: "
                f"source={origin.get('source_points', 'unmarked')}, "
                f"target={origin.get('target_points', 'unmarked')}"
            ),
        ]
        for overlap in diagnostics.get("overlaps", []):
            cameras = overlap.get("cameras", [])
            x_range = overlap.get("x_range", [])
            seam_position = overlap.get("seam_x")
            feather = overlap.get(
                "feather_width",
                diagnostics.get("composition", {}).get("feather_width", 0),
            )
            range_text = (
                f"[{float(x_range[0]):.1f}, {float(x_range[1]):.1f}]"
                if len(x_range) == 2
                else "unavailable"
            )
            seam_text = (
                f"{float(seam_position):.1f}"
                if seam_position is not None
                else "unavailable"
            )
            lines.append(
                f"Overlap {' <-> '.join(str(key) for key in cameras)}: "
                f"X={range_text}, seam={overlap.get('seam', '')} "
                f"@ X={seam_text}, feather={feather}"
            )
        validation_errors = validate_overlap_seams(
            self.current_stitch_profile()
        )
        if validation_errors:
            lines.append("Validation: INVALID")
            lines.extend(f"- {error}" for error in validation_errors)
            self.topology_diagnostics_label.setStyleSheet(
                "QLabel { color: #b91c1c; }"
            )
        else:
            lines.append("Validation: OK")
            self.topology_diagnostics_label.setStyleSheet(
                "QLabel { color: #166534; }"
            )
        self.topology_diagnostics_label.setText("\n".join(lines))

    def update_session_board_controls(self) -> None:
        if not hasattr(self, "session_board_type"):
            return
        board_type = str(self.session_board_type.currentData())
        is_chessboard = board_type == "chessboard"
        is_aruco = board_type == "aruco_grid"
        self.session_columns_label.setText(
            self.t("Inner corner columns")
            if is_chessboard
            else self.t("Marker columns")
            if is_aruco
            else self.t("Charuco squares X")
        )
        self.session_rows_label.setText(
            self.t("Inner corner rows")
            if is_chessboard
            else self.t("Marker rows")
            if is_aruco
            else self.t("Charuco squares Y")
        )
        self.session_square_label.setText(
            self.t("Square size")
            if is_chessboard
            else self.t("Marker length")
            if is_aruco
            else self.t("Charuco square length")
        )
        self.session_marker_label.setText(
            self.t("Charuco marker length")
        )
        self.session_marker_length.setEnabled(board_type == "charuco")
        self.session_marker_label.setEnabled(board_type == "charuco")
        self.session_marker_separation.setEnabled(is_aruco)
        self.session_separation_label.setEnabled(is_aruco)
        self.session_dictionary.setEnabled(not is_chessboard)

    def update_wizard_board_controls(self) -> None:
        if not hasattr(self, "wizard_board_type"):
            return
        board_type = str(self.wizard_board_type.currentData())
        is_chessboard = board_type == "chessboard"
        is_aruco = board_type == "aruco_grid"
        self.wizard_columns_label.setText(
            "内角点列数"
            if is_chessboard
            else "Marker 列数"
            if is_aruco
            else "Charuco 方格列数"
        )
        self.wizard_rows_label.setText(
            "内角点行数"
            if is_chessboard
            else "Marker 行数"
            if is_aruco
            else "Charuco 方格行数"
        )
        self.wizard_square_label.setText(
            "方格边长"
            if is_chessboard
            else "Marker 边长"
            if is_aruco
            else "Charuco 方格边长"
        )
        self.wizard_marker_length.setEnabled(board_type == "charuco")
        self.wizard_marker_label.setEnabled(board_type == "charuco")
        self.wizard_marker_separation.setEnabled(is_aruco)
        self.wizard_separation_label.setEnabled(is_aruco)
        self.wizard_dictionary.setEnabled(not is_chessboard)
        self.wizard_board_hint.setText(
            "实验室棋盘默认：总格 12×9，对应内角点 11×8，"
            "方格边长 25mm。请按实物核对后再确认。"
            if is_chessboard
            else "请按标定板实物填写行列、物理尺寸和字典；"
            "程序不会自动猜测这些参数。"
        )

    def sync_wizard_board_to_session_controls(self) -> None:
        board_type = str(self.wizard_board_type.currentData())
        index = self.session_board_type.findData(board_type)
        if index >= 0:
            self.session_board_type.setCurrentIndex(index)
        self.session_board_columns.setValue(
            self.wizard_board_columns.value()
        )
        self.session_board_rows.setValue(self.wizard_board_rows.value())
        self.session_square_length.setValue(
            self.wizard_square_length.value()
        )
        self.session_marker_length.setValue(
            self.wizard_marker_length.value()
        )
        self.session_marker_separation.setValue(
            self.wizard_marker_separation.value()
        )
        dictionary_index = self.session_dictionary.findText(
            self.wizard_dictionary.currentText()
        )
        if dictionary_index >= 0:
            self.session_dictionary.setCurrentIndex(dictionary_index)
        self.session_board_confirmed.setChecked(
            self.wizard_board_confirmed.isChecked()
        )

    def sync_session_to_wizard_controls(
        self,
        session: CalibrationSession,
    ) -> None:
        if not hasattr(self, "wizard_board_type"):
            return
        definition = session.data["board_definition"]
        index = self.wizard_board_type.findData(definition["type"])
        if index >= 0:
            self.wizard_board_type.setCurrentIndex(index)
        if definition["type"] == "chessboard":
            columns = definition["inner_columns"]
            rows = definition["inner_rows"]
            square = definition["square_size_mm"]
        elif definition["type"] == "aruco_grid":
            columns = definition["markers_x"]
            rows = definition["markers_y"]
            square = definition["marker_length_mm"]
            self.wizard_marker_separation.setValue(
                float(definition["marker_separation_mm"])
            )
        else:
            columns = definition["squares_x"]
            rows = definition["squares_y"]
            square = definition["square_length_mm"]
            self.wizard_marker_length.setValue(
                float(definition["marker_length_mm"])
            )
        self.wizard_board_columns.setValue(int(columns))
        self.wizard_board_rows.setValue(int(rows))
        self.wizard_square_length.setValue(float(square))
        dictionary = definition.get("dictionary")
        if dictionary:
            dictionary_index = self.wizard_dictionary.findText(
                str(dictionary)
            )
            if dictionary_index >= 0:
                self.wizard_dictionary.setCurrentIndex(dictionary_index)
        self.wizard_board_confirmed.setChecked(
            bool(session.data.get("board_confirmed"))
        )
        self.wizard_session_label.setText(
            f"当前 session：{session.directory}\n"
            "已加载已有进度；继续采集不会覆盖旧样本。"
        )
        self.update_wizard_board_controls()

    def wizard_create_session(self) -> None:
        if not self.wizard_board_confirmed.isChecked():
            QMessageBox.information(
                self,
                self.t("Calibration Wizard"),
                "请先核对并确认标定板参数。",
            )
            return
        self.sync_wizard_board_to_session_controls()
        if not self.create_calibration_sample_session():
            return
        self.sync_session_to_wizard_controls(self.calibration_session)
        self.set_calibration_wizard_step(2)
        self.refresh_calibration_wizard()

    def wizard_load_session(self) -> None:
        if not self.load_calibration_sample_session():
            return
        self.sync_session_to_wizard_controls(self.calibration_session)
        self.resume_calibration_wizard()

    def set_calibration_wizard_step(self, index: int) -> None:
        if not hasattr(self, "calibration_wizard_stack"):
            return
        index = min(4, max(0, int(index)))
        self.calibration_wizard_stack.setCurrentIndex(index)
        titles = (
            "步骤 1/5：准备设备",
            "步骤 2/5：确认标定板",
            "步骤 3/5：采集单相机画面",
            "步骤 4/5：采集相邻相机 Pair",
            "步骤 5/5：检查与下一步",
        )
        self.wizard_step_title.setText(titles[index])
        self.wizard_back_button.setEnabled(index > 0)
        self.wizard_next_button.setEnabled(index < 4)
        self.wizard_next_button.setText(self.t("Next"))
        self.refresh_calibration_wizard()
        if index == 3 and self.calibration_session is not None:
            QTimer.singleShot(0, self.wizard_inspect_pair_if_visible)

    def wizard_previous_step(self) -> None:
        self.set_calibration_wizard_step(
            self.calibration_wizard_stack.currentIndex() - 1
        )

    def wizard_next_step(self) -> None:
        step = self.calibration_wizard_stack.currentIndex()
        if step == 4:
            self.request_b2_candidate_solve()
            return
        if step == 0 and not all(
            checkbox.isChecked()
            for checkbox in (
                self.wizard_camera_fixed_check,
                self.wizard_resolution_check,
                self.wizard_board_ready_check,
            )
        ):
            QMessageBox.information(
                self,
                self.t("Calibration Wizard"),
                "请完成并勾选三项现场准备检查。",
            )
            return
        if step == 1 and (
            self.calibration_session is None
            or not self.calibration_session.data.get("board_confirmed")
        ):
            QMessageBox.information(
                self,
                self.t("Calibration Wizard"),
                "请确认标定板参数并创建或加载 session。",
            )
            return
        if step == 2 and not self._wizard_intrinsics_minimum_met():
            QMessageBox.information(
                self,
                self.t("Calibration Wizard"),
                "三台相机都达到至少 20 张合格样本后，才能进入 Pair 采集。",
            )
            return
        if step == 3 and not self._wizard_pairs_minimum_met():
            QMessageBox.information(
                self,
                self.t("Calibration Wizard"),
                "两个 Pair 都达到至少 15 组合格样本后，才能进入最终检查。",
            )
            return
        self.set_calibration_wizard_step(step + 1)

    def resume_calibration_wizard(self) -> None:
        if self.calibration_session is None:
            self.set_calibration_wizard_step(1)
        elif not self._wizard_intrinsics_minimum_met():
            self.set_calibration_wizard_step(2)
        elif not self._wizard_pairs_minimum_met():
            self.set_calibration_wizard_step(3)
        else:
            self.set_calibration_wizard_step(4)

    def _wizard_intrinsics_minimum_met(self) -> bool:
        if self.calibration_session is None:
            return False
        summary = self.calibration_session.summary()["intrinsics"]
        return all(
            summary[camera]["accepted"] >= MINIMUM_INTRINSIC_SAMPLES
            for camera in ("front_left", "front", "front_right")
        )

    def _wizard_pairs_minimum_met(self) -> bool:
        if self.calibration_session is None:
            return False
        summary = self.calibration_session.summary()["stereo_pairs"]
        return all(
            summary[pair]["accepted"] >= MINIMUM_PAIR_SAMPLES
            for pair in ("front_left__front", "front__front_right")
        )

    def _first_incomplete_intrinsic_index(self) -> int | None:
        if self.calibration_session is None:
            return 0
        summary = self.calibration_session.summary()["intrinsics"]
        for index, camera in enumerate(
            ("front_left", "front", "front_right")
        ):
            if summary[camera]["accepted"] < MINIMUM_INTRINSIC_SAMPLES:
                return index
        return None

    def _first_incomplete_pair_index(self) -> int | None:
        if self.calibration_session is None:
            return 0
        summary = self.calibration_session.summary()["stereo_pairs"]
        for index, pair in enumerate(
            ("front_left__front", "front__front_right")
        ):
            if summary[pair]["accepted"] < MINIMUM_PAIR_SAMPLES:
                return index
        return None

    def wizard_capture_intrinsic(self) -> None:
        if self.calibration_session is None:
            QMessageBox.information(
                self,
                self.t("Calibration Wizard"),
                "请先创建或加载 session。",
            )
            return
        selected_index = self.wizard_intrinsic_camera.currentIndex()
        first_incomplete = self._first_incomplete_intrinsic_index()
        if (
            first_incomplete is not None
            and selected_index > first_incomplete
        ):
            QMessageBox.information(
                self,
                self.t("Calibration Wizard"),
                "请先完成前一台相机的 20 张最低样本。",
            )
            return
        camera = str(self.wizard_intrinsic_camera.currentData())
        index = self.session_capture_target.findData(
            f"intrinsic:{camera}"
        )
        if index >= 0:
            self.session_capture_target.setCurrentIndex(index)
        self.capture_calibration_session_sample()
        self.refresh_calibration_wizard()

    def wizard_advance_intrinsic_camera(self) -> None:
        if self.calibration_session is None:
            return
        camera = str(self.wizard_intrinsic_camera.currentData())
        accepted = self.calibration_session.summary()["intrinsics"][camera][
            "accepted"
        ]
        if accepted < MINIMUM_INTRINSIC_SAMPLES:
            QMessageBox.information(
                self,
                self.t("Calibration Wizard"),
                f"{camera} 还需要 "
                f"{MINIMUM_INTRINSIC_SAMPLES - accepted} 张合格样本。",
            )
            return
        self.wizard_intrinsic_camera.setCurrentIndex(
            min(2, self.wizard_intrinsic_camera.currentIndex() + 1)
        )

    def wizard_inspect_pair(self) -> None:
        if self.calibration_session is None:
            QMessageBox.information(
                self,
                self.t("Calibration Wizard"),
                "请先创建或加载 session。",
            )
            return
        frames, timestamps = self._current_session_frames()
        pair_key = str(self.wizard_pair.currentData())
        left, right = pair_key.split("__", 1)
        missing = [key for key in (left, right) if key not in frames]
        if missing:
            self.wizard_pair_detection_status.setText(
                f"缺少当前画面：{', '.join(missing)}"
            )
            return
        try:
            inspection = self.calibration_session.inspect_pair(
                left,
                right,
                frames[left],
                frames[right],
                timestamps.get(left),
                timestamps.get(right),
            )
        except Exception as exc:
            self.wizard_pair_detection_status.setText(str(exc))
            return
        self.wizard_pair_detection_status.setText(
            f"{left}："
            f"{'已识别' if inspection['left_detected'] else '未识别'}"
            f"（{inspection['left_point_count']} 点）；"
            f"{right}："
            f"{'已识别' if inspection['right_detected'] else '未识别'}"
            f"（{inspection['right_point_count']} 点）\n"
            f"共同点：{inspection['common_point_count']} / "
            f"{inspection['minimum_common_points']}，"
            f"{'足够' if inspection['common_points_sufficient'] else '不足'}；"
            f"{inspection['time_status']}"
        )

    def wizard_inspect_pair_if_visible(self) -> None:
        if (
            self.calibration_session is not None
            and hasattr(self, "calibration_wizard_stack")
            and self.calibration_wizard_stack.currentIndex() == 3
        ):
            self.wizard_inspect_pair()

    def wizard_capture_pair(self) -> None:
        if self.calibration_session is None:
            QMessageBox.information(
                self,
                self.t("Calibration Wizard"),
                "请先创建或加载 session。",
            )
            return
        selected_index = self.wizard_pair.currentIndex()
        first_incomplete = self._first_incomplete_pair_index()
        if (
            first_incomplete is not None
            and selected_index > first_incomplete
        ):
            QMessageBox.information(
                self,
                self.t("Calibration Wizard"),
                "请先完成 front_left ↔ front 的 15 组最低样本。",
            )
            return
        if not self.wizard_pair_board_confirmed.isChecked():
            QMessageBox.information(
                self,
                self.t("Calibration Wizard"),
                "请确认两路画面中是同一块物理标定板。",
            )
            return
        pair_key = str(self.wizard_pair.currentData())
        index = self.session_capture_target.findData(f"pair:{pair_key}")
        if index >= 0:
            self.session_capture_target.setCurrentIndex(index)
        self.session_physical_board_id.setText(
            self.wizard_pair_board_id.text()
        )
        self.session_pair_board_confirmed.setChecked(True)
        self.capture_calibration_session_sample()
        self.wizard_inspect_pair()
        self.refresh_calibration_wizard()

    def wizard_advance_pair(self) -> None:
        if self.calibration_session is None:
            return
        pair_key = str(self.wizard_pair.currentData())
        accepted = self.calibration_session.summary()["stereo_pairs"][
            pair_key
        ]["accepted"]
        if accepted < MINIMUM_PAIR_SAMPLES:
            QMessageBox.information(
                self,
                self.t("Calibration Wizard"),
                f"当前 Pair 还需要 "
                f"{MINIMUM_PAIR_SAMPLES - accepted} 组合格样本。",
            )
            return
        self.wizard_pair.setCurrentIndex(
            min(1, self.wizard_pair.currentIndex() + 1)
        )

    def refresh_calibration_wizard(self) -> None:
        if not hasattr(self, "wizard_intrinsic_progress"):
            return
        if self.calibration_session is None:
            self.wizard_session_label.setText("尚未创建或加载采集会话")
            self.wizard_intrinsic_progress.setPlainText(
                "请先在步骤 2 创建或加载 session。"
            )
            self.wizard_pair_progress.setPlainText(
                "请先完成三台相机的单独采集。"
            )
            self.wizard_readiness_label.setText("暂不建议求解")
            self.wizard_b2_mode_label.setText(
                "尚无 session，B-2 入口不可用。"
            )
            if self.calibration_wizard_stack.currentIndex() == 4:
                self.wizard_next_button.setText("暂不可进入 B-2")
                self.wizard_next_button.setEnabled(False)
            self.wizard_review_summary.setPlainText(
                "当前没有可检查的采集会话。"
            )
            self.wizard_details_text.clear()
            return
        session = self.calibration_session
        summary = session.summary()
        intrinsic_lines = []
        for camera in ("front_left", "front", "front_right"):
            item = summary["intrinsics"][camera]
            missing = "、".join(item["missing_coverage_zones"][:5]) or "无"
            intrinsic_lines.append(
                f"{camera}：已接受 {item['accepted']} / 目标 {item['target']}，"
                f"最低 {MINIMUM_INTRINSIC_SAMPLES}；拒绝 {item['rejected']}\n"
                f"  覆盖不足：{missing}\n"
                f"  最近拒绝：{item['last_rejection_reason'] or '无'}\n"
                f"  建议：{rejection_advice(item['last_rejection_reason'])}"
            )
        self.wizard_intrinsic_progress.setPlainText(
            "\n\n".join(intrinsic_lines)
        )
        first_intrinsic = self._first_incomplete_intrinsic_index()
        if first_intrinsic is not None:
            self.wizard_intrinsic_camera.setCurrentIndex(first_intrinsic)

        pair_lines = []
        for pair_key in ("front_left__front", "front__front_right"):
            item = summary["stereo_pairs"][pair_key]
            display = pair_key.replace("__", " ↔ ")
            missing = "、".join(item["missing_coverage_zones"][:5]) or "无"
            pair_lines.append(
                f"{display}：已接受 {item['accepted']} / 目标 {item['target']}，"
                f"最低 {MINIMUM_PAIR_SAMPLES}；拒绝 {item['rejected']}\n"
                f"  位置/距离覆盖不足：{missing}\n"
                f"  最近拒绝：{item['last_rejection_reason'] or '无'}\n"
                f"  建议：{rejection_advice(item['last_rejection_reason'])}"
            )
        self.wizard_pair_progress.setPlainText("\n\n".join(pair_lines))
        first_pair = self._first_incomplete_pair_index()
        if first_pair is not None:
            self.wizard_pair.setCurrentIndex(first_pair)

        readiness = session.readiness()
        b2_access = session.b2_candidate_access()
        color = {
            "ready": "#166534",
            "try_with_risk": "#a16207",
            "not_recommended": "#b91c1c",
        }[readiness["status"]]
        self.wizard_readiness_label.setStyleSheet(
            f"QLabel {{ font-size: 16px; font-weight: 700; "
            f"padding: 8px; color: {color}; }}"
        )
        self.wizard_readiness_label.setText(readiness["label"])
        review_lines = [
            f"Session：{session.directory}",
            f"标定板参数：{'已确认' if session.data.get('board_confirmed') else '未确认'}",
            f"分辨率：{session.data.get('resolution')}",
        ]
        review_lines.extend(
            f"阻塞：{reason}\n  下一步：{rejection_advice(reason)}"
            for reason in readiness["blocking_reasons"]
        )
        review_lines.extend(
            f"风险：{reason}\n  下一步：{rejection_advice(reason)}"
            for reason in readiness["quality_issues"]
        )
        if readiness["status"] == "ready":
            review_lines.append(
                "数量与基础质量门控均满足，可以在 B-2 中生成候选标定。"
            )
            mode_text = (
                "正式候选模式：可进入 B-2；候选通过复核后才可申请应用。"
            )
            entry_text = "进入 B-2 候选求解"
        elif readiness["status"] == "try_with_risk":
            review_lines.extend(
                (
                    "[experimental] 仅允许生成候选与诊断报告，"
                    "禁止应用候选标定。",
                    "请补充 Pair 的位置、距离和倾角覆盖，"
                    "通过正式质量门槛后再申请应用。",
                )
            )
            mode_text = (
                "[experimental] 实验性候选求解（仅报告）："
                "可查看候选内外参、RMS、异常样本、Pair-only 与"
                "候选全景预览；禁止应用。"
            )
            entry_text = "实验性候选求解（仅报告）"
        else:
            mode_text = "当前样本不满足求解前提，B-2 入口已锁定。"
            entry_text = "暂不可进入 B-2"
        self.wizard_b2_mode_label.setText(mode_text)
        if self.calibration_wizard_stack.currentIndex() == 4:
            self.wizard_next_button.setText(entry_text)
            self.wizard_next_button.setEnabled(b2_access["can_enter"])
        self.wizard_review_summary.setPlainText("\n".join(review_lines))
        self.wizard_details_text.setPlainText(
            self.calibration_session_status_text()
        )
        self.update_preview_status_summary()

    def b2_candidate_request_context(self) -> dict[str, Any] | None:
        """Build the stable hand-off contract for the future B-2 module."""
        if self.calibration_session is None:
            return None
        access = self.calibration_session.b2_candidate_access()
        return {
            **access,
            "session_directory": str(
                self.calibration_session.directory.resolve()
            ),
            "topology": self.calibration_session.data.get("topology"),
            "resolution": list(
                self.calibration_session.data.get("resolution", [])
            ),
        }

    def set_b2_candidate_solver_launcher(
        self,
        launcher: Callable[[dict[str, Any]], None] | None,
    ) -> None:
        """Register the future B-2 UI/module without coupling it to capture."""
        self.b2_candidate_solver_launcher = launcher

    def request_b2_candidate_solve(self) -> None:
        context = self.b2_candidate_request_context()
        if context is None or not context["can_enter"]:
            QMessageBox.information(
                self,
                self.t("Calibration Wizard"),
                "当前采样结论为“暂不建议求解”，不能进入 B-2。\n\n"
                "发生了什么：标定板确认、分辨率或最低样本数量尚未满足。\n"
                "下一步：查看本页的“阻塞”和“下一步”提示，补拍后再检查。",
            )
            return
        if self.b2_candidate_solver_launcher is not None:
            self.b2_candidate_solver_launcher(context)
            return
        self.run_b2_candidate_solver(context)

    def run_b2_candidate_solver(self, context: dict[str, Any]) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            output_dir = generate_calibration_candidate(
                context["session_directory"]
            )
        except Exception as exc:
            self.log(
                f"B-2 candidate solve failed: {type(exc).__name__}: {exc}",
                level="ERROR",
            )
            QMessageBox.critical(
                self,
                self.t("Calibration Wizard"),
                f"实验性候选求解失败：\n{exc}\n\n"
                "可能原因：某台相机有效样本不足、某个 Pair 无法求出稳定关系，"
                "或候选输出文件不完整。\n"
                "下一步：保留当前 session，查看日志与拒绝原因，补拍后重试；"
                "正式 calibration.yaml 未被应用。",
            )
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.log(f"B-2 experimental candidate saved: {output_dir}")
        self.live_candidate_directory = output_dir
        self._candidate_ui_cache_path = None
        self._candidate_ui_cache = {}
        self.set_live_stitch_mode("candidate")
        dialog = CalibrationCandidateDialog(output_dir, self)
        dialog.exec()

    def calibration_session_status_text(self) -> str:
        if self.calibration_session is None:
            return ""
        lines = [
            f"Board: {self.calibration_session.data['board_definition']}",
            f"Quality gates: {self.calibration_session.data['quality_gate']}",
        ]
        for group_name in ("intrinsics", "stereo_pairs"):
            for key, group in self.calibration_session.data[
                group_name
            ].items():
                time_risks = sum(
                    1
                    for sample in group.get("samples", [])
                    if "time delta" in " ".join(
                        sample.get("rejection_reasons", [])
                    ).lower()
                    or "timestamps unavailable" in " ".join(
                        sample.get("warnings", [])
                    ).lower()
                )
                lines.append(
                    f"{group_name}/{key}: total={len(group.get('samples', []))}, "
                    f"time_risks={time_risks}"
                )
        return "\n".join(lines)

    def current_session_board_definition(self) -> dict[str, Any]:
        board_type = str(self.session_board_type.currentData())
        if board_type == "chessboard":
            return {
                "type": board_type,
                "inner_columns": int(self.session_board_columns.value()),
                "inner_rows": int(self.session_board_rows.value()),
                "square_size_mm": float(
                    self.session_square_length.value()
                ),
            }
        if board_type == "aruco_grid":
            return {
                "type": board_type,
                "markers_x": int(self.session_board_columns.value()),
                "markers_y": int(self.session_board_rows.value()),
                "marker_length_mm": float(
                    self.session_square_length.value()
                ),
                "marker_separation_mm": float(
                    self.session_marker_separation.value()
                ),
                "dictionary": str(self.session_dictionary.currentText()),
            }
        return {
            "type": "charuco",
            "squares_x": int(self.session_board_columns.value()),
            "squares_y": int(self.session_board_rows.value()),
            "square_length_mm": float(
                self.session_square_length.value()
            ),
            "marker_length_mm": float(
                self.session_marker_length.value()
            ),
            "dictionary": str(self.session_dictionary.currentText()),
        }

    def create_calibration_sample_session(self) -> bool:
        if not self.session_board_confirmed.isChecked():
            QMessageBox.information(
                self,
                self.t("Fisheye Sample Session"),
                "Confirm the measured board definition before creating "
                "a capture session.",
            )
            return False
        profile = self.current_stitch_profile()
        try:
            session = CalibrationSession.create(
                sessions_root=(
                    PROJECT_ROOT / "projects" / "calibration_sessions"
                ),
                topology=str(
                    self.calibration_config.get("stitch_topology", "")
                ),
                resolution=(1920, 1080),
                source_coordinate_space=str(
                    profile.get("source_coordinate_space", "")
                ),
                source_reference_size=list(
                    profile.get("source_reference_size", [])
                ),
                source_contract_version=profile.get(
                    "source_contract_version"
                ),
                board_definition=self.current_session_board_definition(),
                board_confirmed=True,
                quality_gate={
                    "minimum_board_area_ratio": (
                        float(self.session_min_area_percent.value()) / 100.0
                    ),
                    "minimum_blur_variance": float(
                        self.session_blur_threshold.value()
                    ),
                    "maximum_pair_time_delta_seconds": (
                        float(self.session_sync_threshold_ms.value()) / 1000.0
                    ),
                },
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                self.t("Fisheye Sample Session"),
                str(exc),
            )
            return False
        self.calibration_session = session
        self.apply_calibration_session_to_controls(session)
        self.calibration_session_log.setPlainText(
            "Session created. Board definition, resolution and quality gates "
            "are now fixed in session.yaml."
        )
        self.refresh_calibration_session_status()
        self.log(f"Calibration sample session created: {session.directory}")
        return True

    def load_calibration_sample_session(self) -> bool:
        directory = QFileDialog.getExistingDirectory(
            self,
            self.t("Load Session"),
            str(PROJECT_ROOT / "projects" / "calibration_sessions"),
        )
        if not directory:
            return False
        try:
            session = CalibrationSession.load(directory)
        except Exception as exc:
            QMessageBox.warning(
                self,
                self.t("Fisheye Sample Session"),
                str(exc),
            )
            return False
        self.calibration_session = session
        self.apply_calibration_session_to_controls(session)
        self.calibration_session_log.setPlainText(
            "Session loaded. Captures will continue using the immutable "
            "board definition and quality gates stored in session.yaml."
        )
        self.refresh_calibration_session_status()
        self.log(f"Calibration sample session loaded: {session.directory}")
        return True

    def apply_calibration_session_to_controls(
        self,
        session: CalibrationSession,
    ) -> None:
        self.calibration_session_path_label.setText(str(session.directory))
        self.calibration_session_path_label.setToolTip(str(session.directory))
        definition = session.data["board_definition"]
        index = self.session_board_type.findData(definition["type"])
        if index >= 0:
            self.session_board_type.setCurrentIndex(index)
        board_type = definition["type"]
        if board_type == "chessboard":
            columns = definition["inner_columns"]
            rows = definition["inner_rows"]
            square = definition["square_size_mm"]
        elif board_type == "aruco_grid":
            columns = definition["markers_x"]
            rows = definition["markers_y"]
            square = definition["marker_length_mm"]
            self.session_marker_separation.setValue(
                float(definition["marker_separation_mm"])
            )
        else:
            columns = definition["squares_x"]
            rows = definition["squares_y"]
            square = definition["square_length_mm"]
            self.session_marker_length.setValue(
                float(definition["marker_length_mm"])
            )
        self.session_board_columns.setValue(int(columns))
        self.session_board_rows.setValue(int(rows))
        self.session_square_length.setValue(float(square))
        dictionary = definition.get("dictionary")
        if dictionary:
            dictionary_index = self.session_dictionary.findText(
                str(dictionary)
            )
            if dictionary_index >= 0:
                self.session_dictionary.setCurrentIndex(dictionary_index)
        gate = session.data["quality_gate"]
        self.session_min_area_percent.setValue(
            float(gate["minimum_board_area_ratio"]) * 100.0
        )
        self.session_blur_threshold.setValue(
            float(gate["minimum_blur_variance"])
        )
        self.session_sync_threshold_ms.setValue(
            int(
                round(
                    float(gate["maximum_pair_time_delta_seconds"])
                    * 1000.0
                )
            )
        )
        self.session_board_confirmed.setChecked(
            bool(session.data.get("board_confirmed"))
        )
        self.update_session_board_controls()
        self.sync_session_to_wizard_controls(session)

    def update_session_capture_controls(self) -> None:
        if not hasattr(self, "session_capture_target"):
            return
        is_pair = str(
            self.session_capture_target.currentData() or ""
        ).startswith("pair:")
        self.session_physical_board_id.setEnabled(is_pair)
        self.session_pair_board_confirmed.setEnabled(is_pair)

    def _current_session_frames(
        self,
    ) -> tuple[dict[str, np.ndarray], dict[str, float | None]]:
        if self.preview_content_mode == PreviewContentMode.LIVE:
            frames, snapshots = self.stream_manager.latest_frames()
        else:
            frames = {
                key: frame.copy()
                for key, frame in self.frames.items()
                if frame is not None
            }
            snapshots = {}
        timestamps = {
            key: (
                float(snapshot.frame_timestamp)
                if snapshot is not None and snapshot.frame_timestamp > 0
                else None
            )
            for key, snapshot in snapshots.items()
        }
        return frames, timestamps

    def capture_calibration_session_sample(self) -> None:
        if self.calibration_session is None:
            QMessageBox.information(
                self,
                self.t("Fisheye Sample Session"),
                "Create or load a calibration session first.",
            )
            return
        frames, timestamps = self._current_session_frames()
        selection = str(self.session_capture_target.currentData() or "")
        try:
            if selection.startswith("intrinsic:"):
                camera = selection.split(":", 1)[1]
                if camera not in frames:
                    raise ValueError(f"No current raw frame for {camera}.")
                record = self.calibration_session.capture_intrinsic(
                    camera,
                    frames[camera],
                    frame_timestamp=timestamps.get(camera),
                )
            elif selection.startswith("pair:"):
                pair = selection.split(":", 1)[1]
                left, right = pair.split("__", 1)
                missing = [
                    key for key in (left, right) if key not in frames
                ]
                if missing:
                    raise ValueError(
                        f"Missing current raw frames: {', '.join(missing)}"
                    )
                record = self.calibration_session.capture_pair(
                    left,
                    right,
                    frames[left],
                    frames[right],
                    timestamps.get(left),
                    timestamps.get(right),
                    self.session_physical_board_id.text(),
                    self.session_pair_board_confirmed.isChecked(),
                )
            else:
                raise ValueError("Unknown calibration capture target.")
        except BoardDefinitionNotConfirmedError as exc:
            QMessageBox.information(
                self,
                self.t("Fisheye Sample Session"),
                str(exc),
            )
            return
        except Exception as exc:
            self.log(
                f"Calibration session capture failed: "
                f"{exc}\n{traceback.format_exc()}",
                level="ERROR",
            )
            QMessageBox.warning(
                self,
                self.t("Fisheye Sample Session"),
                str(exc),
            )
            return
        reasons = "; ".join(record["rejection_reasons"]) or "none"
        warnings = "; ".join(record.get("warnings", [])) or "none"
        message = (
            f"{record['sample_id']}: "
            f"{'ACCEPTED' if record['accepted'] else 'REJECTED'}\n"
            f"Reasons: {reasons}\nWarnings: {warnings}\n"
            f"下一步建议：{rejection_advice(reasons if reasons != 'none' else '')}"
        )
        self.calibration_session_log.append(message)
        self.refresh_calibration_session_status()
        self.log(f"Calibration session sample: {message}")

    def refresh_calibration_session_status(self) -> None:
        if not hasattr(self, "calibration_session_status_table"):
            return
        table = self.calibration_session_status_table
        rows: list[tuple[str, dict[str, Any]]] = []
        if self.calibration_session is not None:
            summary = self.calibration_session.summary()
            rows.extend(
                (f"{self.t('Intrinsic')}: {key}", value)
                for key, value in summary["intrinsics"].items()
            )
            rows.extend(
                (
                    f"{self.t('Pair')}: {key.replace('__', ' <-> ')}",
                    value,
                )
                for key, value in summary["stereo_pairs"].items()
            )
        table.setRowCount(len(rows))
        for row_index, (name, item) in enumerate(rows):
            missing = item["missing_coverage_zones"]
            missing_text = ", ".join(missing[:4])
            if len(missing) > 4:
                missing_text += f" (+{len(missing) - 4})"
            values = [
                name,
                str(item["accepted"]),
                str(item["target"]),
                str(item["rejected"]),
                missing_text,
                item["last_rejection_reason"],
            ]
            for column, value in enumerate(values):
                table.setItem(
                    row_index,
                    column,
                    QTableWidgetItem(value),
                )
        self.refresh_calibration_wizard()

    def refresh_pairwise_pair_choices(self) -> None:
        if not hasattr(self, "pairwise_pair_combo"):
            return
        selected_name = str(self.pairwise_pair_combo.currentData() or "")
        self.pairwise_pair_combo.clear()
        for pair in pair_definitions(self.calibration_config):
            self.pairwise_pair_combo.addItem(
                f"{pair.left_camera} <-> {pair.right_camera}",
                pair.name,
            )
        if selected_name:
            index = self.pairwise_pair_combo.findData(selected_name)
            if index >= 0:
                self.pairwise_pair_combo.setCurrentIndex(index)

    def selected_pair_definition(self) -> PairDefinition | None:
        selected_name = str(self.pairwise_pair_combo.currentData() or "")
        return next(
            (
                pair
                for pair in pair_definitions(self.calibration_config)
                if pair.name == selected_name
            ),
            None,
        )

    def run_pairwise_candidate_diagnostics(self) -> None:
        snapshots_root = PROJECT_ROOT / "projects" / "calibration_snapshots"
        try:
            results = run_automatic_pairwise_candidates(
                snapshots_root,
                self.calibration_config,
            )
            output_dir = save_pairwise_results(
                PROJECT_ROOT / "projects" / "pairwise_diagnostics",
                results,
            )
        except Exception as exc:
            self.log(
                f"Pairwise candidate diagnostics failed: "
                f"{exc}\n{traceback.format_exc()}",
                level="ERROR",
            )
            QMessageBox.warning(
                self,
                self.t("Pairwise Candidates"),
                str(exc),
            )
            return
        self.pairwise_diagnostic_results = results
        self.render_pairwise_candidate_diagnostics(results, output_dir)
        self.log(
            "Pairwise candidate diagnostics saved: "
            f"{output_dir}; formal profile unchanged."
        )

    def open_manual_pairwise_dialog(self) -> None:
        pair = self.selected_pair_definition()
        if pair is None:
            QMessageBox.information(
                self,
                self.t("Pairwise Candidates"),
                "No pair is selected.",
            )
            return
        try:
            snapshot_dir = latest_snapshot_directory(
                PROJECT_ROOT / "projects" / "calibration_snapshots",
                str(self.calibration_config.get("stitch_topology", "")),
            )
            frames = load_snapshot_frames(snapshot_dir)
            left = frames[pair.left_camera]
            right = frames[pair.right_camera]
        except Exception as exc:
            QMessageBox.warning(
                self,
                self.t("Pairwise Candidates"),
                str(exc),
            )
            return

        dialog = ManualCorrespondenceDialog(
            pair,
            left,
            right,
            self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        result = estimate_pairwise_candidate(
            snapshot_dir,
            self.calibration_config,
            pair,
            manual_left_points=dialog.left_points,
            manual_right_points=dialog.right_points,
        )
        try:
            output_dir = save_pairwise_results(
                PROJECT_ROOT / "projects" / "pairwise_diagnostics",
                [result],
            )
        except Exception as exc:
            self.log(
                f"Manual pairwise candidate save failed: {exc}",
                level="ERROR",
            )
            QMessageBox.warning(
                self,
                self.t("Pairwise Candidates"),
                str(exc),
            )
            return
        self.pairwise_diagnostic_results = [result]
        self.render_pairwise_candidate_diagnostics([result], output_dir)
        self.log(
            "Manual pairwise candidate saved: "
            f"{output_dir}; formal profile unchanged."
        )

    def render_pairwise_candidate_diagnostics(
        self,
        results: list[PairwiseCandidateResult],
        output_dir: Path,
    ) -> None:
        lines = [
            f"Output: {output_dir}",
            "Scope: planar pairwise candidate only; calibration.yaml was not updated.",
        ]
        while self.pairwise_diagnostics_tabs.count():
            widget = self.pairwise_diagnostics_tabs.widget(0)
            self.pairwise_diagnostics_tabs.removeTab(0)
            widget.deleteLater()
        for result in results:
            error_text = result.error or "none"
            reprojection = (
                f"{result.reprojection_error:.3f}px"
                if result.reprojection_error is not None
                else "unavailable"
            )
            lines.append(
                f"{result.pair.left_camera} <-> "
                f"{result.pair.right_camera}: method={result.method}, "
                f"points={result.point_count}, inliers={result.inlier_count}, "
                f"reprojection={reprojection}, error={error_text}"
            )
            for warning in result.warnings:
                lines.append(f"  WARNING: {warning}")
            images = (
                (
                    f"{result.pair.name}: left points",
                    result.left_visualization,
                ),
                (
                    f"{result.pair.name}: right points",
                    result.right_visualization,
                ),
                (
                    f"{result.pair.name}: alpha 50%",
                    result.alpha_overlay,
                ),
                (
                    f"{result.pair.name}: hard cut",
                    result.hard_cut_overlay,
                ),
            )
            for title, image in images:
                view = ImageView(title)
                if image is not None:
                    view.set_image(image, title)
                else:
                    view.set_placeholder(result.error or "No candidate image")
                self.pairwise_diagnostics_tabs.addTab(view, title)
        self.pairwise_diagnostics_summary.setPlainText("\n".join(lines))
        self.diagnostic_output_tabs.setCurrentIndex(
            self.pairwise_output_tab_index
        )

    def refresh_geometry_diagnostics(self) -> None:
        if self.preview_content_mode == PreviewContentMode.LIVE:
            frames, _ = self.stream_manager.latest_frames()
        else:
            frames = {
                key: frame.copy()
                for key, frame in self.frames.items()
                if frame is not None
            }
        active = set(active_topology_camera_keys(self.calibration_config))
        frames = {key: frame for key, frame in frames.items() if key in active}
        max_input_width, use_intrinsics = self.live_stitcher_options()
        pair_mode = str(self.geometry_pair_mode.currentData() or ALPHA_50)
        try:
            result = compute_geometry_diagnostics(
                frames=frames,
                calibration_config=self.calibration_config,
                max_input_width=max_input_width,
                use_intrinsics=use_intrinsics,
                pair_mode=pair_mode,
            )
        except Exception as exc:
            self.log(
                "Geometry diagnostics failed: "
                f"{exc}\n{traceback.format_exc()}",
                level="ERROR",
            )
            QMessageBox.warning(
                self,
                self.t("Geometry Diagnostics"),
                str(exc),
            )
            return
        self.geometry_diagnostic_result = result
        self.render_geometry_diagnostics(result)

    def render_geometry_diagnostics(
        self,
        result: GeometryDiagnosticResult,
    ) -> None:
        metadata = result.metadata
        lines = [
            f"Topology: {metadata.get('topology', '')}",
            f"Camera order: {' -> '.join(metadata.get('camera_order', []))}",
            (
                f"Canvas: {metadata.get('canvas', {}).get('width', 0)} x "
                f"{metadata.get('canvas', {}).get('height', 0)}"
            ),
            f"Runtime stages: {' -> '.join(metadata.get('runtime_stage_order', []))}",
            (
                "Source points: raw-frame pixel coordinates; "
                "editor points: loaded image pixel coordinates."
            ),
            (
                f"Source reference: {metadata.get('source_reference_size')}; "
                f"contract version={metadata.get('source_contract_version')}"
            ),
            f"use_intrinsics={metadata.get('use_intrinsics', False)}",
        ]
        if not metadata.get("use_intrinsics", False):
            lines.append(
                "WARNING: undistortion is not executed. Current output is "
                "fisheye/raw-frame perspective warp; natural global stitching "
                "should not be expected."
            )
        for key in metadata.get("camera_order", []):
            camera = metadata.get("cameras", {}).get(key, {})
            if not camera.get("available"):
                lines.append(f"{key}: no current frame")
                continue
            raw = camera.get("raw_size", [])
            normalized = camera.get("normalized_size", [])
            lines.append(
                f"{key}: raw={raw[0]}x{raw[1]} -> "
                f"normalized={normalized[0]}x{normalized[1]}, "
                f"scale={camera.get('raw_to_normalized_scale', 1.0):.4f}, "
                f"normalized->source factor="
                f"{camera.get('normalized_to_source_factor', 1.0):.4f}"
            )
            lines.append(
                f"  source bounds={camera.get('source_bounds')}, "
                f"coverage={camera.get('source_coverage_fraction')}, "
                f"intrinsics available={camera.get('intrinsics_available')}, "
                f"effective={camera.get('intrinsics_effective')}, "
                f"reference={camera.get('intrinsics_reference_size')}"
            )
            if camera.get("intrinsics_available"):
                lines.append(
                    "  intrinsics scale to raw="
                    f"{camera.get('intrinsics_scale_to_raw')}, "
                    "to normalized="
                    f"{camera.get('intrinsics_scale_to_normalized')}, "
                    "uniform-compatible="
                    f"{camera.get('intrinsics_uniform_scale_compatible')}, "
                    "runtime matrix scaled="
                    f"{camera.get('intrinsics_runtime_matrix_scaled')}"
                )
            if camera.get("intrinsics_warning"):
                lines.append(
                    f"  WARNING: {camera.get('intrinsics_warning')}"
                )
            for warning in camera.get("coordinate_warnings", []):
                lines.append(f"  WARNING: {warning}")
        if result.final_error:
            lines.append(f"Final canvas error: {result.final_error}")
        self.geometry_diagnostics_summary.setPlainText("\n".join(lines))

        while self.geometry_diagnostics_tabs.count():
            widget = self.geometry_diagnostics_tabs.widget(0)
            self.geometry_diagnostics_tabs.removeTab(0)
            widget.deleteLater()
        for key in metadata.get("camera_order", []):
            view = ImageView(f"Warp: {key}")
            if key in result.warped:
                view.set_image(result.warped[key], f"Warp: {key}")
            else:
                view.set_placeholder(f"{key}: no current frame")
            self.geometry_diagnostics_tabs.addTab(view, f"Warp: {key}")
        for pair in metadata.get("pairs", []):
            pair_name = str(pair.get("name", "pair"))
            view = ImageView(f"Pair: {pair_name}")
            if pair_name in result.pair_only:
                view.set_image(
                    result.pair_only[pair_name],
                    f"Pair only: {pair_name}",
                )
            else:
                view.set_placeholder(f"{pair_name}: missing camera frame")
            self.geometry_diagnostics_tabs.addTab(view, f"Pair: {pair_name}")
        final_view = ImageView(self.t("Final Canvas"))
        if result.final_canvas is not None:
            final_view.set_image(
                result.final_canvas,
                self.t("Final Canvas"),
            )
        else:
            final_view.set_placeholder(
                result.final_error or self.t("No frame")
            )
        self.geometry_diagnostics_tabs.addTab(
            final_view,
            self.t("Final Canvas"),
        )
        self.log(
            "Geometry diagnostics refreshed: "
            f"topology={metadata.get('topology', '')}, "
            f"frames={list(result.warped)}, mode={metadata.get('pair_mode')}, "
            f"final_error={result.final_error or 'none'}"
        )

    def populate_seam_table(self) -> None:
        if not hasattr(self, "seam_table"):
            return
        stitch_points = self.current_stitch_points()
        rows = []
        for seam_name, endpoints in stitch_points.items():
            for index, point in enumerate(endpoints[:2]):
                rows.append((seam_name, index, point[0], point[1]))
        self._updating_seam_table = True
        self.seam_table.setRowCount(len(rows))
        for row_index, (seam_name, point_index, x, y) in enumerate(rows):
            values = [seam_name, str(point_index + 1), f"{x:.3f}", f"{y:.3f}", self._seam_role(seam_name)]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column in (0, 1, 4):
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.seam_table.setItem(row_index, column, item)
        self._updating_seam_table = False
        if hasattr(self, "seam_editor"):
            self.seam_editor.set_stitch_points(stitch_points)

    def refresh_seam_editor(self) -> None:
        if not hasattr(self, "seam_editor"):
            return
        stitch_profile = self.current_stitch_profile()
        width = int(stitch_profile.get("canvas", {}).get("width", 2440))
        height = int(stitch_profile.get("canvas", {}).get("height", 1800))
        self.seam_editor.set_canvas(self.canvas, width, height)
        self.seam_editor.set_overlaps(stitch_profile.get("overlaps", []))
        self.seam_editor.set_stitch_points(self.current_stitch_points())
        self.populate_seam_table()

    def center_seams_in_overlaps(self) -> None:
        profile = self.current_stitch_profile()
        height = float(profile.get("canvas", {}).get("height", 0))
        self._updating_seam_table = True
        try:
            for overlap in profile.get("overlaps", []):
                x_range = overlap.get("x_range", [])
                seam_name = str(overlap.get("seam", ""))
                if len(x_range) != 2 or not seam_name:
                    continue
                center_x = (float(x_range[0]) + float(x_range[1])) / 2.0
                for point_index, y in ((0, 0.0), (1, height)):
                    row = self._seam_table_row(seam_name, point_index)
                    if row is None:
                        continue
                    self.seam_table.item(row, 2).setText(f"{center_x:.3f}")
                    self.seam_table.item(row, 3).setText(f"{y:.3f}")
        finally:
            self._updating_seam_table = False
        self.seam_editor.set_stitch_points(
            self._stitch_points_from_table()
        )
        self.log(
            "Seams centered in declared overlaps; changes are not saved yet."
        )

    def on_seam_point_moved(self, seam_name: str, point_index: int, x: float, y: float) -> None:
        if self._updating_seam_table:
            return
        row = self._seam_table_row(seam_name, point_index)
        if row is not None:
            self._updating_seam_table = True
            self.seam_table.item(row, 2).setText(f"{x:.3f}")
            self.seam_table.item(row, 3).setText(f"{y:.3f}")
            self._updating_seam_table = False
        self.statusBar().showMessage(f"{seam_name} point {point_index + 1}: {x:.1f}, {y:.1f}")

    def on_seam_table_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_seam_table or item.column() not in (2, 3):
            return
        try:
            points = self._stitch_points_from_table()
        except Exception:
            return
        self.seam_editor.set_stitch_points(points)

    def save_seam_points(self) -> None:
        try:
            stitch_points = self._stitch_points_from_table()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid seam points", str(exc))
            return
        profile = self.current_stitch_profile()
        validation_errors = validate_overlap_seams(
            profile,
            stitch_points=stitch_points,
        )
        if validation_errors:
            message = (
                f"Topology={self.calibration_config.get('stitch_topology', '')}\n"
                + "\n".join(validation_errors)
            )
            self.log(
                f"Seam validation failed:\n{message}",
                level="ERROR",
            )
            QMessageBox.warning(
                self,
                self.t("Seam validation failed"),
                message,
            )
            return
        previous_stitch_points = profile.get("stitch_points", {})
        previous_feather = self.current_feather_width()
        feather_width = int(self.feather_width.value())
        profile["stitch_points"] = stitch_points
        profile.setdefault("composition", {})["feather_width"] = feather_width
        for overlap in profile.get("overlaps", []):
            overlap["feather_width"] = feather_width
        origin = profile.setdefault("calibration_origin", {})
        if stitch_points != previous_stitch_points:
            origin["seams"] = "field_adjusted"
        if feather_width != previous_feather:
            origin["feather"] = "field_adjusted"
        if not self.persist_calibration_config():
            return
        self.stitcher = self.create_stitcher()
        self.bump_preview_session()
        self.configure_live_stitch_processor()
        self.refresh_seam_editor()
        self.update_topology_diagnostics()
        self.log("Saved seam points to calibration.yaml.")

    def _stitch_points_from_table(self) -> dict[str, list[list[float]]]:
        points: dict[str, list[list[float]]] = {}
        for row in range(self.seam_table.rowCount()):
            seam_name = self.seam_table.item(row, 0).text()
            point_index = int(self.seam_table.item(row, 1).text()) - 1
            x = float(self.seam_table.item(row, 2).text())
            y = float(self.seam_table.item(row, 3).text())
            points.setdefault(seam_name, [[0.0, 0.0], [0.0, 0.0]])
            points[seam_name][point_index] = [x, y]
        return points

    def _seam_table_row(self, seam_name: str, point_index: int) -> int | None:
        for row in range(self.seam_table.rowCount()):
            if (
                self.seam_table.item(row, 0).text() == seam_name
                and int(self.seam_table.item(row, 1).text()) - 1 == point_index
            ):
                return row
        return None

    def _seam_role(self, seam_name: str) -> str:
        roles = {
            "front_right": "front-right seam",
            "right_behind": "right-rear seam",
            "behind_left": "rear-left seam",
            "left_front": "left-front seam",
        }
        return roles.get(seam_name, "custom seam")

    def choose_calibration_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Choose calibration image folder", str(self.calibration_folder))
        if not directory:
            return
        self.calibration_folder = Path(directory)
        self.calibration_folder_label.setText(str(self.calibration_folder))
        self.scan_calibration_folder()

    def capture_calibration_frame(self) -> None:
        camera_key = self.calibration_camera.currentText()
        frame = self.frames.get(camera_key)
        if frame is None:
            QMessageBox.information(
                self,
                self.t("No frame"),
                self.t("Load images or start live preview first, then capture the selected camera frame."),
            )
            return
        self.calibration_folder.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.calibration_folder / f"{camera_key}_{timestamp}.jpg"
        save_image(path, frame)
        self.log(f"Captured calibration frame: {path}")
        self.scan_calibration_folder()

    def save_current_calibration_snapshot(self) -> None:
        camera_order = active_topology_camera_keys(self.calibration_config)
        captured_at = time.time()
        frame_ages: dict[str, float | None] = {
            key: None for key in camera_order
        }

        if self.preview_content_mode == PreviewContentMode.LIVE:
            latest_frames, snapshots = self.stream_manager.latest_frames()
            frames = {
                key: frame
                for key, frame in latest_frames.items()
                if key in camera_order
            }
            for key in camera_order:
                snapshot = snapshots.get(key)
                if snapshot is not None and snapshot.frame_timestamp > 0:
                    frame_ages[key] = max(
                        0.0,
                        captured_at - snapshot.frame_timestamp,
                    )
        else:
            frames = {
                key: frame.copy()
                for key, frame in self.frames.items()
                if key in camera_order and frame is not None
            }

        profile = self.current_stitch_profile()
        width = int(profile.get("canvas", {}).get("width", 2440))
        height = int(profile.get("canvas", {}).get("height", 1800))
        processing_error = ""
        canvas: np.ndarray | None = None
        if frames:
            try:
                _, canvas = self.stitcher.process(frames)
            except Exception as exc:
                processing_error = str(exc)
                self.log(
                    "Calibration snapshot stitch failed: "
                    f"{exc}\n{traceback.format_exc()}",
                    level="ERROR",
                )
        if canvas is None:
            if (
                self.canvas is not None
                and self.canvas.shape[:2] == (height, width)
            ):
                canvas = self.canvas.copy()
            else:
                canvas = np.zeros((height, width, 3), dtype=np.uint8)

        snapshots_root = PROJECT_ROOT / "projects" / "calibration_snapshots"
        try:
            snapshot_dir = save_calibration_snapshot(
                snapshots_root=snapshots_root,
                frames=frames,
                canvas=canvas,
                calibration_config=self.calibration_config,
                frame_ages_seconds=frame_ages,
                processing_error=processing_error,
            )
        except Exception as exc:
            self.log(
                "Calibration snapshot save failed: "
                f"{exc}\n{traceback.format_exc()}",
                level="ERROR",
            )
            QMessageBox.warning(
                self,
                self.t("Calibration snapshot failed"),
                str(exc),
            )
            return

        message = self.t(
            "Calibration snapshot saved: {path}",
            path=snapshot_dir,
        )
        self.statusBar().showMessage(message)
        self.log(message)

    def scan_calibration_folder(self) -> None:
        folder = self.calibration_folder
        folder.mkdir(parents=True, exist_ok=True)
        self.calibration_folder_label.setText(str(folder))
        spec = spec_from_config(self.calibration_config)
        files = [
            path for path in sorted(folder.glob("*"))
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        self.calibration_image_table.setRowCount(len(files))
        ok_count = 0
        for row, path in enumerate(files):
            image = cv2.imread(str(path))
            readable = image is not None
            chessboard = False
            corners = 0
            if image is not None:
                _, chessboard, corners = draw_chessboard_detection(image, spec)
                ok_count += 1 if chessboard else 0
            camera = self._camera_from_filename(path.name)
            values = [
                path.name,
                camera,
                "yes" if readable else "no",
                "yes" if chessboard else "no",
                str(corners),
            ]
            for column, value in enumerate(values):
                self.calibration_image_table.setItem(row, column, QTableWidgetItem(value))
        self.log(f"Scanned {len(files)} calibration images; chessboard OK: {ok_count}.")

    def _camera_from_filename(self, filename: str) -> str:
        lower = filename.lower()
        for key in CAMERA_KEYS:
            if key in lower:
                return key
        return self.calibration_camera.currentText()

    def log(self, message: str, level: str = "INFO") -> None:
        safe_message = redact_log_message(message)
        normalized_level = level.upper()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        display_line = f"{timestamp} [{normalized_level}] {safe_message}"
        log_method = getattr(
            self.app_logger,
            normalized_level.lower(),
            self.app_logger.info,
        )
        log_method(safe_message)
        if hasattr(self, "application_log"):
            self.application_log.moveCursor(QTextCursor.MoveOperation.End)
            self.application_log.insertPlainText(display_line + "\n")
        if hasattr(self, "calibration_log"):
            self.calibration_log.moveCursor(QTextCursor.MoveOperation.End)
            self.calibration_log.insertPlainText(display_line + "\n")
        self.statusBar().showMessage(safe_message.splitlines()[0])


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


