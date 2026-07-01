"""PySide6 user interface for DeepShark View Studio."""

from __future__ import annotations

import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from deep_shark_studio.calibration import (
    calibrate_camera_from_images,
    draw_chessboard_detection,
    spec_from_config,
    undistort_image,
    write_calibration_report,
)
from deep_shark_studio.calibration_snapshot import (
    save_calibration_snapshot,
    topology_diagnostics,
)
from deep_shark_studio.config import (
    CONFIG_DIR,
    PROJECT_ROOT,
    ConfigConflictError,
    config_revision,
    load_config,
    save_config,
)
from deep_shark_studio.project import backup_configs, export_runtime_config, load_project, save_project
from deep_shark_studio.stream_manager import (
    CameraStreamConfig,
    CameraStreamManager,
    STREAM_CONNECTING,
    STREAM_FAILED,
    STREAM_IDLE,
    STREAM_LIVE,
)
from deep_shark_studio.stitch_processing import StitchProcessingManager
from deep_shark_studio.stitcher import SurroundStitcher, load_images_from_directory, save_image
from deep_shark_studio.topology import (
    active_stitch_profile,
    active_topology_camera_keys,
)


CAMERA_KEYS = ["front_left", "front_right", "front", "behind", "left", "right"]
SOURCE_TYPES = ["image_dir", "usb", "video_file", "rtsp"]

ZH_CN = {
    "Language": "语言",
    "English": "英文",
    "Chinese": "中文",
    "Ready": "就绪",
    "Realtime Preview": "实时预览",
    "Camera Config": "相机配置",
    "Calibration": "标定与调参",
    "Project": "项目管理",
    "Choose Image Directory": "选择图片目录",
    "Load Images": "加载图片",
    "Start Live": "开始实时预览",
    "Stop": "停止",
    "Save Results": "保存结果",
    "Multi-camera View": "多路视图",
    "Stitched View": "拼接视图",
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
    "Refresh Seam Canvas": "刷新拼接缝画布",
    "Save Seam Points": "保存拼接缝",
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
    "No project file loaded. Current configs are stored in configs/*.yaml.": "尚未加载项目文件。当前配置保存在 configs/*.yaml。",
    "Save Project As": "项目另存为",
    "Open Project": "打开项目",
    "Backup Current Configs": "备份当前配置",
    "Export Runtime Config": "导出运行配置",
    "Project files bundle calibration.yaml, cameras.yaml, and network.yaml into one portable .dsvs.yaml file.\n\nUse backups before large calibration or seam edits. Runtime export is the compact configuration intended for a future service/QGC bridge.": "项目文件会把 calibration.yaml、cameras.yaml 和 network.yaml 打包成一个便携的 .dsvs.yaml 文件。\n\n大幅修改标定或拼接缝前建议先备份。运行配置导出用于后续独立服务或 QGC 桥接。",
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
}
def i18n(language: str, text: str, **kwargs: Any) -> str:
    translated = ZH_CN.get(text, text) if language == "zh_CN" else text
    return translated.format(**kwargs) if kwargs else translated


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


class ImageView(QLabel):
    """Simple image display surface."""

    doubleClicked = Signal(str)

    def __init__(self, title: str, camera_id: str = ""):
        super().__init__(title)
        self.placeholder = title
        self.camera_id = camera_id
        self.overlay_text = ""
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(220, 150)
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
        self.setMinimumSize(560, 420)
        self.setMouseTracking(True)
        self.setStyleSheet(
            "QLabel { background: #101820; color: #d7e0ea; border: 1px solid #334155; padding: 4px; }"
        )
        self._image: np.ndarray | None = None
        self._pixmap: QPixmap | None = None
        self._points: list[list[float]] = []
        self._drag_index: int | None = None
        self._display_rect = QRectF()
        self._colors = [
            QColor("#f97316"),
            QColor("#22c55e"),
            QColor("#38bdf8"),
            QColor("#f43f5e"),
        ]

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


class SeamEditorView(QLabel):
    """Canvas-coordinate seam editor with draggable seam endpoints."""

    seamPointMoved = Signal(str, int, float, float)

    def __init__(self):
        super().__init__("Drag seam endpoints on the surround canvas.")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(560, 420)
        self.setMouseTracking(True)
        self.setStyleSheet(
            "QLabel { background: #0b1220; color: #d7e0ea; border: 1px solid #334155; padding: 4px; }"
        )
        self._image: np.ndarray | None = None
        self._pixmap: QPixmap | None = None
        self._canvas_size = (2440, 1800)
        self._stitch_points: dict[str, list[list[float]]] = {}
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

        self.source_type = QComboBox()
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
        self.resize(1500, 920)

        self.input_dir = PROJECT_ROOT / "samples" / "input"
        self.output_dir = PROJECT_ROOT / "samples" / "output"
        self.calibration_config = load_config("calibration.yaml")
        self.camera_config = load_config("cameras.yaml")
        self._calibration_config_revision = config_revision("calibration.yaml")
        self._camera_config_revision = config_revision("cameras.yaml")
        self.performance_config = self.camera_config.get("performance", {})
        self.stitcher = self.create_stitcher()
        self.stitch_processor = StitchProcessingManager()
        self.language = str(self.camera_config.get("language", "en"))

        self.frames: dict[str, np.ndarray] = {}
        self.warped: dict[str, np.ndarray] = {}
        self.canvas: np.ndarray | None = None
        self.stream_manager = CameraStreamManager()
        self.stream_snapshots = {}
        self.stream_error_log_counts: dict[str, int] = {}
        self.frame_counts: dict[str, int] = {}
        self.last_tick = time.time()
        self.last_process_time = 0.0
        self.last_preview_time = 0.0
        self.last_health_time = 0.0
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

        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(15)
        self.preview_timer.timeout.connect(self.update_live_preview)

        self.camera_rows: dict[str, CameraRow] = {}
        self.camera_views: dict[str, ImageView] = {}
        self.warped_views: dict[str, ImageView] = {}

        self._build_ui()
        self.refresh_camera_count()
        self.statusBar().showMessage(self.t("Ready"))

    def t(self, text: str, **kwargs: Any) -> str:
        return i18n(self.language, text, **kwargs)

    def create_stitcher(self) -> SurroundStitcher:
        max_width = int(self.performance_config.get("max_input_width", 960))
        if max_width <= 0:
            max_width = None
        use_intrinsics = bool(self.performance_config.get("use_intrinsics", False))
        return SurroundStitcher(self.calibration_config, max_input_width=max_width, use_intrinsics=use_intrinsics)

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
        self.stitch_processor.configure(
            self.preview_session_id,
            self.calibration_config,
            max_width,
            use_intrinsics,
        )

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
        self.language_combo = QComboBox()
        self.language_combo.addItem(self.t("English"), "en")
        self.language_combo.addItem(self.t("Chinese"), "zh_CN")
        self.language_combo.setCurrentIndex(1 if self.language == "zh_CN" else 0)
        self.language_combo.currentIndexChanged.connect(self.change_language)
        language_bar.addWidget(self.language_combo)
        central_layout.addLayout(language_bar)

        self.root_tabs = QTabWidget()
        self.root_tabs.addTab(self._build_preview_workspace(), self.t("Realtime Preview"))
        self.root_tabs.addTab(self._build_camera_config_workspace(), self.t("Camera Config"))
        self.root_tabs.addTab(self._build_calibration_workspace(), self.t("Calibration"))
        self.root_tabs.addTab(self._build_project_workspace(), self.t("Project"))
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

        toolbar = QHBoxLayout()
        self.input_path_label = QLabel(str(self.input_dir))
        self.input_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
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
        toolbar.addWidget(self.choose_input_dir_button)
        toolbar.addWidget(self.load_images_button)
        toolbar.addWidget(self.start_live_button)
        toolbar.addWidget(self.stop_live_button)
        toolbar.addWidget(self.save_results_button)
        toolbar.addWidget(self.grid_view_button)
        toolbar.addWidget(self.stitched_view_button)
        toolbar.addWidget(self.back_to_grid_button)
        toolbar.addWidget(self.preview_mode_label)
        toolbar.addWidget(self.input_path_label, 1)
        root.addLayout(toolbar)

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
        self.canvas_view.setMinimumSize(560, 420)
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
        return page

    def _build_camera_config_workspace(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)

        top = QHBoxLayout()
        top.addWidget(QLabel(self.t("Active camera count")))
        self.camera_count = QSpinBox()
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
        self.canvas_width = QSpinBox()
        self.canvas_width.setRange(320, 8192)
        self.canvas_width.setValue(int(stitch_profile.get("canvas", {}).get("width", 2440)))
        self.canvas_height = QSpinBox()
        self.canvas_height.setRange(240, 8192)
        self.canvas_height.setValue(int(stitch_profile.get("canvas", {}).get("height", 1800)))
        self.canvas_width.valueChanged.connect(self.mark_camera_config_dirty)
        self.canvas_height.valueChanged.connect(self.mark_camera_config_dirty)
        form.addRow(self.t("Width"), self.canvas_width)
        form.addRow(self.t("Height"), self.canvas_height)
        root.addWidget(output_group)

        perf_group = QGroupBox(self.t("Performance"))
        perf_form = QFormLayout(perf_group)
        self.process_fps = QSpinBox()
        self.process_fps.setRange(1, 30)
        self.process_fps.setValue(int(self.performance_config.get("process_fps", 5)))
        self.preview_fps = QSpinBox()
        self.preview_fps.setRange(1, 30)
        self.preview_fps.setValue(int(self.performance_config.get("preview_fps", 2)))
        self.max_input_width = QSpinBox()
        self.max_input_width.setRange(0, 4096)
        self.max_input_width.setSpecialValueText(self.t("Original"))
        self.max_input_width.setValue(int(self.performance_config.get("max_input_width", 960)))
        self.refresh_warped_preview = QCheckBox()
        self.refresh_warped_preview.setChecked(bool(self.performance_config.get("refresh_warped_preview", False)))
        self.use_intrinsics_live = QCheckBox()
        self.use_intrinsics_live.setChecked(bool(self.performance_config.get("use_intrinsics", False)))
        self.process_fps.valueChanged.connect(self.mark_camera_config_dirty)
        self.preview_fps.valueChanged.connect(self.mark_camera_config_dirty)
        self.max_input_width.valueChanged.connect(self.mark_camera_config_dirty)
        self.refresh_warped_preview.toggled.connect(self.mark_camera_config_dirty)
        self.use_intrinsics_live.toggled.connect(self.mark_camera_config_dirty)
        perf_form.addRow(self.t("Stitch FPS"), self.process_fps)
        perf_form.addRow(self.t("Preview FPS"), self.preview_fps)
        perf_form.addRow(self.t("Max input width"), self.max_input_width)
        perf_form.addRow(self.t("Refresh warped previews"), self.refresh_warped_preview)
        perf_form.addRow(self.t("Use undistort live"), self.use_intrinsics_live)
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
        save_project_button.clicked.connect(self.save_project_as)
        open_project_button.clicked.connect(self.open_project_file)
        backup_button.clicked.connect(self.backup_current_configs)
        export_runtime_button.clicked.connect(self.export_runtime_config_file)
        buttons.addWidget(save_project_button)
        buttons.addWidget(open_project_button)
        buttons.addWidget(backup_button)
        buttons.addWidget(export_runtime_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        root.addWidget(group)
        notes = QTextEdit()
        notes.setReadOnly(True)
        notes.setText(self.t(
            "Project files bundle calibration.yaml, cameras.yaml, and network.yaml into one portable .dsvs.yaml file.\n\n"
            "Use backups before large calibration or seam edits. Runtime export is the compact configuration intended for a future service/QGC bridge."
        ))
        root.addWidget(notes, 1)
        return page

    def _build_calibration_workspace(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)

        diagnostics_group = QGroupBox(self.t("Topology diagnostics"))
        diagnostics_layout = QVBoxLayout(diagnostics_group)
        self.topology_diagnostics_label = QLabel()
        self.topology_diagnostics_label.setWordWrap(True)
        self.topology_diagnostics_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        diagnostics_layout.addWidget(self.topology_diagnostics_label)
        root.addWidget(diagnostics_group)

        board_group = QGroupBox(self.t("Chessboard"))
        board_form = QFormLayout(board_group)
        board = self.calibration_config.get("calibration_board", {})
        self.board_columns = QSpinBox()
        self.board_columns.setRange(3, 40)
        self.board_columns.setValue(int(board.get("total_columns", 12)))
        self.board_rows = QSpinBox()
        self.board_rows.setRange(3, 40)
        self.board_rows.setValue(int(board.get("total_rows", 9)))
        self.square_size = QSpinBox()
        self.square_size.setRange(1, 500)
        self.square_size.setValue(int(float(board.get("square_size_mm", 25.0))))
        board_form.addRow(self.t("Total square columns"), self.board_columns)
        board_form.addRow(self.t("Total square rows"), self.board_rows)
        board_form.addRow(self.t("Square size mm"), self.square_size)
        root.addWidget(board_group)

        toolbar = QHBoxLayout()
        self.calibration_camera = QComboBox()
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
        toolbar.addWidget(QLabel(self.t("Camera")))
        toolbar.addWidget(self.calibration_camera)
        toolbar.addWidget(load_image)
        toolbar.addWidget(detect_board)
        toolbar.addWidget(calibrate_folder)
        toolbar.addWidget(undistort_preview)
        toolbar.addWidget(export_report)
        toolbar.addWidget(save_board)
        toolbar.addStretch(1)
        root.addLayout(toolbar)

        calibration_tabs = QTabWidget()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.calibration_image_view = PointEditorView()
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
        self.calibration_warp_view.setMinimumHeight(220)
        right_layout.addWidget(self.calibration_warp_view)
        self.calibration_log = QTextEdit()
        self.calibration_log.setReadOnly(True)
        right_layout.addWidget(self.calibration_log, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        calibration_tabs.addTab(splitter, self.t("Perspective"))

        seam_page = QWidget()
        seam_layout = QVBoxLayout(seam_page)
        seam_toolbar = QHBoxLayout()
        refresh_seam = QPushButton(self.t("Refresh Seam Canvas"))
        save_seam = QPushButton(self.t("Save Seam Points"))
        self.feather_width = QSpinBox()
        self.feather_width.setRange(1, 1000)
        self.feather_width.setValue(self.current_feather_width())
        refresh_seam.clicked.connect(self.refresh_seam_editor)
        save_seam.clicked.connect(self.save_seam_points)
        seam_toolbar.addWidget(refresh_seam)
        seam_toolbar.addWidget(save_seam)
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
        calibration_tabs.addTab(seam_page, self.t("Seams & Image Set"))

        root.addWidget(calibration_tabs, 1)

        self.calibration_camera.currentTextChanged.connect(self.populate_point_table)
        self.point_table.itemChanged.connect(self.on_point_table_changed)
        self.seam_table.itemChanged.connect(self.on_seam_table_changed)
        self.calibration_image: np.ndarray | None = None
        self._updating_point_table = False
        self._updating_seam_table = False
        self.calibration_folder = PROJECT_ROOT / "samples" / "calibration"
        self.populate_point_table()
        self.populate_seam_table()
        self.refresh_seam_editor()
        self.update_topology_diagnostics()
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

    def canvas_status_text(self) -> str:
        if self.preview_content_mode == PreviewContentMode.LIVE:
            return self.t("Live stitched view")
        if self.preview_content_mode == PreviewContentMode.STILL:
            return self.t("Static stitched view")
        return self.t("Live preview stopped")

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
            self.canvas_view.set_overlay(self.canvas_status_text())

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
            self.canvas_view.set_image(self.canvas, self.t("stitched canvas"))
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
        preview_interval = 1.0 / max(1, int(self.performance_config.get("preview_fps", 2)))
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
                    f"{display_name}\n{self.t('Live preview stopped')}"
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
                        f"{display_name}\n{self.t('Connecting')}"
                    )
                    continue
                if snapshot.status == STREAM_FAILED:
                    error = (snapshot.last_error or self.t("Failed")).splitlines()[0][:180]
                    lines = [
                        display_name,
                        self.t("Video stream unavailable"),
                        self.t("Last error: {error}", error=error),
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
                view.set_placeholder(f"{display_name}\n{status}")

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
            if result.result_id != self.last_stitch_error_result_id:
                self.last_stitch_error_result_id = result.result_id
                self.log(f"Stitch failed: {result.error}")
            return

        self.warped = result.warped
        self.canvas = result.canvas
        self.refresh_warped_views()
        if self.canvas is not None and self.should_render_canvas():
            self.canvas_view.set_image(self.canvas, self.t("stitched canvas"))
        self.statusBar().showMessage(self.t(
            "{status}: {count} active frames | stitch {ms:.1f} ms",
            status=self.t("Live frame"),
            count=len(self.frames),
            ms=self.last_stitch_ms,
        ))

    def _process_and_render_frames(self, status_prefix: str) -> None:
        start_time = time.perf_counter()
        active = set(self.active_camera_keys())
        self.frames = {key: frame for key, frame in self.frames.items() if key in active}
        if not self.frames:
            QMessageBox.warning(self, self.t("No images"), self.t("No matching camera images were found."))
            return

        try:
            self.warped, self.canvas = self.stitcher.process(self.frames)
        except Exception as exc:
            QMessageBox.critical(self, self.t("Stitch failed"), str(exc))
            return

        now = time.perf_counter()
        preview_interval = 1.0 / max(1, int(self.performance_config.get("preview_fps", 2)))
        should_refresh_preview = (now - self.last_preview_time) >= preview_interval or status_prefix != "Live frame"
        if should_refresh_preview:
            self.last_preview_time = now
            self.refresh_raw_preview(force=True)
            self.refresh_warped_views()
        if self.canvas is not None and self.should_render_canvas():
            self.canvas_view.set_image(self.canvas, self.t("stitched canvas"))
            if hasattr(self, "seam_editor") and status_prefix != "Live frame":
                self.refresh_seam_editor()
        self.last_stitch_ms = (time.perf_counter() - start_time) * 1000.0
        if now - self.last_health_time >= 1.0 or status_prefix != "Live frame":
            self.last_health_time = now
            self.update_health_table()
        self.statusBar().showMessage(self.t(
            "{status}: {count} active frames | stitch {ms:.1f} ms",
            status=self.t(status_prefix),
            count=len(self.frames),
            ms=self.last_stitch_ms,
        ))

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
        }
        self.camera_config["cameras"] = {key: row.to_config() for key, row in self.camera_rows.items()}

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
        self.statusBar().showMessage(self.t("Saved configs to {path}", path=CONFIG_DIR))

    def closeEvent(self, event) -> None:  # noqa: N802
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
        self.sync_camera_config_widgets()
        self.stitcher = self.create_stitcher()
        self.bump_preview_session()
        self.configure_live_stitch_processor()
        self.populate_point_table()
        self.populate_seam_table()
        self.refresh_seam_editor()
        self.update_topology_diagnostics()

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

            stitch_profile = self.current_stitch_profile()
            self.canvas_width.setValue(
                int(stitch_profile.get("canvas", {}).get("width", 2440))
            )
            self.canvas_height.setValue(
                int(stitch_profile.get("canvas", {}).get("height", 1800))
            )
            if hasattr(self, "feather_width"):
                self.feather_width.setValue(self.current_feather_width())
        finally:
            self._syncing_camera_widgets = False
        self._camera_config_dirty = False
        self.refresh_camera_count()

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
        self.topology_diagnostics_label.setText("\n".join(lines))

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
        self.seam_editor.set_stitch_points(self.current_stitch_points())
        self.populate_seam_table()

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

    def log(self, message: str) -> None:
        if hasattr(self, "calibration_log"):
            self.calibration_log.append(message)
        self.statusBar().showMessage(message)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()

