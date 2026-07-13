"""Compact runtime workflow panel."""

from __future__ import annotations

from typing import Callable, Type

from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from deep_shark_studio.gui.runtime_workflow import RuntimeViewPreset


class StitchRuntimeModePanel(QGroupBox):
    """Expose five supported runtime views and collapse their source assets."""

    def __init__(
        self,
        translate: Callable[[str], str],
        combo_box_class: Type[QComboBox] = QComboBox,
        double_spin_box_class: Type[QDoubleSpinBox] = QDoubleSpinBox,
        parent=None,
    ):
        super().__init__(translate("Runtime View"), parent)
        self.t = translate
        self.setToolTip(
            self.t(
                "Choose one concrete processing view. Grid, Focus, and Stitched only change display layout."
            )
        )
        self._build(combo_box_class, double_spin_box_class)

    def _build(
        self,
        combo_box_class: Type[QComboBox],
        double_spin_box_class: Type[QDoubleSpinBox],
    ) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 7)
        root.setSpacing(5)

        primary = QGridLayout()
        primary.setHorizontalSpacing(8)
        self.runtime_view_combo = combo_box_class()
        self.runtime_view_combo.addItem(
            self.t("Far Default — current profile"),
            RuntimeViewPreset.FAR_DEFAULT.value,
        )
        self.runtime_view_combo.addItem(
            self.t("B-2 View — candidate diagnostics [Advanced]"),
            RuntimeViewPreset.B2_VIEW.value,
        )
        self.runtime_view_combo.addItem(
            self.t("Far Custom — B-2 projection + custom layout"),
            RuntimeViewPreset.FAR_CUSTOM.value,
        )
        self.runtime_view_combo.addItem(
            self.t("Near Current — current perspective"),
            RuntimeViewPreset.NEAR_CURRENT.value,
        )
        self.runtime_view_combo.addItem(
            self.t("Near Fisheye — rectilinear [Experimental]"),
            RuntimeViewPreset.NEAR_FISHEYE.value,
        )
        self.runtime_view_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.runtime_apply_button = QPushButton(self.t("Apply Runtime View"))
        self.runtime_apply_button.setToolTip(
            self.t(
                "Apply to the current preview worker only. This does not write calibration.yaml."
            )
        )
        primary.addWidget(QLabel(self.t("View")), 0, 0)
        primary.addWidget(self.runtime_view_combo, 0, 1)
        primary.addWidget(self.runtime_apply_button, 0, 2)

        self.runtime_selection_summary = QLabel()
        self.runtime_selection_summary.setWordWrap(True)
        self.runtime_selection_summary.setStyleSheet(
            "QLabel { color: #475569; padding: 1px 0; }"
        )
        primary.addWidget(self.runtime_selection_summary, 1, 1, 1, 2)
        root.addLayout(primary)

        status_row = QHBoxLayout()
        self.runtime_effective_status_label = QLabel()
        self.runtime_effective_status_label.setWordWrap(True)
        self.runtime_effective_status_label.setStyleSheet(
            "QLabel { color: #9a3412; font-weight: 600; }"
        )
        self.runtime_details_toggle = QPushButton(
            self.t("Candidate & Projection Sources")
        )
        self.runtime_details_toggle.setCheckable(True)
        self.runtime_details_toggle.setChecked(False)
        self.runtime_details_toggle.toggled.connect(self.set_details_visible)
        status_row.addWidget(self.runtime_effective_status_label, 1)
        status_row.addWidget(self.runtime_details_toggle)
        root.addLayout(status_row)

        self.runtime_details_container = QWidget()
        details = QGridLayout(self.runtime_details_container)
        details.setContentsMargins(8, 6, 8, 4)
        details.setHorizontalSpacing(8)
        details.setVerticalSpacing(5)

        self.b2_source_label = QLabel(self.t("B-2 Projection Candidate"))
        self.b2_candidate_status = QLabel()
        self.b2_candidate_status.setWordWrap(True)
        details.addWidget(self.b2_source_label, 0, 0)
        details.addWidget(self.b2_candidate_status, 0, 1, 1, 3)

        self.far_source_label = QLabel(self.t("Far Custom Layout"))
        self.far_field_layout_candidate_status = QLabel(
            self.t("No Far-field Layout Candidate loaded.")
        )
        self.far_field_layout_candidate_status.setWordWrap(True)
        self.far_field_load_candidate_button = QPushButton(
            self.t("Load Far Custom Layout...")
        )
        self.far_field_clear_candidate_button = QPushButton(self.t("Clear"))
        details.addWidget(self.far_source_label, 1, 0)
        details.addWidget(self.far_field_layout_candidate_status, 1, 1)
        details.addWidget(self.far_field_load_candidate_button, 1, 2)
        details.addWidget(self.far_field_clear_candidate_button, 1, 3)

        self.near_source_label = QLabel(self.t("Near Layout Candidate"))
        self.runtime_candidate_status = QLabel(
            self.t("No Near Layout Candidate loaded.")
        )
        self.runtime_candidate_status.setWordWrap(True)
        self.runtime_load_candidate_button = QPushButton(
            self.t("Load Near Layout...")
        )
        self.runtime_clear_candidate_button = QPushButton(self.t("Clear"))
        self.runtime_open_candidate_button = QPushButton(self.t("Open Folder"))
        details.addWidget(self.near_source_label, 2, 0)
        details.addWidget(self.runtime_candidate_status, 2, 1)
        details.addWidget(self.runtime_load_candidate_button, 2, 2)
        details.addWidget(self.runtime_clear_candidate_button, 2, 3)
        details.addWidget(self.runtime_open_candidate_button, 3, 3)

        self.fisheye_source_label = QLabel(self.t("Near Fisheye Intrinsics"))
        self.runtime_fisheye_source_status = QLabel(
            self.t("No fisheye intrinsics source loaded.")
        )
        self.runtime_fisheye_source_status.setWordWrap(True)
        self.runtime_load_fisheye_button = QPushButton(
            self.t("Load Fisheye Intrinsics...")
        )
        self.runtime_clear_fisheye_button = QPushButton(self.t("Clear"))
        details.addWidget(self.fisheye_source_label, 4, 0)
        details.addWidget(self.runtime_fisheye_source_status, 4, 1)
        details.addWidget(self.runtime_load_fisheye_button, 4, 2)
        details.addWidget(self.runtime_clear_fisheye_button, 4, 3)

        self.runtime_fisheye_balance = double_spin_box_class()
        self.runtime_fisheye_balance.setRange(0.0, 1.0)
        self.runtime_fisheye_balance.setSingleStep(0.05)
        self.runtime_fisheye_balance.setDecimals(2)
        self.runtime_fisheye_balance.setValue(0.6)
        self.runtime_fisheye_fov_scale = double_spin_box_class()
        self.runtime_fisheye_fov_scale.setRange(0.8, 1.2)
        self.runtime_fisheye_fov_scale.setSingleStep(0.05)
        self.runtime_fisheye_fov_scale.setDecimals(2)
        self.runtime_fisheye_fov_scale.setValue(1.0)
        self.fisheye_params_widget = QWidget()
        fisheye_params = QHBoxLayout(self.fisheye_params_widget)
        fisheye_params.setContentsMargins(0, 0, 0, 0)
        fisheye_params.addWidget(QLabel(self.t("Balance")))
        fisheye_params.addWidget(self.runtime_fisheye_balance)
        fisheye_params.addWidget(QLabel(self.t("FOV Scale")))
        fisheye_params.addWidget(self.runtime_fisheye_fov_scale)
        fisheye_params.addStretch(1)
        details.addWidget(self.fisheye_params_widget, 5, 1, 1, 3)

        self.runtime_details_container.setVisible(False)
        root.addWidget(self.runtime_details_container)
        self.set_source_context(RuntimeViewPreset.FAR_DEFAULT)

    def set_details_visible(self, visible: bool) -> None:
        self.runtime_details_container.setVisible(visible)
        self.runtime_details_toggle.setText(
            self.t("Hide Candidate & Projection Sources")
            if visible
            else self.t("Candidate & Projection Sources")
        )

    def set_selected_preset(self, preset: RuntimeViewPreset | str) -> None:
        value = preset.value if isinstance(preset, RuntimeViewPreset) else str(preset)
        index = self.runtime_view_combo.findData(value)
        if index >= 0:
            self.runtime_view_combo.setCurrentIndex(index)

    def set_source_context(self, preset: RuntimeViewPreset | str) -> None:
        if not isinstance(preset, RuntimeViewPreset):
            preset = RuntimeViewPreset(str(preset))
        show_b2 = preset in {
            RuntimeViewPreset.B2_VIEW,
            RuntimeViewPreset.FAR_CUSTOM,
        }
        show_far = preset == RuntimeViewPreset.FAR_CUSTOM
        show_near = preset in {
            RuntimeViewPreset.NEAR_CURRENT,
            RuntimeViewPreset.NEAR_FISHEYE,
        }
        show_fisheye = preset == RuntimeViewPreset.NEAR_FISHEYE

        for widget in (self.b2_source_label, self.b2_candidate_status):
            widget.setVisible(show_b2)
        for widget in (
            self.far_source_label,
            self.far_field_layout_candidate_status,
            self.far_field_load_candidate_button,
            self.far_field_clear_candidate_button,
        ):
            widget.setVisible(show_far)
        for widget in (
            self.near_source_label,
            self.runtime_candidate_status,
            self.runtime_load_candidate_button,
            self.runtime_clear_candidate_button,
            self.runtime_open_candidate_button,
        ):
            widget.setVisible(show_near)
        for widget in (
            self.fisheye_source_label,
            self.runtime_fisheye_source_status,
            self.runtime_load_fisheye_button,
            self.runtime_clear_fisheye_button,
            self.fisheye_params_widget,
        ):
            widget.setVisible(show_fisheye)

        has_sources = preset != RuntimeViewPreset.FAR_DEFAULT
        self.runtime_details_toggle.setEnabled(has_sources)
        if not has_sources:
            self.runtime_details_toggle.setChecked(False)
