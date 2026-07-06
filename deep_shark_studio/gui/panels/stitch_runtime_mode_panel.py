"""Stitch runtime mode controls."""

from __future__ import annotations

from typing import Callable, Type

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSizePolicy,
)

from deep_shark_studio.stitch_runtime_modes import ProjectionSource, StitchRuntimeMode


class StitchRuntimeModePanel(QGroupBox):
    """UI-only panel for selecting runtime stitch modes and candidates."""

    def __init__(
        self,
        translate: Callable[[str], str],
        combo_box_class: Type[QComboBox] = QComboBox,
        double_spin_box_class: Type[QDoubleSpinBox] = QDoubleSpinBox,
        parent=None,
    ):
        super().__init__(translate("Stitch Runtime Mode"), parent)
        self.t = translate
        self.setToolTip(
            self.t("Runtime stitching algorithm, independent from Grid / Focus / Stitched layout.")
        )
        self._build(combo_box_class, double_spin_box_class)

    def _build(
        self,
        combo_box_class: Type[QComboBox],
        double_spin_box_class: Type[QDoubleSpinBox],
    ) -> None:
        layout = QGridLayout(self)
        self.runtime_mode_combo = combo_box_class()
        self.runtime_mode_combo.addItem(
            self.t("Far-field / distance priority"),
            StitchRuntimeMode.FAR_FIELD.value,
        )
        self.runtime_mode_combo.addItem(
            self.t("Near-field / front priority"),
            StitchRuntimeMode.NEAR_FIELD.value,
        )
        self.runtime_mode_combo.addItem(
            self.t("Auto / experimental disabled"),
            StitchRuntimeMode.AUTO.value,
        )
        auto_item = self.runtime_mode_combo.model().item(2)
        if auto_item is not None:
            auto_item.setEnabled(False)

        self.runtime_projection_label = QLabel(self.t("Near-field Projection Source"))
        self.runtime_projection_note = QLabel(self.t("Far-field uses current runtime projection."))
        self.runtime_projection_note.setWordWrap(True)
        self.runtime_projection_combo = combo_box_class()
        self.runtime_projection_combo.addItem(
            self.t("Current Perspective"),
            ProjectionSource.CURRENT_PERSPECTIVE.value,
        )
        self.runtime_projection_combo.addItem(
            self.t("Fisheye Rectilinear [Experimental]"),
            ProjectionSource.FISHEYE_RECTILINEAR_CANDIDATE.value,
        )
        self.runtime_projection_combo.setToolTip(
            self.t("Fisheye Rectilinear is experimental and only affects Near-field preview/runtime.")
        )

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

        self.runtime_load_fisheye_button = QPushButton(
            self.t("Load Fisheye Intrinsics Source...")
        )
        self.runtime_clear_fisheye_button = QPushButton(
            self.t("Clear Fisheye Intrinsics Source")
        )
        self.runtime_fisheye_source_status = QLabel(
            self.t("No fisheye intrinsics source loaded.")
        )
        self.runtime_fisheye_source_status.setWordWrap(True)
        self.runtime_fisheye_source_status.setMaximumHeight(48)
        self.runtime_fisheye_source_status.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )

        self.far_field_custom_layout_check = QCheckBox(
            self.t("Use Far-field Custom Layout")
        )
        self.far_field_layout_candidate_status = QLabel(
            self.t("No Far-field Layout Candidate loaded.")
        )
        self.far_field_layout_candidate_status.setWordWrap(True)
        self.far_field_layout_candidate_status.setMaximumHeight(70)
        self.far_field_layout_candidate_status.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self.far_field_load_candidate_button = QPushButton(
            self.t("Load Far-field Layout Candidate")
        )
        self.far_field_clear_candidate_button = QPushButton(
            self.t("Clear Far-field Layout Candidate")
        )

        self.runtime_candidate_status = QLabel(self.t("No Layout Candidate V2 loaded."))
        self.runtime_candidate_status.setWordWrap(True)
        self.runtime_load_candidate_button = QPushButton(
            self.t("Load Near-field Layout Candidate")
        )
        self.runtime_clear_candidate_button = QPushButton(self.t("Clear Near-field Layout Candidate"))
        self.runtime_open_candidate_button = QPushButton(self.t("Open Candidate Folder"))
        self.runtime_apply_button = QPushButton(self.t("Use for Current Preview Only"))
        self.runtime_apply_button.setToolTip(
            self.t(
                "Only applies to the current preview worker; does not write calibration.yaml or the formal profile."
            )
        )

        layout.addWidget(QLabel(self.t("Mode")), 0, 0)
        layout.addWidget(self.runtime_mode_combo, 0, 1)
        layout.addWidget(self.runtime_projection_note, 0, 2, 1, 2)
        layout.addWidget(self.runtime_projection_label, 1, 0)
        layout.addWidget(self.runtime_projection_combo, 1, 1)
        layout.addWidget(QLabel(self.t("Fisheye Balance")), 1, 2)
        layout.addWidget(self.runtime_fisheye_balance, 1, 3)
        layout.addWidget(QLabel(self.t("Fisheye FOV Scale")), 2, 2)
        layout.addWidget(self.runtime_fisheye_fov_scale, 2, 3)
        layout.addWidget(self.runtime_load_fisheye_button, 2, 0)
        layout.addWidget(self.runtime_clear_fisheye_button, 2, 1)
        layout.addWidget(self.runtime_fisheye_source_status, 3, 0, 1, 4)
        layout.addWidget(self.far_field_custom_layout_check, 4, 0, 1, 2)
        layout.addWidget(self.far_field_layout_candidate_status, 4, 2, 1, 2)
        layout.addWidget(self.far_field_load_candidate_button, 5, 0)
        layout.addWidget(self.far_field_clear_candidate_button, 5, 1)
        layout.addWidget(QLabel(self.t("Near-field Layout Candidate")), 6, 0)
        layout.addWidget(self.runtime_candidate_status, 6, 1, 1, 3)
        layout.addWidget(self.runtime_load_candidate_button, 7, 0)
        layout.addWidget(self.runtime_clear_candidate_button, 7, 1)
        layout.addWidget(self.runtime_open_candidate_button, 7, 2)
        layout.addWidget(self.runtime_apply_button, 7, 3)
