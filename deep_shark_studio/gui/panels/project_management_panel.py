"""Task-oriented project portability panel."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class ProjectManagementPanel(QWidget):
    """Keep the portable package workflow primary and legacy tools secondary."""

    def __init__(self, translate: Callable[[str], str], parent=None):
        super().__init__(parent)
        self.t = translate
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.portable_group = QGroupBox(
            self.t("Portable Project Package (Recommended)")
        )
        portable_layout = QVBoxLayout(self.portable_group)
        self.active_project_status = QLabel(
            self.t("Using local configs; no portable package is active.")
        )
        self.active_project_status.setWordWrap(True)
        self.active_project_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.active_project_status.setStyleSheet(
            "QLabel { color: #334155; background: #f8fafc; "
            "border: 1px solid #cbd5e1; padding: 6px 8px; }"
        )
        portable_layout.addWidget(self.active_project_status)

        self.portable_note = QLabel(
            self.t(
                "Portable packages include configs and active Far/Near/B-2/fisheye candidates. Validate is read-only; Activate backs up current configs before switching."
            )
        )
        self.portable_note.setWordWrap(True)
        portable_layout.addWidget(self.portable_note)

        self.project_status = QLabel(self.t("Last project action: none"))
        self.project_status.setWordWrap(True)
        self.project_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        portable_layout.addWidget(self.project_status)

        primary_actions = QHBoxLayout()
        self.project_package_export_button = QPushButton(
            self.t("Export Portable Package...")
        )
        self.project_package_validate_button = QPushButton(
            self.t("Validate Package (Read-only)...")
        )
        self.project_package_import_button = QPushButton(
            self.t("Validate & Activate Package...")
        )
        self.primary_action_buttons = [
            self.project_package_export_button,
            self.project_package_validate_button,
            self.project_package_import_button,
        ]
        for button in self.primary_action_buttons:
            primary_actions.addWidget(button)
        primary_actions.addStretch(1)
        portable_layout.addLayout(primary_actions)
        root.addWidget(self.portable_group)

        self.advanced_toggle = QPushButton(
            self.t("Show Legacy & Advanced Tools")
        )
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setChecked(False)
        self.advanced_toggle.toggled.connect(self.set_advanced_visible)
        root.addWidget(self.advanced_toggle, 0)

        self.advanced_container = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_container)
        advanced_layout.setContentsMargins(0, 0, 0, 0)

        self.legacy_group = QGroupBox(
            self.t("Legacy Project File (.dsvs.yaml)")
        )
        legacy_layout = QVBoxLayout(self.legacy_group)
        self.legacy_note = QLabel(
            self.t(
                "Legacy .dsvs.yaml files contain only the three config files. They do not carry candidates and are retained for compatibility, not team transfer."
            )
        )
        self.legacy_note.setWordWrap(True)
        legacy_layout.addWidget(self.legacy_note)
        legacy_actions = QHBoxLayout()
        self.save_legacy_project_button = QPushButton(
            self.t("Save Legacy Project File...")
        )
        self.open_legacy_project_button = QPushButton(
            self.t("Open Legacy Project File...")
        )
        legacy_actions.addWidget(self.save_legacy_project_button)
        legacy_actions.addWidget(self.open_legacy_project_button)
        legacy_actions.addStretch(1)
        legacy_layout.addLayout(legacy_actions)
        advanced_layout.addWidget(self.legacy_group)

        maintenance_group = QGroupBox(self.t("Local Maintenance"))
        maintenance_layout = QVBoxLayout(maintenance_group)
        maintenance_note = QLabel(
            self.t(
                "Backups protect active configs. Runtime Snapshot is an experimental service hand-off and is not a portable project."
            )
        )
        maintenance_note.setWordWrap(True)
        maintenance_layout.addWidget(maintenance_note)
        maintenance_actions = QHBoxLayout()
        self.backup_configs_button = QPushButton(
            self.t("Backup Active Configs")
        )
        self.export_runtime_button = QPushButton(
            self.t("Export Runtime Snapshot [Experimental]")
        )
        maintenance_actions.addWidget(self.backup_configs_button)
        maintenance_actions.addWidget(self.export_runtime_button)
        maintenance_actions.addStretch(1)
        maintenance_layout.addLayout(maintenance_actions)
        advanced_layout.addWidget(maintenance_group)
        self.advanced_container.setVisible(False)
        root.addWidget(self.advanced_container)
        root.addStretch(1)

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

    def set_advanced_visible(self, visible: bool) -> None:
        self.advanced_container.setVisible(visible)
        self.advanced_toggle.setText(
            self.t("Hide Legacy & Advanced Tools")
            if visible
            else self.t("Show Legacy & Advanced Tools")
        )


__all__ = ["ProjectManagementPanel"]
