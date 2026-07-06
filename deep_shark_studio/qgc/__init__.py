"""Headless DeepShark payload service helpers for QGC integration."""

from .video_output import (
    FfmpegVideoSink,
    NullVideoSink,
    VideoOutputConfig,
    build_ffmpeg_command,
)

__all__ = [
    "DeepSharkRuntimeService",
    "DeepSharkRuntimeServiceConfig",
    "FfmpegVideoSink",
    "NullVideoSink",
    "VideoOutputConfig",
    "build_ffmpeg_command",
    "build_runtime_service_config",
    "build_stream_configs",
]


def __getattr__(name: str):
    if name in {
        "DeepSharkRuntimeService",
        "DeepSharkRuntimeServiceConfig",
        "build_runtime_service_config",
        "build_stream_configs",
    }:
        from .runtime_service import (
            DeepSharkRuntimeService,
            DeepSharkRuntimeServiceConfig,
            build_runtime_service_config,
            build_stream_configs,
        )

        values = {
            "DeepSharkRuntimeService": DeepSharkRuntimeService,
            "DeepSharkRuntimeServiceConfig": DeepSharkRuntimeServiceConfig,
            "build_runtime_service_config": build_runtime_service_config,
            "build_stream_configs": build_stream_configs,
        }
        globals().update(values)
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
