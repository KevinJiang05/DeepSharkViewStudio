param(
    [Parameter(Mandatory = $true)]
    [int]$ProcessId,
    [Parameter(Mandatory = $true)]
    [string]$OutputPath,
    [switch]$Screen
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class DeepSharkWindowCapture {
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left; public int Top; public int Right; public int Bottom; }
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int command);
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr hWnd, IntPtr hdc, uint flags);
}
"@

$process = Get-Process -Id $ProcessId
$handle = $process.MainWindowHandle
if ($handle -eq [IntPtr]::Zero) {
    throw "Process $ProcessId does not own a visible main window."
}

[DeepSharkWindowCapture]::ShowWindow($handle, 9) | Out-Null
Start-Sleep -Milliseconds 250

$rect = New-Object DeepSharkWindowCapture+RECT
if (-not [DeepSharkWindowCapture]::GetWindowRect($handle, [ref]$rect)) {
    throw "Unable to read the main-window bounds."
}

$width = $rect.Right - $rect.Left
$height = $rect.Bottom - $rect.Top
if ($width -le 0 -or $height -le 0) {
    throw "The main-window bounds are invalid: ${width}x${height}."
}

$directory = Split-Path -Parent $OutputPath
if ($directory) {
    [System.IO.Directory]::CreateDirectory($directory) | Out-Null
}

$bitmap = New-Object System.Drawing.Bitmap $width, $height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$printed = $false
try {
    if ($Screen) {
        $graphics.CopyFromScreen($rect.Left, $rect.Top, 0, 0, $bitmap.Size)
    }
    else {
        $hdc = $graphics.GetHdc()
        try {
            # PW_RENDERFULLCONTENT asks DWM-backed windows to paint even while
            # another app is in front. This avoids capturing the occluding window.
            $printed = [DeepSharkWindowCapture]::PrintWindow($handle, $hdc, 2)
        }
        finally {
            $graphics.ReleaseHdc($hdc)
        }
        if (-not $printed) {
            $graphics.CopyFromScreen($rect.Left, $rect.Top, 0, 0, $bitmap.Size)
        }
    }
    $bitmap.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
}
finally {
    $graphics.Dispose()
    $bitmap.Dispose()
}

[pscustomobject]@{
    ProcessId = $ProcessId
    Handle = $handle
    Left = $rect.Left
    Top = $rect.Top
    Width = $width
    Height = $height
    PrintedFromWindow = $printed
    CaptureMode = if ($Screen) { "screen" } elseif ($printed) { "window" } else { "screen-fallback" }
    OutputPath = $OutputPath
}
