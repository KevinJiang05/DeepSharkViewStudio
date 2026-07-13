param(
    [Parameter(Mandatory = $true)]
    [int]$ProcessId,
    [Parameter(Mandatory = $true)]
    [int]$X,
    [Parameter(Mandatory = $true)]
    [int]$Y,
    [switch]$Physical
)

$ErrorActionPreference = "Stop"
Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class DeepSharkWindowInput {
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left; public int Top; public int Right; public int Bottom; }
    [StructLayout(LayoutKind.Sequential)]
    public struct POINT { public int X; public int Y; }
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);
    [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr hWnd, out RECT rect);
    [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr hWnd, ref POINT point);
    [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr hWnd, uint message, UIntPtr wParam, IntPtr lParam);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool BringWindowToTop(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")] public static extern bool GetCursorPos(out POINT point);
    [DllImport("user32.dll")] public static extern void mouse_event(uint flags, uint dx, uint dy, uint data, UIntPtr extra);
}
"@

$process = Get-Process -Id $ProcessId
$handle = $process.MainWindowHandle
if ($handle -eq [IntPtr]::Zero) {
    throw "Process $ProcessId does not own a visible main window."
}

$window = New-Object DeepSharkWindowInput+RECT
$client = New-Object DeepSharkWindowInput+RECT
$origin = New-Object DeepSharkWindowInput+POINT
if (-not [DeepSharkWindowInput]::GetWindowRect($handle, [ref]$window)) {
    throw "Unable to read the main-window bounds."
}
if (-not [DeepSharkWindowInput]::GetClientRect($handle, [ref]$client)) {
    throw "Unable to read the main-window client bounds."
}
if (-not [DeepSharkWindowInput]::ClientToScreen($handle, [ref]$origin)) {
    throw "Unable to map the main-window client origin."
}

$screenX = $window.Left + $X
$screenY = $window.Top + $Y
$clientX = $screenX - $origin.X
$clientY = $screenY - $origin.Y
$clientWidth = $client.Right - $client.Left
$clientHeight = $client.Bottom - $client.Top
if ($clientX -lt 0 -or $clientY -lt 0 -or $clientX -ge $clientWidth -or $clientY -ge $clientHeight) {
    throw "Point ($X, $Y) maps outside the client area (${clientWidth}x${clientHeight})."
}

if ($Physical) {
    $previousCursor = New-Object DeepSharkWindowInput+POINT
    [DeepSharkWindowInput]::GetCursorPos([ref]$previousCursor) | Out-Null
    [DeepSharkWindowInput]::BringWindowToTop($handle) | Out-Null
    [DeepSharkWindowInput]::SetForegroundWindow($handle) | Out-Null
    Start-Sleep -Milliseconds 350
    [DeepSharkWindowInput]::SetCursorPos($screenX, $screenY) | Out-Null
    [DeepSharkWindowInput]::mouse_event(0x0002, 0, 0, 0, [UIntPtr]::Zero)
    [DeepSharkWindowInput]::mouse_event(0x0004, 0, 0, 0, [UIntPtr]::Zero)
    Start-Sleep -Milliseconds 180
    [DeepSharkWindowInput]::SetCursorPos($previousCursor.X, $previousCursor.Y) | Out-Null
}
else {
    $packed = [IntPtr](($clientY -shl 16) -bor ($clientX -band 0xffff))
    [DeepSharkWindowInput]::PostMessage($handle, 0x0200, [UIntPtr]::Zero, $packed) | Out-Null
    [DeepSharkWindowInput]::PostMessage(
        $handle,
        0x0201,
        [UIntPtr]::new([uint64]1),
        $packed
    ) | Out-Null
    [DeepSharkWindowInput]::PostMessage($handle, 0x0202, [UIntPtr]::Zero, $packed) | Out-Null
}

[pscustomobject]@{
    ProcessId = $ProcessId
    WindowPoint = "${X},${Y}"
    ClientPoint = "${clientX},${clientY}"
    Mode = if ($Physical) { "physical" } else { "window-message" }
}
