# DeepSharkViewStudio 阶段 C：候选正确性与性能报告

日期：2026-07-13（Asia/Shanghai）  
范围：Far-field Default、B-2 View、Far-field Custom、Near Current、Near Fisheye；真实三路 RTSP、CPU/OpenCL、FFmpeg/UDP/QGC；不实现 Auto，不改正式标定结果。

## 1. 结论

阶段 C 的代码修复、性能优化、真实设备测量、输出链验证和回归已完成，可以进入用户验收。

- 五条生产处理链均已通过契约测试和像素级回归；Far Custom 仍严格使用 B-2 per-camera projection 与 `b2_weight_selection`。
- 真实三路 1920×1080 RTSP 帧已用于五链路 CPU 测量；RTX 4060 Laptop GPU 已用于 B-2 OpenCL 实测和同帧一致性检查。
- 最慢链路 Near Fisheye 的真实帧 p95 为 209.82 ms，因此共享包默认值设为 4 FPS，保留约 40 ms 的 p95 调度余量。
- 用户真实 `configs/cameras.yaml` 中手工设置的 30 FPS 没有被覆盖；4 FPS 只写入共享代码默认、无头服务默认与 `cameras.example.yaml`。
- 真实 Far Default → FFmpeg → UDP/QGC 链在 11.75 秒内提交 31 帧、写入 31 帧，restart=0、无 stderr；本次 FFmpeg PID 停止后消失，QGC 进程和 UDP 5600 监听保持存活。QGC 画面由用户于本轮手动确认满足要求。
- 完整测试发现共 308 项：305 通过、1 项因 Windows 缺少创建符号链接权限而跳过；唯一 1 fail + 1 error 是仓库既有 dual-topology fixture 与当前正式标定不一致，不是阶段 C 回归。
- 正式 `calibration.yaml`、真实 `cameras.yaml`、`network.yaml` 的 SHA-256 与阶段 C 开始时完全一致。

本报告不声称五个视图在“全新启动”时都已有生产候选。当前仓库没有持久化 Far Custom 候选，也没有匹配 Near Fisheye 的 schema v3 Near 候选；这两条链的真实帧性能使用生产算法加只存在于内存的测试绑定测得，不能替代候选生成和项目包验收。

## 2. 当前真实调用链与边界

| 用户视图 | UI/运行时边界 | Projection | Composition / 输出 | 当前启动资源 |
|---|---|---|---|---|
| Far Default | `RuntimeViewPreset.FAR_DEFAULT` → template worker → `RuntimeStitchController._process_far_field()` | `SurroundStitcher.warp_all_with_masks()`，正式 Perspective profile | `compose_horizontal_feather()` → 2200×700 | 可直接使用 |
| B-2 View | `RuntimeViewPreset.B2_VIEW` → candidate worker → `CandidatePanoramaProcessor.process()` | B-2 标定候选的每相机 remap | 预计算权重选择 → 1800×600 | 自动发现最新 B-2 候选 |
| Far Custom | template worker → `RuntimeStitchController._process_far_field_custom()` | `B2CandidateProjectionProvider` 返回 B-2 per-camera warped image 与几何 mask | camera adjust → `b2_weight_selection` → candidate crop | 无持久化 Far layout candidate |
| Near Current | Near runtime → `CurrentPerspectiveProjectionProvider` | 正式 Perspective warp 与几何 mask | `render_near_field_from_warped()` → Near layout crop | schema v2 候选存在，但启动不自动加载 |
| Near Fisheye | Near runtime → `FisheyeRectilinearProjectionProvider` | fisheye map cache → rectilinear remap → template Perspective warp | 与 Near Current 共用 Near compositor | 无匹配的 schema v3 Near candidate |

关键边界证据：

- Far Custom provider 选择发生在 `deep_shark_studio/stitch_runtime_controller.py:148`；B-2 source 只会进入 `B2CandidateProjectionProvider`。
- Far candidate loader 在 `deep_shark_studio/seam/far_field_layout_candidate_runtime.py:82` 校验 projection source，并在 `:104`–`:117` 强制 B-2 projection 与 `b2_weight_selection` 成对，禁止静默降级成 Perspective feather。
- Near candidate schema v2 在 `deep_shark_studio/seam/layout_candidate_runtime.py:203` 被定义为 Current Perspective；schema v3 才允许显式 fisheye projection 和 intrinsics source。
- Near selected projection 必须与 candidate projection 一致，检查位于 `deep_shark_studio/stitch_runtime_controller.py:252`。
- Auto 仍未实现，也没有被阶段 C 接入；所有五个命名视图均落到既有明确边界。

## 3. 正确性修复

### 3.1 黑色像素与 valid mask

旧链路多处用 `np.any(image != 0)` 推断有效区域，会把真实黑色像素误判为无效。本轮把几何有效性作为独立契约贯穿各 provider、adjust 和 compositor：

- `deep_shark_studio/stitcher.py:190` 为每个相机/输入分辨率生成并缓存 Perspective 几何 mask；`warp_all_with_masks()` 在 `:219` 返回图像与 mask。
- `deep_shark_studio/projection/runtime_providers.py:48`、`:101`、`:234` 分别为 Current、B-2、Fisheye provider 输出明确的 `valid_masks` 和 provenance。
- `deep_shark_studio/stitching/camera_layout_adjust.py:39` 同时变换图像和 mask；只有旧调用者没有显式 mask 时才保留非零像素 fallback。
- `deep_shark_studio/topology.py:380`、Near/Far compositors 均消费显式 mask。黑色有效像素回归覆盖于 `tests/test_topology.py`、`tests/test_projection_runtime_provider.py` 和 `tests/test_camera_layout_adjust.py`。

### 3.2 分辨率、K 和输出画布

- B-2 runtime 对输入分辨率采用严格拒绝，不再静默 resize/夹断：CPU 路径位于 `deep_shark_studio/calibration_candidate.py:1513`，OpenCL 路径位于 `:1605`。
- 正式 intrinsics 的 K 会按标定参考分辨率缩放：`deep_shark_studio/calibration.py:156`；`fx/skew/cx` 按 X、`fy/skew/cy` 按 Y 缩放。
- Near candidate 输出若大于正式 canvas 会在加载期拒绝：`deep_shark_studio/seam/layout_candidate_runtime.py:162`；投影后还会在 `deep_shark_studio/stitch_runtime_controller.py:269` 再校验实际 canvas。
- Far Custom oversize crop 会显式报错，不再静默截短；相应回归位于 `tests/test_far_field_custom_layout.py`。

### 3.3 NaN、hole 与权重选择

- B-2 和 Far Custom 都使用有限值权重检查，只有至少一个相机具有 finite weight 的像素才可写入：`deep_shark_studio/calibration_candidate.py:1490`、`deep_shark_studio/seam/far_field_custom_compositor.py:301`。
- 非有限权重在 affine/selection 前清理；测试覆盖 NaN 输入不会产生随机选择或 hole。
- CPU compose 改为预计算 selection masks + `cv2.copyTo`，不再每帧创建三份全画布临时布尔数组。

### 3.4 缓存失效与资源上限

- `SurroundStitcher` 的 intrinsics remap 与几何 mask cache 按相机和实际分辨率键控，并有有界淘汰：`deep_shark_studio/stitcher.py:90`、`:351`。
- Near runtime plan key 覆盖 candidate、camera adjust、pair、vertical safety、seam、shape 和 mask 内容；不缓存帧像素。LRU 最多 2 项、总计 128 MiB：`deep_shark_studio/seam/near_field_compositor.py:30`。
- Layout Tuner 新增 `deep_shark_studio/layout_tuner_cache.py`，把冻结帧、projection 参数、runtime source revision 与 mapping revision 纳入 provenance，阻止用陈旧投影保存新候选。
- 1000 组随机极端输入中，Near planned path 与 legacy path 的图像、全部 mask、metrics、crop 完全一致；8 线程 200 次并发 cache hit 逐像素一致。

### 3.5 线程、session 与 UI 真值

- `StitchProcessingWorker` 的 latest-only 队列、generation/configuration id 和滚动 p50/p95/result FPS 位于 `deep_shark_studio/stitch_processing.py:53`、`:137`、`:224`；失败 reconfigure 先构造后交换，旧健康 worker 不被半提交覆盖。
- 发现并修复了真实 UI 饥饿：30 FPS submit 高于约 8 FPS result 时，旧逻辑不断把最小可接受 request id 推向未来，Canvas 永远等不到可接受结果。`deep_shark_studio/gui/main_window.py:6889` 现在固定当前相机集合的第一个 request gate；回归 `test_frame_set_gate_pins_first_request_when_submit_outpaces_results` 覆盖 request 10/11 先提交、result 10 后返回的场景。
- 运行日志于 23:14:48 记录 `request=1 result=1 canvas_rendered=True`，随后相机集合稳定；这证明修复后的第一帧可被 Canvas 接受。
- selected/effective view 分离、自动预览重放、Stop 后不擅自重启、150 ms fisheye debounce、候选加载事务化和独立 runtime apply error 均由 `tests/test_preview_state.py` / `tests/test_ui_workflow.py` 覆盖。
- Canvas/warped/raw 渲染按实际可见 tab 和 preview FPS gating，检查位于 `deep_shark_studio/gui/main_window.py:6923`、`:7016`，减少不可见页面的 QImage/Pixmap 开销。

## 4. 性能结果

### 4.1 可复现的隔离合成基准

命令使用 `tools/benchmark_stage_c.py`，每个模式独立进程、warm-up 10、samples 50、输入 1920×1080。结果 FPS 为 50 次端到端处理耗时的倒数；RSS 是 Windows Working Set，不含 GPU 显存。

| 模式 | p50 ms | p95 ms | 结果 FPS | 峰值 RSS MiB | 后端 |
|---|---:|---:|---:|---:|---|
| Far Default | 79.56 | 94.17 | 12.30 | 90.1 | CPU |
| B-2 View | 13.75 | 14.74 | 72.63 | 296.0 | CPU |
| Far Custom | 30.83 | 32.80 | 32.27 | 151.6 | CPU |
| Near Current | 70.87 | 75.66 | 14.03 | 103.9 | CPU |
| Near Fisheye | 165.00 | 212.54 | 5.80 | 155.8 | CPU |
| B-2 View | 8.10 | 9.79 | 119.53 | 287.5 | OpenCL / RTX 4060 |

阶段 C correctness baseline 与最终隔离结果方向对比：

| 模式 | correctness baseline p95 ms | 最终 p95 ms | p95 降幅 |
|---|---:|---:|---:|
| Far Default | 261.36 | 94.17 | 64.0% |
| B-2 View CPU | 98.22 | 14.74 | 85.0% |
| Far Custom | 159.96 | 32.80 | 79.5% |
| Near Current | 445.77 | 75.66 | 83.0% |
| Near Fisheye | 564.21 | 212.54 | 62.3% |

Baseline 曾把五模式放在同一进程；最终数字为模式隔离，因此该表用于证明优化方向，不作为严格同进程 A/B。Near compositor 另有同一 harness 的严格 before/after：无 vertical safety p95 323.85 → 76.57 ms（4.23×），启用 vertical safety p95 496.29 → 86.23 ms（5.76×）。

### 4.2 真实三路 RTSP 帧

三路 1920×1080 帧的采集时间偏差为 32 ms，帧龄 14–46 ms；拿到一组帧后立即释放 capture workers，再使用同一真实帧集测算法，避免运动内容差异影响模式比较。

| 模式 | p50 ms | p95 ms | 结果 FPS | 说明 |
|---|---:|---:|---:|---|
| Far Default | 95.04 | 169.02 | 9.40 | 正式 profile |
| B-2 View CPU | 18.26 | 19.82 | 54.50 | 当前 B-2 候选 |
| Far Custom | 41.51 | 46.56 | 23.84 | 生产链 + 内存 identity Far layout |
| Near Current | 124.68 | 154.11 | 7.97 | 当前持久化 schema v2 Near layout |
| Near Fisheye | 194.43 | 209.82 | 5.14 | 生产链 + 内存 fisheye projection binding |

这组数字包含真实图像内容，但不包含持续 RTSP 解码、Qt 渲染或 FFmpeg 编码；后两项由 4.4 的持续链验证补充。

### 4.3 真实画面 CPU / OpenCL

| 指标 | B-2 CPU | B-2 OpenCL |
|---|---:|---:|
| p50 | 18.26 ms | 9.89 ms |
| p95 | 19.82 ms | 12.02 ms |
| 结果 FPS | 54.50 | 100.11 |
| OpenCL active | 否 | 是，NVIDIA GeForce RTX 4060 Laptop GPU |

同一真实画面输出差异：

- 发生任一通道差异的像素：13,265 / 1,080,000（1.228%）；
- 最大通道差：1；平均绝对通道差：0.00556；
- 差值大于 1 的像素：0；
- 因此属于 CPU/OpenCL 的 uint8 rounding 差异，没有结构性画面差异。

OpenCL 只有在 `haveOpenCL`、`setUseOpenCL(True)` 和默认 device 均验证成功后才报告 active；初始化或运行期 `cv2.error` 会锁存 CPU fallback，并在 telemetry 中记录原因。模拟 fallback 回归通过，真实 RTX 4060 路径没有发生 fallback。

### 4.4 持续链、drop、frame age 与 FFmpeg/QGC

用户真实配置仍显式请求 30 FPS。在持续 Far Default UI 运行中观察到：result FPS 6.8、p50 175.9 ms、p95 211.0 ms、最大帧龄 0.1 s、failed=0；累计 submitted 10,442 / dropped 6,533 / completed 3,907。高 drop 是过载的 latest-only 保护行为，证明 30 FPS 不是可兑现默认值。

无窗口 Far Default 以新的 4 FPS 默认运行 11.75 秒：

- capture 启动后稳定为三路实时，结束前帧龄 13–27 ms；
- submitted 31，FFmpeg written 31；
- output state 在运行期为 `running`，PID 7056，restart 0、无 stderr、无 last_error；
- 停止后 state=`stopped`、PID 清空，PID 7056 已退出，无任何 `ffmpeg.exe` 残留；
- QGC PID 25648 持续存活，并继续拥有 `127.0.0.1:5600/UDP`；
- 用户手动确认 QGC 可见画面满足要求。

Windows 主动 terminate FFmpeg 后保留 `exit_code=1` 作为历史诊断值，但状态为 `stopped` 且 `last_error` 为空；运行期没有把它误报成 failure。

### 4.5 RSS / 稳定性

B-2 压测使用相同进程分别执行 500 和 2000 次：

| 后端 | 样本 | p95 ms | setup 后 Working Set 增量 |
|---|---:|---:|---:|
| CPU | 500 | 19.00 | 17.9 MiB |
| CPU | 2000 | 17.76 | 27.1 MiB |
| OpenCL | 500 | 9.90 | 7.6 MiB |
| OpenCL | 2000 | 9.63 | 17.3 MiB |

吞吐没有随样本数恶化，也未出现持续线性同量级增长；观察到的是 OpenCV/allocator working-set 保留。仍需 30–60 分钟真实 RTSP + QGC soak 才能对长期进程内存和 GPU 显存给出发布级结论。

## 5. 默认 FPS 决策

真实帧最慢 p95 为 Near Fisheye 209.82 ms：

```text
1000 / 209.82 = 4.77 results/s
```

因此选择最高的安全整数 4 FPS，周期 250 ms，相比 p95 仍有约 40 ms 余量。变更点：

- `deep_shark_studio/config.py:21`：共享 `DEFAULT_PROCESS_FPS = 4`；
- GUI 缺省、reload 缺省和 scheduler 均引用该常量；
- `deep_shark_studio/qgc/runtime_service.py:51` 与 CLI parser 使用同一常量；
- `configs/cameras.example.yaml:42` 为 4；
- `DEFAULT_PREVIEW_FPS` 保持 2；用户真实 `configs/cameras.yaml:42` 的 30 保持不动，继续被视为显式 override。

## 6. 测试与完整性

### 6.1 测试结果

- 默认值与 QGC/启动定向：34/34 通过。
- 阶段 C 定向套件：249 项中 247 通过；唯一 1 fail + 1 error 是下述既有 dual fixture。
- 完整发现：308 项，305 通过，1 skip，1 fail，1 error，耗时 90.174 秒。
- skip：Windows 当前账户没有创建符号链接所需权限；严格 package validator 的其他 path escape 测试均通过。

既有异常：

- `tests.test_topology.TopologyConfigTests.test_dual_profile_geometry_is_horizontal_and_isolated`
- `tests.test_topology.TopologyImageTests.test_dual_output_is_left_right_not_top_bottom`

当前正式配置的 dual overlap 是 `[700, 900]`，其 seam 解析为 `1353.9915`，所以第一项断言失败，第二项被运行时正确的“seam outside overlap”检查拒绝。Git HEAD 已含相同 fixture 和断言；阶段 C 只在该文件新增 valid-black 回归，没有改变 dual 测试或正式 calibration。

### 6.2 正式配置未变

```text
calibration.yaml  3780E4E2FF7AB10D015F3B71CB40E61A909238B4BB87A18D8AC15B3F65A1D7D6
cameras.yaml      7BC51075444D7D8C0F2DE62D35627A1D9A688235C17B8B2FD2F836511DE0F02F
network.yaml      E4E267F9AD6C44A84F3FDA1CD9EE9907E1D35B9E901930E9C9FFB2FFB8DD3FA5
```

`git diff --check` 通过。候选目录、项目数据和 `configs/active_project.yaml` 均未被本轮创建、删除或改写。

## 7. 当前资产可用性与剩余风险

### P1：五视图演示资产不完整

- 工作区没有 `candidate_type: far_field_layout` 的 Far Custom 候选。
- 四个持久化 Near candidates 均为 schema v2 / Current Perspective，没有 schema v3 / Fisheye projection binding。
- 没有 `configs/active_project.yaml`，也没有可恢复的 `project.dcsvs.yaml`，所以全新启动只自动具备 Far Default 和最新 B-2 View。

影响：不能把 Far Custom / Near Fisheye 标记为“生产启动即用”，也无法完成这两个模式的项目包迁移 smoke。生成、保存或激活候选属于项目数据写入，需要用户另行批准。

### P2：Near Fisheye 仍需实机重新调 layout

真实帧内存绑定结果 black-pixel ratio 为 37.39%，provider 同时警告 template source points 是在 raw/template 图像上标定，rectilinear remap 后可能需要重新调布局。算法正确不等于现有布局可用于演示。

### P2：长期资源稳定性尚需 soak

已做 B-2 2000 次处理和约 12 秒真实 FFmpeg 链；尚未覆盖 30–60 分钟五模式轮换、相机断线重连、QGC receiver 重启以及 GPU memory telemetry。

### P2：OpenCL 只加速 B-2 View

Far Custom 需要 per-camera warped images，而当前 OpenCL 快路径为避免下载只返回最终 canvas，因此 Far Custom provider 仍使用 CPU B-2 projection。贸然共享 OpenCL canvas-only 快路径会改变模式语义。

### P3：正常停止的 FFmpeg exit code 可进一步归一化

Windows 主动停止产生历史 `exit_code=1`，虽然 state 和 error domain 正确，但教师演示状态面板若直接解读 exit code 可能产生困惑。可在后续仅对“明确 stop requested”显示 `operator stop`，保留原始 code 在诊断详情。

### P3：旧 dual topology fixture

修正 fixture 或废弃 dual profile 应单独处理；不要为了阶段 C 绿色而修改真实 `calibration.yaml`。

## 8. 验收命令

```powershell
$env:PYTHONUTF8='1'
$env:QT_QPA_PLATFORM='offscreen'
$python='D:\Develop\envs\deep-shark-view-studio\Scripts\python.exe'

& $python -m unittest -v tests.test_clean_startup tests.test_qgc_runtime_service
& $python -m unittest discover -v -s tests -p 'test_*.py'

& $python tools\benchmark_stage_c.py --warmup 10 --samples 50
& $python tools\benchmark_stage_c.py --modes b2_view --b2-opencl --warmup 10 --samples 50

& $python -m py_compile deep_shark_studio\config.py deep_shark_studio\gui\main_window.py deep_shark_studio\qgc\runtime_service.py
git diff --check
Get-FileHash -Algorithm SHA256 configs\calibration.yaml,configs\cameras.yaml,configs\network.yaml
```

完整发现当前预期仍会报告上述两个同源 dual fixture 异常；其他新增失败应视为回归。

## 9. 教师演示 smoke checklist

1. 确认三路相机在线、QGC 已监听 UDP 5600，且没有遗留 `ffmpeg.exe`。
2. 双击 `StartDeepSharkViewStudio.cmd`；确认只出现一个 Studio 窗口，视图默认为 Far Default，并自动开始实时预览。
3. 10 秒内确认三路 raw frame、Canvas 和状态摘要更新；selected/effective 均为 Far Default，failed=0，frame age 小于 0.5 秒。
4. 将实际演示配置的 Stitch FPS 设为 4，确认 result FPS 稳定且 drop 不持续快速增长。30 FPS 是当前用户 override，不是承诺吞吐。
5. 切换 B-2 View；确认 effective status、projection 和后端 telemetry 一致。若启用 OpenCL，应显示 RTX 4060 和 `opencl_active=true`；若不可用，应明确显示 CPU fallback 原因。
6. 只有在正式候选已生成并装入项目包后，才演示 Far Custom、Near Current、Near Fisheye；逐项确认 selected=effective、无尺寸/投影 mismatch、Canvas 非陈旧帧。
7. 启动 QGC 输出服务；确认 PID 出现，submitted/written 增长、restart=0、无 stderr；在 QGC 中确认画面持续变化。
8. 停止 QGC 输出；5 秒内确认该 PID 消失、QGC 仍运行、Studio Live 仍继续。
9. 关闭 Studio；确认没有遗留 Python/FFmpeg 子进程，并复核三份正式配置哈希。

## 10. 建议的下一阶段

阶段 C 代码现已停止在“等待用户验收”状态。后续不应继续顺手修改算法或项目数据。若用户批准，建议按以下顺序单独开展：

1. 生成并人工验收 Far Custom production candidate；
2. 基于 rectilinear 画面重新调 Near layout，保存匹配的 schema v3 Near Fisheye candidate；
3. 导出包含 B-2、Far、Near、fisheye source 的可迁移项目包，在无关目录复制、校验、激活、重启；
4. 进行 30–60 分钟五模式 + QGC soak，并采集 CPU、GPU memory、RSS、frame age 和 reconnect 数据；
5. 最后再修复或移除旧 dual fixture，并完成 Demo Release Candidate 冻结。
