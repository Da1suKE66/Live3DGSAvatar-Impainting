# 03 · 只用正脸训练，哪种方法能补好侧脸？

**新增复测包含三个训练种子，并在 55 张此前剔除的大角度实拍上验证到 −55° 和 +63°。** 复测另行核对了脸缘掩模在极端姿态下的偏差。[最新结论与逐种子结果](SECOND_ROUND.md)

## 第一轮筛选（训练种子 1701）

第一轮在原生 SyncFix、EDGAR、HuFix 与各半混合中筛选可继续复测的教师。 用正脸训练得到的 base 渲染缺失角度，教师修复这些视图，再把修复图与真实正脸混合训练 3DGS。共用同一个初始模型后，原生 SyncFix 在第一轮完整配方的 49 张留出侧脸上的感知距离最低（LPIPS 0.1377）。EDGAR 的侧脸像素最接近实拍（PSNR 22.755 dB），比原生高 0.665 dB，42/49 帧有提升，也比只用真实帧高 0.311 dB。代价是侧脸 LPIPS 上升到 0.1409。几种方法各有改善的指标，本次没有整体替代基线的赢家。[同初值逐帧配对](paired.csv) · [EDGAR 训练与评测](logs/matched-edgar/)

HuFix 相对原生的 PSNR 增益在固定初值后缩小到 0.259 dB。若只看正 yaw 一侧，各半混合的 LPIPS 从 0.1468 降到 0.1413；负向则从 0.1296 升到 0.1396。两次转头分开看，更能判断改动适合哪个方向。[完整指标](summary.csv)

## 第一轮共用同一个初始化文件后的实拍结果

PSNR 用实拍脸部掩模内的像素计算，越高越好。LPIPS 用冻结的 VGG 在整张 512×512 黑底图上计算，越低越好。负向和正向分别有 26 和 23 帧。最后一列是同一上游验证规则选出的训练步数。

| 训练条件 | 侧脸 PSNR ↑ (dB) | 侧脸 LPIPS ↓ | 负向 LPIPS ↓ | 正向 LPIPS ↓ | 正脸 LPIPS ↓ | 选中步数 |
|---|---:|---:|---:|---:|---:|---:|
| 原生 SyncFix | 22.089 | **0.1377** | 0.1296 | 0.1468 | 0.0775 | 2,500 |
| 只用真实帧 | 22.444 | 0.1395 | **0.1286** | 0.1519 | 0.0764 | 3,000 |
| ELITE HuFix | 22.348 | 0.1410 | 0.1367 | 0.1458 | 0.0779 | 3,000 |
| EDGAR exp6 | **22.755** | 0.1409 | 0.1360 | 0.1464 | 0.0751 | 4,500 |
| SyncFix 与 HuFix 各半 | 22.204 | 0.1404 | 0.1396 | **0.1413** | **0.0744** | 6,000 |

原生流程先用训练好的 base 渲染待修复视角，再从未训练的 UV 高斯初始化模型训练 avatar。核验发现，上游脚本在各次启动时独立随机生成一组表情映射权重。为隔离这个变量，上表所有 avatar 条件都读取了原生 SyncFix 用过的同一个初始 PLY，42,566 个高斯和表情网络参数逐字节相同。教师组沿用原生 20 张修复标签的相机和 alpha、相同锐化和暗孔处理。关闭修复的对照每步用 8 张实拍。各组均按上游完整预算和验证规则选模，所选步数会随标签变化。[初始化核验与计划](logs/shared-init-plan.md) · [逐帧配对](paired.csv) · [实际训练命令](logs/matched-elite/matched-elite.log)

- **只用真实帧** 相比原生 SyncFix，平均 PSNR 相差 +0.355 dB，LPIPS 相差 +0.0019。49 张侧脸中，29 张 PSNR 更高，21 张 LPIPS 更低。[逐帧结果](logs/matched-real/frames.csv)

- **ELITE HuFix** 相比原生 SyncFix，平均 PSNR 相差 +0.259 dB，LPIPS 相差 +0.0033。49 张侧脸中，25 张 PSNR 更高，19 张 LPIPS 更低。[逐帧结果](logs/matched-elite/frames.csv)

- **EDGAR exp6** 相比原生 SyncFix，平均 PSNR 相差 +0.665 dB，LPIPS 相差 +0.0032。49 张侧脸中，42 张 PSNR 更高，17 张 LPIPS 更低。[逐帧结果](logs/matched-edgar/frames.csv)

- **SyncFix 与 HuFix 各半** 相比原生 SyncFix，平均 PSNR 相差 +0.115 dB，LPIPS 相差 +0.0027。49 张侧脸中，23 张 PSNR 更高，23 张 LPIPS 更低。[逐帧结果](logs/matched-blend/frames.csv)

原生筛出的 20 个相机里，17 个在正 yaw 方向（+10° 到 +90°），另 3 个为 −5°、−10° 和 −15°。报告同时列出两侧结果，便于看清这个不对称。各半混合是在看过最初测试分数之后选定的探索性条件，比例在训练前固定。[20 张标签](logs/full-elite/full-elite-labels.log) · [追加混合计划](logs/blend-plan.md)

### 上游默认独立初始化时的观察

按原脚本分别生成初始权重时，原生 SyncFix、HuFix、EDGAR 和各半混合的侧脸 PSNR 分别为 22.089、22.824、22.100、22.448 dB，侧脸 LPIPS 分别为 0.1377、0.1411、0.1457、0.1400。HuFix 在该轮有 41/49 帧的 PSNR 更高，平均差 +0.735 dB。这一差值包含表情网络初值和教师两种变化。各组的逐帧结果保留在 full-* 系列目录；matched-* 为复用原生初始化的重跑。种子 1701 的原生参照是 full-native。[原生](logs/full-native/) · [HuFix](logs/full-elite/) · [EDGAR](logs/full-edgar/) · [混合](logs/full-blend/) · [真实帧对照](logs/full-real/)

## 相同预算下的七条论文路线

同一个 1,200 步 base，分别续训 250 步。每次更新取六张真实帧，有修复标签的条件再取两张虚拟视角。这组固定六个对称相机（±20°、±35°、±50°），按最后一步评测。与上表分别比较，训练预算、视角分布、初始化及选模规则均有差异。

| 方法与实际执行范围 | 侧脸 PSNR ↑ (dB) | 侧脸 LPIPS ↓ | 运行记录 |
|---|---:|---:|---|
| base，续训前 | 22.128 | 0.1465 | [日志](logs/base/) |
| 只用真实帧续训 | 22.554 | 0.1364 | [日志](logs/real/) |
| 加原始渲染，沿用基础轮廓 | 22.417 | 0.1365 | [日志](logs/raw/) |
| SyncFix，原生 RGBA 后处理 | 22.379 | **0.1362** | [日志](logs/native-syncfix/) |
| SyncFix，共用基础轮廓 | **22.558** | 0.1373 | [日志](logs/syncfix/) |
| ELITE，作者的 HuFix 二维修复模块 | 22.438 | 0.1376 | [日志](logs/elite/) |
| EDGAR，作者的 exp6 视图修复器 | 22.420 | 0.1387 | [日志](logs/edgar/) |
| Difix3D+，官方 Difix-ref 单次修复并回训 | 22.405 | 0.1386 | [日志](logs/difix/) |
| EG3D，正脸 W+ 反演后转头并对齐关键点 | 22.420 | 0.1408 | [日志](logs/eg3d/) |
| FATE，仅 FFHQ 对齐与 GFPGAN 修复阶段 | 22.313 | 0.1379 | [日志](logs/fate-prior/) |
| Morphable Diffusion，使用本次人脸网格体素和相机方向 | 22.063 | 0.1470 | [日志](logs/morphable/) |

仅真实帧续训，就把侧脸 LPIPS 从 0.1465 降到了 0.1364。短预算中，七条路线均未同时超过这项对照的侧脸 PSNR 和 LPIPS。Morphable Diffusion 用作者的 FaceScape 权重做了 16 视角联合采样、50 个去噪步，再取六个目标角度回训。该数字衡量 468 点网格适配到这个头像后的结果。FATE 的完整 SphereHead 权重下载在本机返回 HTTP 403，本次只测了人脸修复子模块。ELITE 保留了项目现有 3DGS 初始化。[Morphable 原生输入预检](logs/morphable/flame-preflight.log) · [FATE 权重预检](logs/fate-prior/whole-head-preflight.log) · [权重、推理参数与复跑步骤](REPRODUCE.md)

## 另外三条路线停在哪里

| 方法 | 本次实际检查及缺项 |
|---|---|
| DiffusionAvatars | 作者列出的八个预训练身份均与本次视频不同。本次采集缺该身份的多视角 NPHM 跟踪、规范坐标等渲染条件与对应模型。 [预检日志](logs/diffusion-avatars/preflight.log) · [作者仓库](https://github.com/tobias-kirschstein/diffusion-avatars) |
| GAF | 检查的作者仓库版本只有 README 和预告图，未找到训练、推理入口或权重。 [预检日志](logs/gaf/preflight.log) · [作者仓库](https://github.com/tangjiapeng/GAF) |
| CAP4D | 用发布的样例核对了输入字段与加载器。本机缺授权的 FLAME 2023 no-jaw 模型、这段视频的 FLAME 拟合及相应的参考图描述。 [预检日志](logs/cap4d/preflight.log) · [作者仓库](https://github.com/felixtaubner/cap4d) |

## 这次能验证多大角度

- 上游起点是 CherryRH/Live3DGSAvatar 的 `feat/geom-v1`，提交 `5cfbdef3fde47614c636a95308afa72d034e5fcc`。从同一视频按时间取 192 张正脸训练、32 张正脸验证，绝对 yaw 均不超过 10°。468 点网格的身份和表情形变仅由训练正脸拟合；测试时提供逐帧驱动和相机。来源帧号与时间在三个切分之间无重叠。[预先计划](PLAN.md) · [分帧清单](logs/split.csv)
- 独立测试段有 49 张绝对 yaw 至少 15° 的帧，以及 38 张正脸。实拍极值是 −21.84° 和 +22.05°。49 张侧脸主要来自两次连续转头；第一轮是一份采集、一个身份、一个 3D 训练种子（1701），第二轮额外重复 1702、1703。结果可用于决定下一个要试的模块，跨人效果仍需新视频验证。
- 在 ±20°、±35°、±50° 的同相机几何检查中，HuFix 的平均网格内孔洞为 2.56%，轮廓外溢为 5.89%。原生 SyncFix 分别为 2.82% 和 6.14%。两项用预测 alpha 对照网格投影计算。±50° 的本地对照中，镜框和太阳穴仍可见拉伸或涂抹。这六个虚拟相机没有同姿态同步实拍。第二轮改用未被训练采纳的真实大角度照片评测侧脸。当前网格的表示范围仍主要是脸部。[原生逐角度指标](logs/full-native/virtual.csv) · [HuFix 逐角度指标](logs/matched-elite/virtual.csv)
- 仓库保留代码、逐帧分数、训练/推理日志及权重指纹。原始人像、带人像的渲染和训练模型保存在工作机。`python3 experiments/profile/compare.py` 可用已提交的数据复核 28 组模型在原测试段的结果。大角度实拍的两套掩模由 `compare_wide.py` 复核。[复跑说明](REPRODUCE.md) · [资产与版本](logs/provenance.json) · [核验记录](logs/verification.json)
