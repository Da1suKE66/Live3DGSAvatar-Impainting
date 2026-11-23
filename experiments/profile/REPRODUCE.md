# 复跑侧脸实验

以下命令在本分支的独立 checkout 中执行。每个条件使用独立的产物名，训练器会拒绝覆盖同名模型。数值复核只依赖 Python 3.11 标准库。

```bash
python3 experiments/profile/compare.py
```

## 准备输入

使用带时间切分的 512×512 MediaPipe 采集目录。每个 `datasets/{train,validation,test}/` 需要逐帧编号的 RGBA 图、包含 local_id/source_index/seconds/yaw/tracking_ok 的 tracks.json，以及含原始 61 维 blend_weight、K、w2v、width、height 的 render_inputs.npz。训练图的透明通道外保留源 RGB，供关键点检测。从已有采集复用表情系数和刚体姿态，脚本仅从所选正脸重新拟合网格。topology.npz 存 468 点模板的 faces、uv_faces、uvs。规范网格 OBJ、Face Landmarker 模型、三套时间分区、原视频的指纹见 [provenance.json](logs/provenance.json) 和 [帧清单](logs/split.csv)。

选择工作区内一个新目录作为 RUN。已有工作机示例采用项目的 Python 3.11、CUDA 12.8 扩展、分开的教师依赖。SyncFix 模型目录内的 config.yaml 指向本机 SDXL VAE。本次 base 的 Python 渲染器来自新分支，复用了工作机上已编译的光栅化扩展。

```bash
export WORKSPACE=/home/lakeside/Documents/3DAvatar
export PYTHON_BIN="$WORKSPACE/.venv/bin/python" ASSETS="$WORKSPACE/assets" UPSTREAM="$WORKSPACE/upstream"
export RUN="$WORKSPACE/runs/profile-replay" CUDA_VISIBLE_DEVICES=0
export SYNCFIX_SITE="$WORKSPACE/.deps/syncfix-site" LEGACY_SITE="$WORKSPACE/.deps/profile-legacy"
export FATE_SITE="$WORKSPACE/.deps/profile-fate" MORPHABLE_SITE="$WORKSPACE/.deps/profile-md"
export HF_HOME="$WORKSPACE/.deps/profile-hub" MPLCONFIGDIR="$WORKSPACE/.deps/matplotlib"
export NUMBA_CACHE_DIR="$WORKSPACE/.deps/cache"

"$PYTHON_BIN" experiments/profile/prepare.py --capture "$WORKSPACE/runs/capture-3d" \
  --assets "$ASSETS" --run "$RUN" --syncfix-repo "$UPSTREAM/SyncFix" \
  --syncfix-python "$PYTHON_BIN" --syncfix-model "$WORKSPACE/runs/github-reference/syncfix-model"
ln -s "$RUN/local.yaml" configs/local.yaml

for method in base real raw native-syncfix syncfix difix edgar elite eg3d fate-prior morphable; do
  bash experiments/profile/run_short.sh "$method" || exit 1
done
```

base 阶段训练基座并运行学长的 make_repairs.py，保存同一基帧、三个训练参考和六个目标视角。每个后续条件把输出交给同一 scripts/train.py。raw 使用未修复 RGB。native-syncfix 使用上游键控、锐化和修孔后的 RGBA。其他短预算教师共用基础渲染 alpha > 0.5 的轮廓，因此这个消融主要检验轮廓内的颜色修复。重复运行教师时采用新 RUN，避免混入旧视角。

## 模型对应关系

各仓库 commit、第三方包版本和本次使用的权重 SHA-256 收录在 [环境及资产记录](logs/provenance.json)。目录 `$ASSETS/models/profile/` 中使用以下名字。

| 方法 | 作者发布与本地输入 |
|---|---|
| EG3D | NVIDIA NGC 的 ffhqrebalanced512-128.pkl 存为 eg3d.pkl。W+ 反演 160 步，用人脸关键点将转头结果对齐到六个原生相机。 [源码](https://github.com/NVlabs/eg3d) |
| Morphable Diffusion | download_data.sh 中第一个 Google Drive 文件是 CLIP ViT-L/14，本地名 morphable.ckpt；第二个是 FaceScape 模型，本地名 morphable-second.ckpt。采样 16 视角、50 步、CFG 2，采用公开网格表面体素与原生相机方向，将六个对应视角映射回原图。官方 MICA/FLAME 预检单列。 [源码](https://github.com/xiyichen/morphablediffusion) |
| Difix3D+ | nvidia/difix_ref 的完整 Diffusers 快照放在 difix_ref/。采用单步参考版图像增强，时间步 199，本实验做一次 3D 标签回训。 [源码](https://github.com/nv-tlabs/Difix3D) |
| EDGAR | 解压作者 exp6 检查点到 edgar_exp6.pt，另备 sd-turbo/ 和 sapiens-pretrain-0.3b-torchscript/。使用 application 预处理、白底、FP32。 [源码](https://github.com/kwanyun/edgar-code) |
| ELITE | 作者的 2d_prior.pth，用 hufix 的官方 sample_batch_multi_tensor 做 RGB 修复。其模型名硬编码为 stabilityai/sd-turbo，需要先按所列 revision 缓存到 HF_HOME。 [源码](https://github.com/kaist-ami/ELITE) |
| FATE | GFPGANv1.3.pth 用于 FFHQ 人脸修复阶段。此消融止于恢复已对齐的脸图并逆映射。作者列出的 SphereHead 检查点在本机请求返回 HTTP 403；完整反演、全头训练还需要相应身份资产。 [源码](https://github.com/zjwfufu/FateAvatar) |
| SyncFix | 使用学长 make_repairs.py 和作者 inference.py。同一侧的三个相机联合修复，并使用训练参考图。 [源码](https://github.com/Deming77/SyncFix) |

## 复查原生输入

准备好本次采集和相应的作者仓库后，可以逐项复查发布状态和输入。脚本输出仓库版本、输入形状及缺项，返回码 2 表示停在同身份推理之前。

```bash
subject="$RUN/datasets/subject"
"$PYTHON_BIN" experiments/profile/probe.py gaf "$UPSTREAM/GAF" "$subject"
"$PYTHON_BIN" experiments/profile/probe.py cap4d "$UPSTREAM/cap4d" "$subject"
"$PYTHON_BIN" experiments/profile/probe.py diffusion-avatars "$UPSTREAM/diffusion-avatars" "$subject"
"$PYTHON_BIN" experiments/profile/probe.py morphable-diffusion "$UPSTREAM/morphablediffusion" "$subject"
```

## 完整上游配方

`native-recipe.sh` 在 RUN 下建立独立副本，进入 train_base.sh 和 train_avatar.sh。其 Python 启动器将 conda 的解释器选择映射到 PYTHON_BIN。配置保留本实验的时间切分、网格和 3D 训练随机种子。自动生成标签时需要在 PYTHONPATH 中提供 SyncFix 的依赖和源文件。原版分支按验证组合分早停并装入最佳权重。

```bash
export OMP_NUM_THREADS=4 TORCH_HOME="$ASSETS/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHONPATH="$SYNCFIX_SITE:$UPSTREAM/SyncFix/src" bash experiments/profile/native-recipe.sh
ln -s "$RUN/full-repo/models/subject/base" "$RUN/models/subject/full-base"
ln -s "$RUN/full-repo/models/subject/avatar" "$RUN/models/subject/full-native"
bash experiments/profile/train_full_control.sh real > "$RUN/full-real.log" 2>&1

expt="$RUN/full-replacement"
"$PYTHON_BIN" experiments/profile/full_labels.py "$RUN" "$expt" stage
PYTHONPATH="$LEGACY_SITE" "$PYTHON_BIN" experiments/profile/elite.py "$expt" "$UPSTREAM/ELITE" "$ASSETS/models/profile/2d_prior.pth"
PYTHONPATH="$LEGACY_SITE" "$PYTHON_BIN" experiments/profile/edgar.py "$expt" "$UPSTREAM/edgar-code" "$ASSETS/models/profile"
for method in elite edgar; do
  "$PYTHON_BIN" experiments/profile/full_labels.py "$RUN" "$expt" "$method"
  bash experiments/profile/train_full_control.sh "$method" > "$RUN/full-$method.log" 2>&1 || exit 1
done
"$PYTHON_BIN" experiments/profile/blend.py "$RUN/full-repo/datasets/subject/repair/repairs" \
  "$expt/labels/elite" "$expt/labels/blend"
bash experiments/profile/train_full_control.sh blend > "$RUN/full-blend.log" 2>&1
for method in real elite edgar blend; do
  bash experiments/profile/train_full_control.sh "$method" --shared-init > "$RUN/matched-$method.log" 2>&1 || exit 1
done
for name in full-base full-native full-real full-elite full-edgar full-blend matched-real matched-elite matched-edgar matched-blend; do
  PYTHONPATH="$PWD/src" "$PYTHON_BIN" experiments/profile/score.py "$RUN" "$name" || exit 1
done
```

原生键控筛出的 20 个相机、各自的 alpha、锐化系数和暗孔处理固定。ELITE 与 EDGAR 替换这些像素的 RGB，各使用同一张正脸身份参考。上游 train_avatar.sh 为每个方法单独初始化未训练的高斯模型。基座生成训练标签。结果报告的受控比较用 --shared-init 额外指定原生 SyncFix 的统一初始 PLY，保留已保存的表情权重网络。四个带有修复图的配方都按 2 张修复配 6 张真实图开始，计划在步数预算的后半程改为 1 张配 7 张。所有分组按相同的验证规则选模。仅真实图的完整对照每步取 8 张真实帧。扩散全部发生在制作标签时，最终头像由原有光栅器渲染。上游 init_uv.py 未固定其表情网络的随机初值。新机器上的完整重跑会生成另一个初始 PLY，各方法在该次运行内部仍共享它。想精确复用这次的初值，可使用原工作机 RUN/full-repo/tmp/subject_avatar_unified.ply；SHA-256 记录在 verification.json。

对应单张照片和渲染的本地对照图可在完成以上评测后生成。

```bash
"$PYTHON_BIN" experiments/profile/figure.py "$RUN" "$RUN/comparison.jpg"
```
