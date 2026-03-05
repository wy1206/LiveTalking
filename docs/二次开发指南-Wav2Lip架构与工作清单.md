# LiveTalking 技术架构与 Wav2Lip 实现总结 · 二次开发工作清单

本文档基于对项目完整代码的通读，不修改任何现有实现，仅做归纳与指引。

---

## 一、整体技术架构

### 1.1 分层结构

```
┌─────────────────────────────────────────────────────────────────────────┐
│  前端 / 客户端                                                           │
│  web/webrtcapi.html, client.js → /offer, /human, /humanaudio 等 HTTP API│
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  入口与路由 (app.py)                                                     │
│  - aiohttp Web 服务，CORS 开放                                           │
│  - /offer → 建 WebRTC 连接，按 session 创建 BaseReal 子类实例             │
│  - /human → 文本/对话送入 TTS + 数字人                                    │
│  - /humanaudio, /interrupt_talk, /is_speaking 等                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  传输层 (webrtc.py)                                                      │
│  - HumanPlayer 持有 BaseReal，启动 worker 线程执行 render()               │
│  - PlayerStreamTrack：音视频轨，从 _queue 取帧，按 20ms/40ms 打时间戳      │
│  - 音视频：16kHz mono / 25fps BGR24 → 经 aiortc 推给浏览器                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  数字人抽象层 (basereal.py)                                              │
│  - BaseReal：TTS 选择、音频 chunk（20ms）、自定义动作、录制、打断           │
│  - 子类：LipReal(wav2lip) / MuseReal(musetalk) / LightReal(ultralight)  │
│  - 核心：put_msg_txt → TTS → put_audio_frame → ASR/特征 → 口型推理 → 推流 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  TTS (ttsreal.py)│    │  ASR/特征        │     │  口型模型        │
│  EdgeTTS/Azure/  │    │  lipasr.py       │    │  wav2lip/models │
│  GPT-SoVITS 等   │    │  (mel + 队列)    │    │  (Wav2Lip 推理)  │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

- **多模型支持**：`app.py` 根据 `--model` 选择 `LipReal` / `MuseReal` / `LightReal`，并加载对应模型与 avatar。
- **多会话**：每个 WebRTC offer 生成一个 `sessionid`，对应一个 `nerfreals[sessionid]` 实例，互不共享。

### 1.2 数据流概览（Wav2Lip 路径）

1. **文本 → 语音**
  用户输入或 LLM 回复 → `put_msg_txt()` → TTS 的 `msgqueue` → `process_tts` 线程 → 流式 PCM 以 20ms 一帧送入 `put_audio_frame()`。
2. **音频 → 特征与口型**
  - `LipASR.run_step()` 每轮从 `get_audio_frame()` 取 `batch_size*2` 个 20ms 帧，拼成一段波形。  
  - 用 `wav2lip.audio.melspectrogram()` 算 mel，按 stride 切为多个 mel chunk，送入 `feat_queue`。  
  - 同一批音频帧原样送入 `output_queue`，供后续与视频帧对齐输出。
3. **口型推理**
  - 推理线程从 `feat_queue` 取一包 mel，从 `output_queue` 取 `batch_size*2` 个音频帧。  
  - 静音：不跑模型，直接取 avatar 的 `full_imgs` 循环/镜像帧。  
  - 有语音：用 `face_imgs` + mel 输入 Wav2Lip，得到口型脸图，再 `paste_back_frame` 贴回 `full_imgs` 对应位置。
4. **合成与推流**
  - `process_frames` 从 `res_frame_queue` 取（合成帧, idx, 音频帧），加 Logo、录制，再放入 WebRTC 的 video/audio `_queue`。  
  - 时间戳由 `webrtc.py` 的 `next_timestamp()` 按 40ms/20ms 递增，保证音画同步。

---

## 二、Wav2Lip 在项目中的实现方式

### 2.1 模型与代码位置

- **实际使用的模型类**：`wav2lip/models/wav2lip_v2.py` 中的 `Wav2Lip`（`models/__init__.py` 导出）。  
- **输入**：  
  - 人脸：`(B, 6, 256, 256)`，即下半脸 mask + 原脸拼接的 6 通道。  
  - 音频：mel 谱 `(B, 1, 80, 16)`（80 mel bins，16 时间步）。
- **输出**：口型区域 RGB `(B, 3, 256, 256)`，再贴回全身图。

### 2.2 音频与 mel 的对应关系

- **采样率**：16kHz；**fps**：50（即 20ms 一 chunk，320 样本）。  
- **Mel**：`wav2lip/audio.py` 的 `melspectrogram()`，参数在 `wav2lip/hparams.py`（n_fft=800, hop=200, 80 mel 等）。  
- **LipASR**：  
  - 使用滑动窗：`stride_left_size`(l)、`stride_right_size`(r)、中间有效区，对应 `app.py` 的 `-l -m -r`。  
  - 每 `run_step()` 取 `batch_size*2` 个 20ms 帧，拼成一段波形 → 一次 mel → 按 `mel_step_size=16` 切多段 mel chunk，放入 `feat_queue`。  
  - 推理端每消费一包 mel，就消费 `batch_size*2` 个音频帧，保证「一段 mel」与「一批视频帧 + 对应音频」对齐。

### 2.3 Avatar 数据结构（Wav2Lip）

- **运行时加载路径**：`./data/avatars/{avatar_id}/`（见 `lipreal.py` 的 `load_avatar()`）。  
- **目录约定**：  
  - `full_imgs/`：全身图序列，按帧序号命名（如 00000000.png）。  
  - `face_imgs/`：裁剪并 resize 到 256×256 的人脸图，与 full_imgs 一一对应。  
  - `coords.pkl`：每帧人脸在全身图中的 bbox `(y1, y2, x1, x2)` 列表。
- **生成方式**：运行 `wav2lip/genavatar.py`（需指定 `--avatar_id`、`--video_path` 等），输出在 **results/avatars/**；用于线上时需**拷贝或链接到 data/avatars/**，与 README 中「将解压的 avatar 拷到 data/avatars」一致。

### 2.4 静音与说话切换

- **静音**：当 `output_queue` 取到的 `batch_size*2` 帧全部为静音（type!=0 或静音填充）时，不调用 Wav2Lip，直接用 `frame_list_cycle[idx]` 的全身图（可加镜像循环），保证不说话时是静态或循环视频。  
- **说话**：有任一非静音帧则跑 Wav2Lip，用 `face_list_cycle` 的人脸 + mel，得到口型脸，再通过 `paste_back_frame()` 按 `coords.pkl` 贴回全身图。

### 2.5 线程与队列分工（Wav2Lip）

- **主 render 线程**（在 `webrtc` 的 worker 里）：循环调用 `asr.run_step()`，驱动音频进队列；必要时 sleep 控制推流压力。  
- **推理线程**：`lipreal.inference()`，读 `feat_queue` + `output_queue`，写 `res_frame_queue`。  
- **process_frames 线程**：读 `res_frame_queue`，合成最终帧 + 写 WebRTC 的 video/audio queue。  
- 三个队列：`feat_queue`（mel）、`output_queue`（原始音频帧+类型+事件）、`res_frame_queue`（合成帧+idx+对应音频）。

---

## 三、二次开发中你需要做的工作

以下按「必须 / 建议 / 可选」分类，便于你按优先级排期。

### 3.1 环境与依赖

- **Python / CUDA**：按 README，Python 3.10、PyTorch 2.5、CUDA 12.4 等已写明；二次开发若改推理框架或模型，需同步验证版本。  
- **模型文件**：Wav2Lip 需 `models/wav2lip.pth`（如从 wav2lip256.pth 重命名）；确保路径与 `app.py` 中 `load_model("./models/wav2lip.pth")` 一致。  
- **Avatar**：新数字人需先跑 `wav2lip/genavatar.py` 生成 `full_imgs`、`face_imgs`、`coords.pkl`，再放到 `data/avatars/{avatar_id}`；注意 genavatar 默认写 `results/avatars`，需拷贝到 `data/avatars` 或改脚本输出路径。

### 3.2 新增/更换 TTS / 语音输入

- **TTS**：在 `ttsreal.py` 增加新的 `BaseTTS` 子类，实现 `txt_to_audio()`，在 20ms chunk 粒度调用 `self.parent.put_audio_frame(stream[idx:idx+self.chunk], eventpoint)`；在 `basereal.py` 的 `BaseReal.__init_`_ 里增加 `opt.tts == 'xxx'` 的分支并实例化。  
- **语音输入**：若要从麦克风或上行音频驱动口型，需把 PCM（16kHz, 20ms）送入 `put_audio_frame()` 或 `put_audio_file()`；现有 `/humanaudio` 已支持上传音频文件，可复用或扩展为实时流。

### 3.3 新增/更换口型模型（如其他 Wav2Lip 变体）

- **模型结构**：若仍用 256×256、80×16 mel 的输入，可沿用当前 `lipasr` + `lipreal.inference` 的队列与 batch 设计，只替换 `load_model()` 和 `model(...)` 的调用（例如换权重或换 `wav2lip/models` 下另一版 Wav2Lip）。  
- **输入尺寸不同**：若新模型是 96 或 384 等，需要：  
  - 改 `wav2lip/hparams.py` 或 mel 尺寸；  
  - 改 `lipreal.py` 中 `face_list_cycle` 的 resize、`img_batch` 的构造以及 `paste_back_frame` 的尺寸逻辑；  
  - 同步改 `wav2lip/genavatar.py` 的 `--img_size` 与输出 face 尺寸。
- **warm_up**：`lipreal.warm_up(batch_size, model, 256)` 的第三个参数需与新模型的人脸分辨率一致。

### 3.4 多会话与并发

- **会话隔离**：每个 session 已有独立 `nerfreals[sessionid]` 和独立队列；若要做「每连接指定 avatar/音色」，只需在 `offer` 或后续接口里把 `avatar_id`/音色参数传入 `build_nerfreal(sessionid)` 或 `LipReal` 的构造（当前是全局 `opt` 一份，需改为 per-session 或从请求体读）。  
- **并发上限**：`app.py` 有 `max_session` 与 `nerfreals` 的清理（connectionstatechange）；若部署多进程/多机，需要自己做会话与负载均衡。

### 3.5 打断与实时控制

- **打断**：已提供 `flush_talk()` 与 `/interrupt_talk`；内部会清 TTS 队列和 ASR 队列。若要加强「唤醒词/按钮即停」，只需在前端或 ASR 里调用现有接口。  
- **事件回调**：音频帧带 `eventpoint`（如 TTS 的 start/end），在 `process_frames` 里通过 `video_track._queue.put((frame, eventpoint))` 传到 `PlayerStreamTrack.recv()`，再 `_player.notify(eventpoint)`；可在此挂接「每句开始/结束」等，做字幕或业务逻辑。

### 3.6 画质、分辨率与性能

- **分辨率**：当前 Wav2Lip 分支为 256×256 人脸；若要更高清，需用支持更大分辨率的模型或后处理（如超分），并注意显存与 `batch_size`。  
- **性能**：推理在独立线程中，batch 一次多帧；若 GPU 吃满，可适当调大 `batch_size`（注意延迟）；若 CPU 编码成为瓶颈，可看日志里 finalfps 与 inferfps 的对比，必要时优化编码或降分辨率。

### 3.7 前端与协议

- **前端**：`web/webrtcapi.html` + `client.js` 完成 offer/answer、收轨、发文本；若要做「对话/LLM 集成」，只需在发文本处改为调用你的后端再调 `/human`（type=chat 已走 `llm_response`）。  
- **LLM**：`llm.py` 使用 DashScope（qwen-plus），流式回复并按句号切句调用 `put_msg_txt`；若换其他 LLM 或 API，只需改 `llm.py` 的客户端与解析逻辑。

### 3.8 部署与运维

- **端口**：README 已说明 TCP 8010、UDP 1–65536（WebRTC）；若用 Docker 或 K8s，需映射这些端口。  
- **日志**：`logger` 中已有 inferfps、finalfps 等，便于排查「推理快但推流慢」或「队列堆积」问题。

---

## 四、关键文件索引（便于二次开发时定位）


| 功能                  | 文件路径                                     |
| ------------------- | ---------------------------------------- |
| 入口、路由、模型选择          | `app.py`                                 |
| WebRTC 推流、时间戳       | `webrtc.py`                              |
| 数字人基类、TTS 选择、合成与推流  | `basereal.py`                            |
| Wav2Lip 数字人逻辑、推理与贴图 | `lipreal.py`                             |
| 音频→mel、队列驱动         | `lipasr.py`                              |
| Wav2Lip 网络结构（256）   | `wav2lip/models/wav2lip_v2.py`           |
| Mel 与音频预处理          | `wav2lip/audio.py`, `wav2lip/hparams.py` |
| Avatar 生成（Wav2Lip）  | `wav2lip/genavatar.py`                   |
| TTS 实现              | `ttsreal.py`                             |
| LLM 流式对话            | `llm.py`                                 |
| 前端页面与 WebRTC        | `web/webrtcapi.html`, `web/client.js`    |


---

## 五、小结

- **架构**：HTTP + WebRTC 提供实时音视频；每个会话一个 BaseReal 子类实例，内部由 TTS → 音频队列 → ASR/mel → 口型模型 → 合成帧 → 推流，形成闭环。  
- **Wav2Lip**：使用 `wav2lip_v2` 的 256×256 模型；mel 由 `wav2lip.audio` 按 16kHz 与 hparams 计算；LipASR 负责滑动窗、mel 切块与双队列对齐；静音不推理，说话时人脸+mel 推理后贴回全身图。  
- **二次开发**：在不动现有项目代码的前提下，可按上述清单扩展 TTS、语音输入、口型模型、多会话/avatar、打断与事件、前端与 LLM，并注意 avatar 路径、模型输入尺寸与 warm_up、以及并发与性能调优。

按此文档即可在理解整体架构与 Wav2Lip 实现方式的基础上，有条理地开展二次开发工作。