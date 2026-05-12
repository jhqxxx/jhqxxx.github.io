# 初识常见的音频系统
**音频数据的输入输出**
## Linux
### ALSA
**内核级音频架构**
* Linux 内核自带的底层音频驱动框架
* 直接操作声卡硬件，提供最基本的音频捕获和播放功能
* 所有其他音频系统最终都依赖 ALSA 与硬件通信
* 缺点：不支持多应用同时占用声卡、功能较基础
#### ubuntu 依赖安装
```bash
sudo apt install libasound2-dev
```
#### Fedora 依赖安装
```bash
sudo dnf install alsa-lib-devel
```

### JACK
**专业音频工作站**
* 面向音乐制作和专业音频处理的低延迟音频服务器
* 核心优势：极低的音频延迟（可低至 1-5ms）
* 支持应用间灵活的音频路由（类似物理调音台的跳线功能）
* 常见于：Ardour、LMMS、Guitarix 等专业软件
* 缺点：配置复杂

| 平台   | 支持状态                    |
| ------- | ----------------------- |
| Linux   | ✅ 原生主力，功能最完整       |
| macOS   | ⚠️ 可用但维护较弱，CoreAudio 更优 |
| Windows | ❌ 非官方移植，配置困难，几乎无人使用     |

### PulseAudio
**桌面音频服务器（传统主流）**
* 2008 年起成为大多数 Linux 发行版的默认音频系统
* 解决 ALSA 的多应用并发问题，实现软件混音
* 提供网络音频传输、 per-application 音量控制等功能
* 但架构较老旧，延迟较高
* 正逐步被 PipeWire 取代

### PipeWire
**新一代统一音频架构（现代主流）**
* 由 Red Hat 开发，2015 年发布，近年快速普及
* 设计目标：统一音频（PulseAudio + JACK）+ 视频（摄像头）+ 低延迟专业音频
* 关键优势：
    * 兼容 PulseAudio 和 JACK 的应用
    * 比 PulseAudio 更低的延迟，比 JACK 更易用
    * 更好的蓝牙音频支持（如 LDAC、aptX）
    * 现代安全架构（Flatpak 应用沙盒支持）
* Fedora、Ubuntu 22.10+、Debian 12+ 等已默认采用

## Windows
### WASAPI
**Windows 的现代标准音频 API**
* 取代老旧的 DirectSound、WaveOut
* 两种模式：
    * 共享模式（默认）：系统混音，兼容性好
    * 独占模式：绕过系统混音，直接访问硬件，延迟更低

### ASIO
Windows 专业音频驱动协议/标准
* 绕过 Windows 音频系统，直连硬件
* 延迟：可低至 1-5ms（实时演奏无感知延迟）
* 支持多通道 I/O（专业声卡 8+ 输入输出）
* 位精确传输，无系统重采样

| 特性   | WASAPI 独占模式  | ASIO          |
| ---- | ------------ | ------------- |
| 延迟   | 较低（~10-30ms） | 极低（<10ms）     |
| 兼容性  | 通用，无需专用驱动    | 需声卡厂商 ASIO 驱动 |
| 适用场景 | 高保真播放、半专业制作  | 专业录音棚、实时演奏    |

## MacOS
### Core Audio
Apple 生态的音频框架
* 业界公认最优秀的音频架构之一
* 延迟极低（iOS 可达 2-5ms）
* 统一 API 覆盖消费级到专业级
* 硬件深度优化（Apple Silicon 专属优化）

## Android
### AAudio 
**Android 原生低延迟音频 API**
* 音乐制作 App
* 游戏音频引擎
* 实时音频处理应用

## Browser
### Web Audio API 
* 浏览器原生的音频处理接口
* 支持实时合成、音效、可视化、空间音频

## rust音频IO处理
选择 cpal库

