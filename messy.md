# VoxCPM
## VAE
* encoder: 16kHz -> [2, 5, 8, 8]:640倍下采样 -> 25Hz的audio latents
* decoder: 25Hz -> []:上采样 -> 16kHz

## LocEnc
* 将audio latens压缩为紧凑的音频嵌入
* 得到历史音频上下文

## Text-Semantic Language Model
* 捕捉高级语言结构
* 生成适合上下文的语音模式
* MiniCPM4：从原始文本中进行丰富的上下文理解和更自然的音律预测
* 字符级BPE分词：缓解词汇稀疏问题
* 输入：音频嵌入+文本嵌入
* 通过处理文本标记和历史音频上下文，TSLM 学习生成语义内容和
韵律结构，这些内容和结构在整个话语过程中自然演变，反映出潜在的语言含义，
而不是简单地将音素映射到声学特征

## fsq
* 将连续隐藏状态从 TSLM 投影到结构化格子上，以创建半离散表示

## Residual Acoustic Language Model (RALM)
* 重建那些传统离散方法为了提高稳定性而牺牲的细微声音特征
* 恢复说话人身份、频谱精细结构和微韵律变化

## LocDiT



## Audio VAE
* encoder_dim: 128
* encoder_rates: [2, 5, 8, 8]
* laten_dim: 64
* decoder_dim: 1536
* decoder_rates: [8, 8, 5, 2]