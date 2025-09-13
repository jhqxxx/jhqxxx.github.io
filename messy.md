cross_attention: q与k,v来自不同的输入，k,v来自同一个输入
softmax((WqS2,)(WkS1 ))WvS1

window_size: 112
merge_size: 2
patch_size: 14

grid: (1, 32, 90)

vit_merger_window_size = window_size / merge_size / patch_size = 4
merge_grid_h = grid_h / merge_size = 16
merge_grid_w = grid_w / merge_size = 45

window_index = arange(0, grid_t\*merge_grid_h\*merge_grid_w).reshape((grid_t, merge_grid_h, merge_grid_w)) ->(1, 16, 45)

如果merge_grid_h/merge_grid_w % vit_merger_window_size != 0,需要padding
padding元素为-100 -> (1, 16, 48)

num_window_h = (llm_grid_h + pad_h) / vit_merger_window_size = 4
num_window_w = (llm_grid_w + pad_w) / vit_merger_window_size = 12

window_index_padded = window_index_padded.reshape(grid_t, num_window_h, vit_merger_window_size, num_window_w, vit_merger_window_size) -> (1, 4, 4, 12, 4)

window_index_padded = window_index_padded.permute((0, 1, 3, 2, 4)) -> (grid_t, num_window_h, num_window_w, vit_merger_window_size, vit_merger_window_size) -> (1, 4, 12, 4, 4) 

根据元素!=-100,计算窗口（vit_merger_window_size， vit_merger_window_size）内不是padding元素的数量 -> (1, 4, 14) 再展平得到seq_len每个窗口序列长度，无填充的窗口->16，有填充的窗口->4
计算seq_len的元素累计和，再乘上merge_size，得到每个窗口没merge时的token数量->merge前的token数量
cu_window_seqlens: [  64,  128,  192,  256,  320,  384,  448,  512,  576,  640,  704,  720,  784,
  848,  912,  976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1440, 1504, 1568,
 1632, 1696, 1760, 1824, 1888, 1952, 2016, 2080, 2144, 2160, 2224, 2288, 2352,
 2416, 2480, 2544, 2608, 2672, 2736, 2800, 2864, 2880]

window_index_padded展平，这时窗口内的索引已经处理到一起，将不是padding的索引取出-> merge后的窗口索引

merge_unit = merge_size<sup>2</sup> = 4

hidden_state.reshape((seq_len / merge_unit, merge_unit, hidden_dim))
使用窗口索引重新排列hidden_state，再reshape回去(seq_len, hidden_dim)
旋转位置编码也进行一样的操作
再将theta重复一份，得到head_dim维度的theta，再计算sin,cos

当全attenion时，使用无窗口的索引
[0, grid_h*grid_W]
cu_seqlens: 
 [   0, 2880]

 
1. 窗口索引生成：将输入的特征网格划分成多个窗口，并为每个窗口内的有效token分配唯一索引->将窗口内的数据处理到一起

2. 记录每个窗口中有效token的数量，用于后续注意力计算时定位窗口边界->生成attention_mask