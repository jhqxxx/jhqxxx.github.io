## Qwen2.5VL视觉编码器

#### 输入数据
* pixel_values: 
    ```text
    Tensor[[grid_t*grid_h*grid_w, 1176], bf16, cuda:0]
    ```
* image_grid_thw: 
    ```text
    image_grid_thw: [[grid_t, grid_h, grid_w]]
    Tensor[[1, 3], u32, cuda:0]
    ```

### PatchEmbed

[PatchEmbed](patch_embed.md)

### PositionEncode

[2D_RoPE](2D_RoPE.md)