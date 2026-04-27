def debug_dinov3_output():
    """调试DINOv3的实际输出结构"""
    from transformers import AutoModel
    import torch
    
    # 加载你的DINOv3模型
    
    model = AutoModel.from_pretrained(
        "backbone",
        torch_dtype=torch.float32,
        local_files_only=True,
        output_hidden_states=True  # 关键：获取所有隐藏层
    )
    
    # 创建测试输入
    batch_size = 1
    height, width = 256, 256
    x = torch.randn(batch_size, 3, height, width)
    
    print("=== DINOv3输出结构调试 ===")
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    with torch.no_grad():
        outputs = model(x)
    
    print(f"输出类型: {type(outputs)}")
    
    # 检查输出属性
    if hasattr(outputs, '__dict__'):
        print("输出属性:", list(outputs.__dict__.keys()))
    
    # 检查是否有hidden_states
    if hasattr(outputs, 'hidden_states'):
        hidden_states = outputs.hidden_states
        print(f"\n隐藏状态数量: {len(hidden_states)}")
        for i, state in enumerate(hidden_states):
            print(f"层 {i}: 形状 {state.shape}")
    else:
        print("没有hidden_states属性")
    
    # 检查其他可能的属性
    for attr in ['last_hidden_state', 'pooler_output', 'attentions']:
        if hasattr(outputs, attr):
            value = getattr(outputs, attr)
            print(f"{attr}: 形状 {value.shape if hasattr(value, 'shape') else type(value)}")
    
    # 检查是否是元组
    if isinstance(outputs, tuple):
        print(f"\n输出是元组，长度: {len(outputs)}")
        for i, item in enumerate(outputs):
            print(f"元组[{i}]: 类型 {type(item)}, 形状 {item.shape if hasattr(item, 'shape') else 'N/A'}")

# 运行调试
if __name__ == "__main__":
    debug_dinov3_output()