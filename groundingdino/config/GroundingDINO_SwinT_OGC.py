batch_size = 1 # 批量大小
modelname = "groundingdino" # 模型名称
backbone = "swin_T_224_1k" # 主干网络
position_embedding = "sine" # 位置编码类型
pe_temperatureH = 20 # 位置编码温度参数 H
pe_temperatureW = 20 # 位置编码温度参数 W
return_interm_indices = [1, 2, 3] # 返回中间层的索引
backbone_freeze_keywords = None # 冻结主干网络的关键字
enc_layers = 6 # 编码器层数
dec_layers = 6 # 解码器层数
pre_norm = False # 是否使用 pre-norm
dim_feedforward = 2048 # 前馈网络的维度
hidden_dim = 256 # 隐藏层维度
dropout = 0.0 # Dropout 比例
nheads = 8 # 注意力头数
num_queries = 900 # 查询的数量
query_dim = 4 # 查询的维度
num_patterns = 0 # 模式数量
num_feature_levels = 4 # 特征层级数量
enc_n_points = 4 # 编码器中的采样点数
dec_n_points = 4 # 解码器中的采样点数
two_stage_type = "standard" # 两阶段类型
two_stage_bbox_embed_share = False # 两阶段 bbox 嵌入是否共享
two_stage_class_embed_share = False # 两阶段类别嵌入是否共享
transformer_activation = "relu" # Transformer 激活函数
dec_pred_bbox_embed_share = True # 解码器预测 bbox 嵌入是否共享
dn_box_noise_scale = 1.0 # DN 盒噪声比例
dn_label_noise_ratio = 0.5 # DN 标签噪声比例
dn_label_coef = 1.0 # DN 标签系数
dn_bbox_coef = 1.0 # DN bbox 系数
embed_init_tgt = True # 嵌入初始化目标
dn_labelbook_size = 2000 # DN 标签簿大小
max_text_len = 256 # 最大文本长度
text_encoder_type = "bert-base-uncased" # 文本编码器类型
use_text_enhancer = True # 是否使用文本增强器
use_fusion_layer = True # 是否使用融合层
use_checkpoint = True # 是否使用检查点
use_transformer_ckpt = True # 是否使用 Transformer 检查点
use_text_cross_attention = True # 是否使用文本交叉注意力
text_dropout = 0.0 # 文本 Dropout 比例
fusion_dropout = 0.0 # 融合 Dropout 比例
fusion_droppath = 0.1 # 融合 Droppath 比例
sub_sentence_present = True # 是否存在子句
