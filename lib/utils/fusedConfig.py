FUSED_CONFIG = {
    # Config for  EMBED Layers
    "GENDER_NUM_EMBED" : 2,
    "GENDER_EMBED_DIM" : 1,

    "AGE_NUM_EMBED" : 6,
    "AGE_EMBED_DIM" : 3,

    "BODY_SHAPE_NUM_EMBED" : 6,
    "BODY_SHAPE_EMBED_DIM" : 3,

    "ATTACHMENT_NUM_EMBED" : 11,
    "ATTACHMENT_EMBED_DIM" : 6,
    
    "UPPER_BODY_CLOTHING_NUM_EMBED" : 25,
    "UPPER_BODY_CLOTHING_EMBED_DIM" : 6,

    "LOWER_BODY_CLOTHING_NUM_EMBED" : 23,
    "LOWER_BODY_CLOTHING_EMBED_DIM" : 6,

    # Config for embed fc layers
    "EMBED_FC1_OUT" : 256,
    "EMBED_FC2_OUT" : 512,

    # Config for resnet fc layers
    "RESNET_FC1_OUT" : 256,
    "RESNET_FC2_OUT" : 512,

    "FC1_OUT" : 1024,
    "FC2_OUT" : 512,

    "lr" : 0.00076,
    "margin" : 0.11,
    "batchSize" : 64,
    
    "FUSED" : True,
    "FSI_TYPE" : "CONCAT_ATTN",
    "FSI_ALPHA" : 0.6
}