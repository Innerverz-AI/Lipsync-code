{
    BASE: {
        MODEL_ID: 'LIPSYNC', 
        SAME_PROB: 0,
        BATCH_PER_GPU: 3,
        MAX_STEP: 100000,
        SAVE_ROOT: 'train_results',
        PACKAGES_PATH: '../PACKAGES',
        FRAME_NUM_PER_VIDEO: 10000,
        VAL_SIZE: 0,
        VAL_STEP: 0,
        IMG_SIZE: 256,
        RUN_ID: 'test',
        OPT_URL: ['taylorsync1', 'taylorsync2', 'taylorsync3', 'taylorsync4', 'taylorsynctrain1', 'taylorsynctrain2'],
        NO_LMKS : false,
    },

    # weight of loss
    LOSS: {
        W_ADV: 1,
        W_VGG: 0,
        W_ID: 0,
        W_L1: 10,
        W_RECON: 0,
        W_CYCLE: 0,
        W_FEAT: 10,
        W_LPIPS: 100,
        W_SYNC: 0.1,
        W_REG_SYNC: 0,
        W_W_REG: 0.1,
    },

    CYCLE: {
        LOSS: 10,
        TRAIN_IMAGE: 500,
        VALID_IMAGE: 500,
        CKPT: 2000,
    },

    CKPT: {
        # ckpt path
        # load checkpoints from ./train_result/{ckpt_id}/ckpt/G_{ckpt_step}.pt
        # if ckpt_id is empty, load G_latest.pt and D_latest.pt
        TURN_ON: false,
        ID_NUM: null,
        STEP: null,
        G_PRE_PATH: 'ckpt/Oct_1016_dense_G_00315000.pt',
        D_PRE_PATH: 'ckpt/Oct_1016_dense_D_00315000.pt',
    },

    WANDB: {
        TURN_ON: true,
        ALERT_THRES: 1000,
    },

    OPTIMIZER: {
        TYPE: 'Adam', # [Ranger, Adam]
        BETA: [0.0, 0.999], # default: Adam (0.9, 0.999) / Ranger (0.95, 0.999)
        LR_G: 0.0001,
        LR_D: 0.00001,
    },

    NETWORK: {
        REF_INPUT: false,
        ADD_SKIP: false,
    },

    DATASET: {
        TRAIN_PATH:{
            GT_IMG:
                [
                    '/ssd2t/DATASET/VoxCeleb2/1id_1video/512over/*/*/*/aligned_imgs',
                    '/ssd2t/DATASET/VoxCeleb2/Oct1_sync_crop_part2/*/*/*/aligned_imgs',
                ],
            OPT_IMG:
                [
                    '/ssd2t/DATASET/VoxCeleb2/demo_video/512over/*/*/*/aligned_imgs',
                ],
        },

        TEST_PATH:{
            GT_IMG:
                [
                    '/ssd2t/DATASET/VoxCeleb2/1id_1video/512over/*/*/*/aligned_imgs',
                    '/ssd2t/DATASET/VoxCeleb2/Oct1_sync_crop_part2/*/*/*/aligned_imgs',
                ],
            OPT_IMG:
                [
                    '/ssd2t/DATASET/VoxCeleb2/demo_video/512over/*/*/*/aligned_imgs',
                ]
        }
    }

}