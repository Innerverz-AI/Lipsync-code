{
    BASE: {
        MODEL_ID: 'LIPSYNC', 
        SAME_PROB: 0,
        BATCH_PER_GPU: 4,
        MAX_STEP: 600000,
        SAVE_ROOT: 'train_results',
        PACKAGES_PATH: '../PACKAGES',
        FRAME_NUM_PER_VIDEO: 10000,
        VAL_SIZE: 0,
        VAL_STEP: 0,
        IMG_SIZE: 256,
        RUN_ID: 'test',
        SYNC_STEP : 0,
        NO_LMKS : true,
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
        W_SYNC: 0,
    },

    CYCLE: {
        LOSS: 10,
        TRAIN_IMAGE: 1000,
        VALID_IMAGE: 1000,
        CKPT: 5000,
    },

    CKPT: {
        # ckpt path
        # load checkpoints from ./train_result/{ckpt_id}/ckpt/G_{ckpt_step}.pt
        # if ckpt_id is empty, load G_latest.pt and D_latest.pt
        TURN_ON: false,
        ID_NUM: null,
        STEP: null,
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
        MULTI_DET_VIDEOS: '/home/8414sys/Lipsync_training_last/multi_det_videos.txt',
        TRAIN_PATH:{
            GT_IMG:
                [
                    '/ssd2t/DATASET/VoxCeleb2/1id_1video/512over/*/*/*/aligned_imgs',
                    # '/ssd2t/DATASET/VoxCeleb2/Oct1_sync_crop_part2/*/*/*/aligned_imgs',
                ],
        },

        TEST_PATH:{
            GT_IMG:
                [
                    '/ssd2t/DATASET/VoxCeleb2/1id_1video/512over/*/*/*/aligned_imgs',
                    # '/ssd2t/DATASET/VoxCeleb2/Oct1_sync_crop_part2/*/*/*/aligned_imgs',
                ],
        }
    },




}