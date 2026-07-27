import os

def get_params(argv='1'):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=False,
        finetune_mode=False,  # Finetune on existing model, requires the pretrained model path set - pretrained_model_weights
        pretrained_model_weights='/projects/0/prjs1261/seld/TAU2020/2_1_dev_split0_accdoa_foa_model.h5',

        # INPUT PATH
        dataset_dir='/projects/0/prjs1261/seld/TAU2020',

        # OUTPUT PATHS
        feat_label_dir='/projects/0/prjs1261/seld/TAU2020_labels_audio_sphere',

        model_dir='/projects/0/prjs1261/seld/TAU2020_saved_models_audio_sphere',            # Dumps the trained models and training curves in this folder
        dcase_output_dir='/projects/0/prjs1261/seld/TAU2020_results_audio_sphere',    # recording-wise results are dumped in this path.

        # DATASET LOADING PARAMETERS
        mode='dev',         # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',      # 'foa' - ambisonic or 'mic' - microphone signals

        # FEATURE PARAMS
        fs=32000,
        hop_len_s=0.01,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=128,

        # We do not use salsalite
        use_salsalite=False,  # Used for MIC dataset only. If true use salsalite features, else use GCC features
        fmin_doa_salsalite=50,
        fmax_doa_salsalite=2000,
        fmax_spectra_salsalite=9000,

        # MODEL TYPE
        model='audio_sphere',   # 'seldnet' - baseline CNN SELDnet, 'audio_sphere' - pre-trained AudioSphere encoder

        # build_model() reads these two; ablation tasks (100-111) set them from
        # the table above, legacy tasks fall back to audio_sphere_ckpt at the
        # bottom of this function.
        ckpt_path=None,             # resolved below if left None
        model_class='AudioSphere',  # AudioSphere | AudioSphereChannelMasked | AudioSphereIVCosine

        freeze_encoder=True,    # freeze the AudioSphere encoder (pass to AudioSphereSELD in build_model!)

        multi_accdoa=False,  # False - Single-ACCDOA or True - Multi-ACCDOA
        thresh_unify=15,     # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.

        # DNN MODEL PARAMETERS
        label_sequence_length=20,    # Feature sequence length
        batch_size=128,              # Batch size
        dropout_rate=0.05,           # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,            # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 2],       # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer
        self_attn=True,
        nb_heads=8,
        nb_self_attn_layers=2,

        nb_rnn_layers=2,
        rnn_size=128,

        nb_fnn_layers=1,
        fnn_size=128,             # FNN contents, length of list = number of layers, list value = number of nodes

        nb_epochs=100,            # Train for maximum epochs
        lr=1e-3,

        # METRIC
        average='macro',          # Supports 'micro': sample-wise average and 'macro': class-wise average
        lad_doa_thresh=20,

    )

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

    elif argv == '2':
        print("FOA + ACCDOA + SELDNET BASELINE\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False
        params['model'] = 'seldnet'

    elif argv == '3':
        print("FOA + multi ACCDOA + SELDNET BASELINE\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'seldnet'

    elif argv == '4':
        print("FOA + ACCDOA + AUDIOSPHERE\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False
        params['model'] = 'audio_sphere'

    elif argv == '5':
        print("FOA + multi ACCDOA + AUDIOSPHERE\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['model'] = 'audio_sphere'


    params['patience'] = int(params['nb_epochs'])     # Stop training if patience is reached
    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution  # 20 * 10 = 200
    params['t_pool_size'] = [feature_label_resolution, 1, 1]     # CNN time pooling (baseline SELDnet only)

    if '2020' in params['dataset_dir']:
        params['unique_classes'] = 14
    elif '2021' in params['dataset_dir']:
        params['unique_classes'] = 12
    elif '2022' in params['dataset_dir']:
        params['unique_classes'] = 13
    elif '2023' in params['dataset_dir']:
        params['unique_classes'] = 13

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params