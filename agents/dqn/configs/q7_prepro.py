import time

class config():
    # env config
    render_train     = False
    render_test      = True
    env_name         = "Pong-v0"
    overwrite_render = False
    record           = False

    # output config
    millis = str(int(round(time.time() * 1000)))
    output_path  = "results/q7_prepro_"+millis+"/"
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path + "monitor/"
    load_path = "results2/cnn_50_price_norm/"


    # model and training config
    num_episodes_test = 5
    grad_clip         = False
    clip_val          = 50
    saving_freq       = 100000
    log_freq          = 50
    eval_freq         = 100000
    record_freq       = 250000
    soft_epsilon      = 0.05

    # nature paper hyper params
    nsteps_train       = 5000000
    batch_size         = 128
    buffer_size        = 1000000
    target_update_freq = 10000
    gamma              = 1
    learning_freq      = 4
    state_history      = 50
    skip_frame         = 4
    lr_begin           = 0.00025
    lr_end             = 0.00005
    lr_nsteps          = nsteps_train/2
    eps_begin          = 1
    eps_end            = 0.1
    eps_nsteps         = 1000000
    learning_start     = 50000
    
