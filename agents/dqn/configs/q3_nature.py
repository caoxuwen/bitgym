class config():
    # env config
    render_train     = False
    render_test      = False
    overwrite_render = True
    record           = False
    high             = 255.

    # output config
    output_path  = "results/q3_nature/"
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"

    # model and training config
    num_episodes_test = 50
    grad_clip         = True
    clip_val          = 10
    saving_freq       = 25000
    log_freq          = 50
    eval_freq         = 25000
    soft_epsilon      = 0.05

    # hyper params
    nsteps_train       = 200000
    batch_size         = 32
    buffer_size        = 500
    target_update_freq = 500
    #gamma              = 0.99
    gamma              = 1
    learning_freq      = 4
    state_history      = 50
    lr_begin           = 0.00025
    lr_end             = 0.0001
    lr_nsteps          = nsteps_train/2
    eps_begin          = 1
    eps_end            = 0.1
    eps_nsteps         = nsteps_train/2
    learning_start     = 200
    
