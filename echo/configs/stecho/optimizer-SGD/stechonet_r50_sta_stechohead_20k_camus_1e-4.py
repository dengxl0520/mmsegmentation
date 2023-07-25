_base_ = '../stechonet_r50_sta_stechohead_20k_camus.py'

optim_wrapper = dict(
    _delete_=True,
    optimizer=dict(
        type='SGD', 
        lr=0.0001,
        momentum=0.9, 
        weight_decay=0.0005)
    )
