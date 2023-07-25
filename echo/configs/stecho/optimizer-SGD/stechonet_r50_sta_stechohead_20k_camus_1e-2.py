_base_ = '../stechonet_r50_sta_stechohead_20k_camus.py'

optimizer = dict(
    _delete_=True,
    type='SGD', 
    lr=0.01,
    momentum=0.9, 
    weight_decay=0.0005)
optim_wrapper = dict(
        _delete_=True,
        optimizer=optimizer)
