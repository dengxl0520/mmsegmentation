_base_ = '../stechonet_r50_sta_stechohead_20k_camus.py'

optimizer = dict( type='SDG', lr=0.001)
optim_wrapper = dict(optimizer=optimizer)
