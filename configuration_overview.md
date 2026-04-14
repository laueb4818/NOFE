## regular
### dataset configurations
- sigma_method: 'variance', 'perplexity', 'ones'
- region: fixed region, random crop -> use script 'create_dataset.py' vs. 'create_dataset_random_crop.py'
    - for fixed region: could use full resolution with one sample per date and many dates or sample random subsets with multiple graphs per date -> use latter method for series of test runs
- node features: channel_only, channel_and_position, ~~position_only~~

naming pattern:
toyset_reg_[fix/rand]crop_[var/perp/ones]_[w/wo]Pos

needed sets:
- toyset_reg_fixcrop_ones_woPos (test1)
- toyset_reg_fixcrop_ones_wPos  (test2)


### model configuration
- model size: width, ker_width, depth
    - test model 1: w=8, k=32, d=3
    - test model 2: w=8, k=32, d=6
    - test model 3: w=8, k=128, d=3
    - test model 4: w=8, k=128, d=6
    - test model 5: w=64, k=32, d=3
    - test model 6: w=64, k=32, d=6
    - test model 7: w=64, k=128, d=3
    - test model 8: w=64, k=128, d=6

naming pattern: test model i: w=n1, k=n2, d=n3 -> 'testmodel[i]_w[n1]_kw[n2]_d[n3]'

### testing pattern:
1. basic model testing:
    - dataset:
        - sigma_method='ones'
        - region=fixed (but large and with random sampling, not full resolution)
        - node features=channel_only
    - model: test all models

-> choose model: 'model_A'

2. test channel config:
    - dataset:
        - sigma_method='ones'
        - region=fixed (but large and with random sampling, not full resolution)
        - node features=channel_and_position
    - model: model_A (choose best model from test 1)

-> choose node feature setup: 'feat_conf_A'

3. sigma testing:
    - dataset:
        - sigma_method={variance, perplexity}
        - region=fixed (but large and with random sampling, not full resolution)
        - node features='feat_conf_A' (result from test 2)
    - model: model_A (choose best model from test 1)

4. samppling method:
    - dataset:
        - sigma_method= result test 3
        - region=random_crop
        - node features='feat_conf_A' (result from test 2)
    - model: model_A (choose best model from test 1) + choose a bigger model too because this might be heavier calculations



## subgraph
### dataset configurations
- sigma_method: 'variance', 'perplexity', 'ones'
- region: fixed region, random crop -> use script 'create_dataset.py' vs. 'create_dataset_random_crop.py'
- node features: channel_only, channel_and_position, ~~position_only~~
- output: ~~regular_subset~~, random
