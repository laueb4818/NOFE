## TODO
- [ ] restructure to reduce number of individual files and subdirectories (see 'suggested_resturcturing.md')
- [ ] create a script combining subgraph and random crop
    - [x] ERA5
    - [ ] dMRI
    - [ ] THOR
    - [ ] alphaEarth
- [ ] finish Generator methods that have been skipped so far, e.g. different sampling methods
- [ ] implement data generator for liver data
- [ ] implement random crop generation for other tracks as well

## ISSUES
- [ ] large ERA5 dataset (random crop) yields NaN values for the loss calculation on springfield while it seems to work fine locally on mac. Potential reason: indexing need to be done via torch.long

## IDEAS
- [ ] applications: predict ERA5 variables based on NOFE predictions from THOR