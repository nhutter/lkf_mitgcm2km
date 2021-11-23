# lkf_mitgcm2km
LKF detection and tracking algorithms for MITgcm 2-km Pan-Arctic simulation

Requirements: 
- install `lkf_tools` from https://github.com/nhutter/lkf_tools
- create conda environment with all necessary packages

Changes required on different machines:
- add [path](https://github.com/nhutter/lkf_mitgcm2km/blob/b7803680823ab1595f416176afe7c7bfa637d54b/gen_dataset_model.py#L8) to `lkf_tools`
- adapt [path](https://github.com/nhutter/lkf_mitgcm2km/blob/b7803680823ab1595f416176afe7c7bfa637d54b/gen_dataset_model.py#L21) to grid files of the simulations in binary format (`DXC.bin`,`DYC.bin`, etc.)
- adapt [path](https://github.com/nhutter/lkf_mitgcm2km/blob/b7803680823ab1595f416176afe7c7bfa637d54b/gen_dataset_model.py#L22) to all output files of the simulations in binary format. `SIuice`, `SIvice`, and `SIarea` are required.

