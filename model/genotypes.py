from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

DISC_NWOT = Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6))
DISC_ACC = Genotype(normal=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))
DISC_LOSS = Genotype(normal=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
DISC_CKPT_ACC = Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 2), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))
DISC_CKPT_LOSS = Genotype(normal=[('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))
CONT_ACC_STEP = Genotype(normal=[('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('avg_pool_3x3', 3), ('avg_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6))
CONT_ACC_GRAD_STEP = Genotype([('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3), ('max_pool_3x3', 1), ('dil_conv_3x3', 1), ('avg_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 2), ('dil_conv_3x3', 1), ('avg_pool_3x3', 3), ('dil_conv_3x3', 1)], reduce_concat=range(2, 6))
CONT_LOSS_STEP = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 2), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('sep_conv_3x3', 1), ('max_pool_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 4), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))
CONT_LOSS_GRAD_STEP = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 3), ('avg_pool_3x3', 1), ('dil_conv_5x5', 3), ('max_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 3), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))


ARCHITECTURES = {
    "DISC_NWOT": DISC_NWOT,
    "DISC_ACC": DISC_ACC,
    "DISC_LOSS": DISC_LOSS,
    "DISC_CKPT_ACC": DISC_CKPT_ACC,
    "DISC_CKPT_LOSS": DISC_CKPT_LOSS,
    "CONT_ACC_STEP": CONT_ACC_STEP,
    "CONT_ACC_GRAD_STEP": CONT_ACC_GRAD_STEP,
    "CONT_LOSS_STEP": CONT_LOSS_STEP,
    "CONT_LOSS_GRAD_STEP": CONT_LOSS_GRAD_STEP,
}
