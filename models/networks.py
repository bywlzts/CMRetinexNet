import torch
import models.archs.DMRetinexLLIE as DMRetinexLLIE

# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'DMRetinexLLIE':
        netG = DMRetinexLLIE.DMRetinexLLIE(nf=opt_net['nf'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG

