from .vse import VSE
from .scan import SCAN
from .bfan import BFAN
from .imram import IMRAM
from .naaf import NAAF
from .sgraf import SGRAF


__factory = {
    'vse': VSE,
    'scan': SCAN,
    'bfan': BFAN,
    'imram': IMRAM,
    'naaf': NAAF,
    'sgraf': SGRAF,
}


def get_attention_encoder(name, opt):
    if name not in __factory.keys():
        raise KeyError("Invalid value of opt.attn_name")
    return __factory[name](opt)