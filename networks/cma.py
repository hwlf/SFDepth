from networks.depth_decoder import DepthDecoder
from networks.multi_embedding import MultiEmbedding
from networks.seg_decoder import SegDecoder
from utils.depth_utils import *



from networks.seg_decoder import SegDecoder
from utils.depth_utils import *

class CMA(nn.Module):
    def __init__(self, num_ch_enc=None, opt=None):
        super(CMA, self).__init__()

        self.scales = opt.scales
        cma_layers = opt.cma_layers
        self.opt = opt

        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        in_channels_list = [32, 64, 128, 256, 16]

        # 仅保留语义解码器
        self.seg_decoder = SegDecoder(num_ch_enc, num_output_channels=19,
                                      scales=[0])

        att_d_to_s = {}
        att_s_to_d = {}
        for i in cma_layers:
            att_s_to_d[str(i)] = MultiEmbedding(in_channels=in_channels_list[i],
                                                num_head=opt.num_head,
                                                ratio=opt.head_ratio)
        self.att_s_to_d = nn.ModuleDict(att_s_to_d)

    def forward(self, input_features):
        seg_outputs = {}
        x = input_features[-1]
        x_s = None

        for i in range(4, -1, -1):
            if x_s is None:
                x_s = self.seg_decoder.decoder[-2 * i + 8](x)
            else:
                x_s = self.seg_decoder.decoder[-2 * i + 8](x_s)

            x_s = [upsample(x_s)]

            if i > 0:
                x_s += [input_features[i - 1]]

            x_s = torch.cat(x_s, 1)
            x_s = self.seg_decoder.decoder[-2 * i + 9](x_s)

            if self.opt.sgt:
                seg_outputs[('s_feature', i)] = x_s
            if i in self.scales:
                if i == 0:
                    outs = self.seg_decoder.decoder[10 + i](x_s)
                    seg_outputs[("seg_logits", i)] = outs[:, :19, :, :]

        return seg_outputs
