from graphviz import Digraph
from torch.autograd import Variable
from unet import UNet


def make_dot(var):
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='10',
                     ranksep='0.1',
                     height='0.5')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="30,14"))
    seen = set()

    def add_nodes(var):
        if var not in seen:
            if isinstance(var, Variable):
                value = '(' + (', ').join(['%d' % v for v in var.size()]) + ')'
                dot.node(str(id(var)), str(value), fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'previous_functions'):
                for u in var.previous_functions:
                    dot.edge(str(id(u[0])), str(id(var)))
                    add_nodes(u[0])
    add_nodes(var.creator)
    return dot
unet = UNet(opt.colordim)
x = Variable(torch.rand(1, 1, 572, 572)).cuda()
h_x = unet(x)
make_dot(h_x)
