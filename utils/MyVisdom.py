from visdom import Visdom
import numpy as np

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main', server="http://130.63.188.108", port=12345):
        self.viz = Visdom(server=server, port=port)
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y, batch_size, lr):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title="{}, batch size = {}, lr = {}".format(title_name, batch_size, lr),
                xlabel='Iterations',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')