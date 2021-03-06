from visdom import Visdom
import numpy as np

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main', server="http://130.63.188.108", port=12345):
        self.viz = Visdom(server=server, port=port)
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y, training_id, lr, x_label):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title="{}, training_id = {}, lr = {}".format(title_name, training_id, lr),
                xlabel=x_label,
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')