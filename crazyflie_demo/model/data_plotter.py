import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
import numpy as np
from crazyflie_dynamics import CrazyflieDynamics
import crazyflie_param as P

plt.ion()  # enable interactive drawing

# self.state = np.array([
#     [P.x0],     # 0
#     [P.y0],     # 1
#     [P.z0],     # 2
#     [P.psi0],   # 3
#     [P.theta0], # 4
#     [P.phi0],   # 5
#     [P.u0],     # 6
#     [P.v0],     # 7
#     [P.w0],     # 8
#     [P.r0],     # 9
#     [P.q0],     # 10
#     [P.p0],     # 11
# ])

class DataPlotter:
    def __init__(self):
        # Number of subplots
        self.num_rows = 6
        self.num_cols = 1

        # Create figure and axis handles
        self.fig, self.ax = plt.subplots(self.num_rows, self.num_cols, sharex=True, figsize=(6,12))

        # Instatiate lists to hold the time and data histories
        self.time_history = [] # time
        self.xref_history = [] # reference position x_c
        self.x_history = [] # position x
        self.yref_history = [] # reference position y_c
        self.y_history = [] # position y
        self.zref_history = [] # reference position z_c
        self.z_history = [] # position z

        # Added rotational components
        self.psiref_history = []
        self.psi_history = []
        # self.thetaref_history = []
        self.theta_history = []
        # self.phiref_history = []
        self.phi_history = []

        # Create a handle for every subplot
        self.handle = []
        self.handle.append(MyPlot(self.ax[0], ylabel='x(m)', title='CF Data'))
        self.handle.append(MyPlot(self.ax[1], ylabel='y(m)'))
        self.handle.append(MyPlot(self.ax[2], ylabel='z(m)'))

        self.handle.append(MyPlot(self.ax[3], ylabel='psi(deg)'))
        self.handle.append(MyPlot(self.ax[4], ylabel='theta(deg)'))
        self.handle.append(MyPlot(self.ax[5], xlabel='t(s)', ylabel='phi(deg)'))

    def update(self, t, reference, states, ctrl):
        # Update the time history of all plot variables
        self.time_history.append(t)
        self.xref_history.append(reference.item(0))
        self.x_history.append(states.item(0))
        self.yref_history.append(reference.item(1))
        self.y_history.append(states.item(1))
        self.zref_history.append(reference.item(2))
        self.z_history.append(states.item(2))

        # Add angular control views
        self.psiref_history.append(reference.item(3))
        self.psi_history.append(states.item(3))
        # self.thetaref_history.append(reference.item(4))
        self.theta_history.append(states.item(4))
        # self.phiref_history.append(reference.item(5))
        self.phi_history.append(states.item(5))

        # Update the plots with associated handles
        self.handle[0].update(self.time_history, [self.x_history, self.xref_history])
        self.handle[1].update(self.time_history, [self.y_history, self.yref_history])
        self.handle[2].update(self.time_history, [self.z_history, self.zref_history])

        self.handle[3].update(self.time_history, [self.psi_history, self.psiref_history])
        self.handle[4].update(self.time_history, [self.theta_history])
        self.handle[5].update(self.time_history, [self.phi_history])

class MyPlot:
    def __init__(self, ax, xlabel='', ylabel='', title='', legend=None):
        """
        ax - handle to the axes of the figure
        legend - a tuple of strings that identify the data
        """
        self.legend = legend
        self.ax = ax
        self.colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
        self.line_styles = ['-', '-', '--', '-.', ':']

        self.line = []

        # Configure the axes
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel(xlabel)
        self.ax.set_title(title)
        self.ax.grid(True)

        # Keeps track of initialization
        self.init = True
    
    def update(self, time, data):
        """
        Adds data to the plot
        Time is a list
        Data is a list of lists, each list is a line on the plot
        """
        if self.init == True:
            for i in range(len(data)):
                self.line.append(Line2D(time, data[i],
                    color=self.colors[np.mod(i, len(self.colors) - 1)],
                    ls=self.line_styles[np.mod(i, len(self.line_styles) - 1)],
                    label=self.legend if self.legend != None else None))
                self.ax.add_line(self.line[i])
            self.init = False
            # add legend if one is specified
            if self.legend != None:
                plt.legend(handles=self.line)
        else: # Add new data to the plot
            # Updates the x and y data of each line.
            for i in range(len(self.line)):
                self.line[i].set_xdata(time)
                self.line[i].set_ydata(data[i])

        # Adjusts the axis to fit all of the data
        self.ax.relim()
        self.ax.autoscale()

# Run some tests
if __name__ == "__main__":
    data_plot = DataPlotter()
    cf = CrazyflieDynamics()

    # PWM is 0 - 65535
    # RPM is 4070.3 - 21666.4 Eq. 2.6.1
    # omega is 426.2 - 2268.9
    u = np.array([
        [1900],
        [2000],
        [1900],
        [2000],
    ])
    t = 0
    r = np.array([
        [0.0], # x
        [0.0], # y
        [0.5], # z
        # [0.5], # psi
        # [0.5], # theta 
        # [0.5], # phi
    ])
    
    t = P.t_start
    while t < P.t_end:
        # Propagate dynamics at rate Ts
        t_next_plot = t + P.t_plot
        while t < t_next_plot:
            y = cf.update(u)
            t = t + P.Ts
        # animation.update(cf.state)
        data_plot.update(t, r, cf.state, u)
        plt.pause(0.2)
    
    # Keeps the program from closing until the user presses a button.
    print('Press key to close')
    plt.waitforbuttonpress()
    plt.close()