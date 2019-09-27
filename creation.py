import numpy as np
import random
import math
import collections
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation
import json
import copy

import wordvectors.physicaldata.tools as tools

class Energy_Modes(Enum):
    neighbouring = 1
    rectangular = 2

class LogTypes(Enum):
    NoStateChange = 1
    RowNoStateChange = 2


def change_file_location (properties_location, new_location):
    """
    Helper function to change the string in the properties file that points
    to the location of the simulated data. This function can be usefull if one
    changes computer or the file location in general.
    Parameters
    ----------
    properties_location :
        String specifying where the properties are saved on the hard drive
    new_location :
        String with the path of the new location of the simulated data
    """
    dc = DataCreator2DGrid(file=None)
    dc.Load_properties(properties_location)
    dc.file = new_location
    dc.Save_properties(properties_location)

def Variate_beta(startbeta, stopbeta, num_steps, iterations, number_of_particles,size_of_system ):
    """
    Simulates a number of systems with different beta values in order to find different properties
    of these systems. The number of no change of states and the longest row of no state change ist outputted
    as plot. Additionaly some simulated states are plotted.
    Parameters
    ----------
    startbeta :
        Starting value for the beta variable
    stopbeta :
        Ending value for the beta variable
    num_steps :
        The number of systems that will be calculated with linearly changing beta
    iterations :
        The number of time steps for simulation one system
    number_of_particles :
        The number of particles that will be simulated for the systems
    size_of_system :
        The size of the system that will be simulated
    """
    prg = tools.progress_log(num_steps)
    delta_beta = (stopbeta-startbeta)/num_steps
    current_beta = startbeta
    data = [[] for k in range(num_steps)]
    for k in range(num_steps):
        datacreator = DataCreator2DGrid('temp.txt',number_of_particles, size_of_system,beta=current_beta)
        datacreator.Simulate_System(iterations)
        data[k] = [datacreator._Get_visual_system(), 
                    datacreator.log[LogTypes.NoStateChange], datacreator.log[LogTypes.RowNoStateChange], current_beta]
        current_beta += delta_beta
        prg.update_progress(k)
    prg.finished("Simulation finished!")
    
    datacreator = DataCreator2DGrid('temp.txt',number_of_particles, size_of_system,beta=current_beta)
    num_of_plots = min(10,num_steps)
    states = [data[(k*num_steps)//num_of_plots][0] for k in range(num_of_plots)]
    titles = [data[(k*num_steps)//num_of_plots][3] for k in range(num_of_plots)]
    datacreator.plot_states(states, 5,titles=titles, discrete=False)

    nostatechange = [data[k][1] for k in range(num_steps)]
    rownostatechange = [data[k][2] for k in range(num_steps)]
    xbeta = [data[k][3] for k in range(num_steps)]
    plt.figure()
    plt.title("Number of no change of state")
    plt.plot(xbeta, nostatechange)
    plt.figure()
    plt.title("Number of longest row of no change of state")
    plt.plot(xbeta, rownostatechange)

    return data

class base_DataCreator(object):
    """Base class for data creation. Introduces general functions and variables that are needed for simulating
    physical systems and plotting them in several ways."""

    def __init__(self, coupling_constant, beta, energy_mode, number_of_particles, size_of_system, file, overwrite_file):
        """
        Parameters
        ----------
        coupling_constant :
            strength of coupling, will be used in child classes for calculating the energy of the system
        beta :
            inverse of temperature for the system. Beta specifies the probability to change the state of 
            the system into a state with higher energy
        energy_mode :
            Determines the way the energy is calculated. For the mode the enum class Energy_Modes is used
        number_of_particles :
            The number of particles that will be simulated
        size_of_system :
            The size of the system that will be simulated
        file :
            Path of type string to which the calculated states will be saved
        overwrite_file :
            Boolean which determines wether the given file will be cleared or not
        """
        self.coupling_constant = coupling_constant
        self.beta = beta
        self.energy_mode = energy_mode
        self.number_of_particles = number_of_particles
        self.size_of_system = size_of_system
        self.file = file
        self.log = {}
        self.last_state = None
        if overwrite_file:
            s = open(file,'w')
            s.close()
        self._seperator = '#'
        self._decay_constant_vis = 0.99

    def get_possibilities(self):
        raise NotImplementedError()
    
    def Is_move_possible(self,particles, movement, particle_idx):
        raise NotImplementedError()
    
    def Calculate_energy(self, particles):
        raise NotImplementedError()

    def Generate_random_state(self):
        raise NotImplementedError()

    def Generate_state(self, stateproperties):
        raise NotImplementedError()

    def Particle_to_visual(self,particles):
        raise NotImplementedError()

    def Get_properties(self):
        """
        Returns a dictionary with properties of the data creator object.
        Properties:
        - file
        - coupling constant
        - beta
        - energy_mode
        - number_of_particles
        - size_of_system
        """
        properties = {}
        properties["file"] = self.file
        properties["coupling_constant"] = self.coupling_constant
        properties["beta"] = self.beta
        properties["energy_mode"] = self.energy_mode
        properties["number_of_particles"] = self.number_of_particles
        properties["size_of_system"] = self.size_of_system
        return properties
    
    def Load_properties(self, properties):
        """
        Loads the given properties dicitonary into the data creator object.
        Parameters
        ----------
        properties :
            dictionary of properties of a data creator object as recieved from
            the Get_properties method. Alternatively a string with the location of
            the properties dictionary on the harddrive saved with the tools module.
        """
        if type(properties) == str:
            properties = tools.load_data(properties)
        self.file = properties["file"]
        self.coupling_constant = properties["coupling_constant"]
        self.beta = properties["beta"]
        self.energy_mode = properties["energy_mode"]
        self.number_of_particles = properties["number_of_particles"]
        self.size_of_system = properties["size_of_system"]
        return properties

    
    def Save_properties(self, properties_file_location):
        """
        Saves properties of the creation object to the hard drive.
        Parameters
        ----------
        properties_file_location :
            The path where the properties shall be saved to.
        """
        tools.save_data(self.Get_properties(), properties_file_location)


    def Get_particles_from_file(self):
        """
        Iterator which gives back all states of the corresponding file in a iterative
        way in the particle format.
        """
        stream = open(self.file, mode='r')
        for line in stream:
            for particles in line.split(self._seperator):
                yield self.str_to_particles(particles)
        stream.close()

    def _Get_visual_system(self):
        """
        Goes over all states of the corresponding file and creates a visualization wich
        can be used for ploting with Plot_state.
        """
        visual_state = self.Particle_to_visual([])
        for particles in self.Get_particles_from_file():
            visual_state = np.minimum(self._decay_constant_vis *visual_state + self.Particle_to_visual(particles), 1.0)
        return visual_state
    
    def Visualize_system(self):
        """
        Visualizes the time evolution of the system with a plot which is a superposition
        of serveral states at different timesteps with an linear decay of intensity with time.
        """
        self.Plot_state(self._Get_visual_system(), discrete=False)
    
    def Animate_system(self, iterations=1000):
        """
        Visualize the time evolution of the system by creating a animation of some states over
        the whole time evolution. For running the animation in jupyter the command %matplotlib qt5
        is needed. To get back images into the notebook use %matplotlib inline 
        Parameters
        ----------
        iterations: 
            The number of states which are used for the animation.
        """
        visual_state = self.Particle_to_visual([])
        states_to_visualize = []
        mod = self.get_number_of_states() // iterations
        for idx, particles in enumerate(self.Get_particles_from_file()):
            visual_state = np.minimum(self._decay_constant_vis *visual_state + self.Particle_to_visual(particles), 1.0)
            if idx % mod == 0:
                states_to_visualize.append(copy.deepcopy(visual_state))
        self.Plot_animation(states_to_visualize)

    def Plot_animation(self, states_to_visualize):
        """
        Method that animates the given states.
        Parameters
        ----------
        states_to_visualize: 
            lists of states in particles format that will be animated
        """
        # First set up the figure, the axis, and the plot element we want to animate
        cmap = plt.get_cmap('Greys')
        fig = plt.figure()
        ax =plt.axes()
        im = plt.imshow(states_to_visualize[0], interpolation='nearest', cmap=cmap)
        # initialization function: plot the background of each frame
        def init():
            im.set_data(states_to_visualize[0])
            return [im]

        # animation function.  This is called sequentially
        def animate(i):
            im.set_data(states_to_visualize[i])
            return [im]

        # call the animator.  blit=True means only re-draw the parts that have changed.
        #self.anim to keep animation running. See: https://github.com/matplotlib/matplotlib/issues/1656
        self.anim = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=len(states_to_visualize), interval=20, blit=True)
        #anim.save('basic_animation.html', fps=30, extra_args=['-vcodec', 'libx264'])
        
        plt.show()

    def Plot_state(self, particles, discrete):
        """
        Plots the given states.
        Parameters
        ----------
        particles :
            The state that will be plotted in the particles format
        discrete :
            Specifies which colormap shall be used for the plot. If True,
            only black and white will be used. For the visualization of a
            time evolution discrete should be set to False. 
        """
        if discrete:
            cmap = mpl.colors.ListedColormap(['white','black'])
        else:
            cmap = plt.get_cmap('Greys')
        if discrete:
            plt.imshow(self.Particle_to_visual(self.str_to_particles(particles)), interpolation='nearest', cmap=cmap)
        else:
            plt.imshow(self.str_to_particles(particles), interpolation='nearest', cmap=cmap)

    def plot_states (self, particles, number_of_coloumns=4, titles = [], discrete = True):
        """
        Plots the given states into several plots, one plot for one state. With 
        number_of_coloumns the number of plots in one line is configured. Optionally
        titles for the individual plots can be given.
        Parameters
        ----------
        particles : 
            A list of states that will be plotted in the particles format
        number_of_coloumns :
            Number of plots in one line
        titles :
            A list of titles for the corresponding states
        discrete :
            Specifies which colormap shall be used for the plot. If True,
            only black and white will be used. For the visualization of a
            time evolution discrete should be set to False. 
        """
        numb_of_lines = math.ceil(len(particles)/number_of_coloumns)
        plt.figure(figsize=(15,numb_of_lines * (15/number_of_coloumns)))
        for k in range(len(particles)):
            plt.subplot(numb_of_lines,number_of_coloumns,k+1)
            if len(titles) > k:
                plt.title(titles[k])
            self.Plot_state(particles[k],discrete=discrete)

    def vis_most_frequent_states(self, number_of_plots=9, start_idx=1):
        """
        Visualizes the most frequent states by plotting them into a grid.
        Parameters
        ----------
        number_of_plots :
            Number of states which will be plotted
        start_idx :
            The starting index in the list of the most frequent states.
            So 1 meaning starting with the most frequent state (0 should
            be skipped because its the number of states which are not represented
            in the dictionary)
        """
        count, dic, rev_dic = tools.Create_dic_from_file(self.file, max(100,number_of_plots), seperator=self._seperator)
        states = [self.str_to_particles(count[k][0]) for k in range(start_idx,start_idx + number_of_plots)]
        titles = [count[k][1] for k in range(start_idx,start_idx + number_of_plots)]
        self.plot_states(states,3,titles)
        return [states,titles]

    def Next_step(self, particles):
        """
        Simulates one time step for the given systems (position of particles).
        For that it uses the methods of markov-chain monte carlo to calculate a possible
        next state and accepting it with a propability that is related to the difference
        in energy between the new and the old state. With the variable beta one can modify 
        the temperature of the system.
        Parameters
        ----------
        particles :
            Current state given in particles format from which the time evolved state shall
            be calculated.
        """
        new_particles = np.copy(particles)
        idx = random.randint(0,len(particles)-1) #choosing the particle to be moved
        possibilities = self.get_possibilities() #np.array([-self.size_of_system, -1,1,self.size_of_system]) #possible movements
        idx_mov = random.randint(0,len(possibilities)-1)
        if self.Is_move_possible(particles, possibilities[idx_mov], idx) == False: #if move is not possible -> return old state of particles
            return particles
        #new_particles[idx] += possibilities[idx_mov]
        new_particles[idx] = self.Apply_movement(new_particles[idx], possibilities[idx_mov])
        [e_old, e_new] = [self.Calculate_energy(particles), self.Calculate_energy(new_particles)]
        prob = min(math.exp(- self.beta * (e_new - e_old)),1)
        new_particles.sort()
        if prob > random.random():
            return new_particles
        return particles

    def Apply_movement(self, particle, movement):
        """
        Applies the given movement onto the particle with given index.
        Parameters
        ----------
        particle :
            The particle in particles format on which the movement shall be applied
        movement :
            The movement direction that shall be applied
        """
        return particle + movement
    
    #TODO: Bug: If the simulation should start with the last simulated state it throws an Value error at
    #elif self.last_state != None and new_system == False:
    #ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
    #TODO: Implement functionality to get multithreading by calculating several systems in parallel and stiching results together
    #https://docs.python.org/2.7/library/multiprocessing.html
    def Simulate_System(self, iterations, start_particles=None, new_system=False):
        """
        Simulates a system via the methods of markov-chain monte carlo over given number 
        of time steps and saves it directly to hard drive using the given file location.
        Parameters
        ----------
        iterations :
            Number of time steps for applying the monte carlo method. A time step of no change
            in the state is counted as one iteration, too.
        start_particles :
            The state from which to start the time evolution.
        new_system :
            Specifies wheater the last simulated state should be used (=False), if possible, 
            or a new random state should be used (=True).
        """
        _states_per_line = 100
        prg = tools.progress_log(iterations)
        _nostatechange = 0
        _rownostatechange = 0
        _maxrow = 0
        if start_particles != None:
            p = np.copy(start_particles)
        elif type(self.last_state) != type(None) and new_system == False:
            p = self.last_state
        else:
            p = self.Generate_random_state()

        stream = open(self.file, mode='a')
        #track = [[] for k in range(iterations+1)]
        #track[0] = p
        for k in range(iterations):
            new_p = self.Next_step(p)
            if (len(new_p) == len(p)) and all(new_p == p): #For ising model len has to be checked
                _nostatechange += 1
                _rownostatechange += 1
            else:
                _maxrow = max(_maxrow,_rownostatechange)
                _rownostatechange = 0
            p = new_p
            #track[k+1] = p
            text = str(list(p)) #With first converting to list an separator "," is added to the string
            if k%_states_per_line == _states_per_line -1:
                text += "\n"
            else:
                text += self._seperator
            stream.write(text)
            prg.update_progress(k)
        
        self.log[LogTypes.NoStateChange] = _nostatechange
        self.log[LogTypes.RowNoStateChange] = _maxrow
        self.last_state = p
        prg.finished()
        stream.close()

    def get_number_of_states(self):
        """
        Returns the number of states currently simulated and saved on the hard drive.
        """
        number = 0
        for particles in self.Get_particles_from_file():
            number += 1
        return number

    def str_to_particles(self, str_particles):
        """
        Returns the state given as string in the particles format.
        Parameters
        ----------
        str_particles :
            The state given in particles form but as string.
        """
        if type(str_particles) == str:
            return json.loads(str_particles)
        else:
            return str_particles

class DataCreator2DGrid(base_DataCreator):
    """
    Child class of the base_DataCreator class. Provides functions to simulate a physical system
    of particles on a 2D quadratic grid. States are normally handled in the 'particles' format
    meaning a particle that is located at the position [x,y] is in particles format at integer 
    position x+y*size_of_system. A state in particles format lists all current particles in that
    position format.
    """
    #Attention: coupling constant was +1 before and is due to changes in energy calculation changed to -1
    def __init__(self,file , number_of_particles=2, size_of_system=7, coupling_constant=-1, beta=0.7, energy_mode=Energy_Modes.neighbouring, overwrite_file=False):
        """
        Parameters
        ----------
        coupling_constant :
            strength of coupling, will be used in child classes for calculating the energy of the system
        beta :
            inverse of temperature for the system. Beta specifies the probability to change the state of 
            the system into a state with higher energy
        energy_mode :
            Determines the way the energy is calculated. For the mode the enum class Energy_Modes is used
        number_of_particles :
            The number of particles that will be simulated
        size_of_system :
            The size of the system that will be simulated
        file :
            Path of type string to which the calculated states will be saved
        overwrite_file :
            Boolean which determines wether the given file will be cleared or not
        """
        super().__init__(coupling_constant=coupling_constant, beta=beta, energy_mode=energy_mode, 
                            number_of_particles=number_of_particles, size_of_system=size_of_system, file=file, overwrite_file=overwrite_file)

    def Generate_random_state(self):
        """
        Generates a random 2D state in the particles format. 
        The particles are set randomly onto the grid.
        """
        #state = np.zeros([_size_of_system,_size_of_system])
        particles = np.array([],dtype= int)
        for k in range(self.number_of_particles):
            valid_idx = False
            while valid_idx == False:
                idx= random.randint(0,self.size_of_system**2-1)
                valid_idx = True
                for p in particles:
                    if p == idx:
                        valid_idx = False
            particles = np.append(particles, idx)
            #state[idx//_size_of_system][idx%_size_of_system] = 1
            particles.sort()
        return particles

    def Generate_state(self, particle_positions, plot = True):
        """
        Generates a state in the particles format.
        Parameters
        ----------
        particles_position :
            A list of positions of particles in the following format [x,y]
            x meaning the x coordinate of the particle and y the y coordinate
        plot :
            If set to true the generated state will be plotted for visualization
        """
        particles = []
        for i,p in enumerate(particle_positions):
            particles.append(p[0]+p[1]*self.size_of_system)
            if i >= self.number_of_particles:
                break
        particles.sort()
        if plot:
            self.Plot_state(particles,discrete=True)
        return particles

    def Get_properties(self):
        """
        Returns a dictionary with properties of the data creator object.
        Properties:
        - file
        - coupling constant
        - beta
        - energy_mode
        - number_of_particles
        - size_of_system
        """
        properties = super().Get_properties()
        return properties
    
    def Load_properties(self, properties):
        """
        Loads the given properties dicitonary into the data creator object.
        Parameters
        ----------
        properties :
            dictionary of properties of a data creator object as recieved from
            the Get_properties method. Alternatively a string with the location of
            the properties dictionary on the harddrive saved with the tools module.
        """
        properties = super().Load_properties(properties)

    def Particle_to_visual(self, particles):
        """
        Converts the given state in the particle format into a format suitable for plotting.
        So a two dimensional array is used for specifining which positions the particles
        occupy.
        Parameters
        ----------
        particles :
            The state in the particles format which will be converted
        """
        state = np.array([[0 for l in range(self.size_of_system)] for k in range(self.size_of_system)])
        for p in particles:
            state[p//self.size_of_system][p%self.size_of_system] = 1
        return state

    def get_possibilities(self):
        """
        Returns a list of all time evolutions one certain particle can do in one time
        step using the particles format.
        """
        return np.array([-self.size_of_system, -1,1,self.size_of_system])

    def Is_move_possible (self, particles, movement, particle_idx):
        """
        Tests if a given movement is valid for the given state. So it tests if the movement
        takes one particle out of the grid or moves it into another particle.
        Parameters
        ----------
        particles :
            The current state on which one time evolution step shall be applied
        movement :
            The chosen movement from the get_possibilities method
        particle_idx :
            The index of the particle that shall be moved
        """
        [xmov,ymov] = [np.sign(movement) * (abs(movement)%self.size_of_system), np.sign(movement) * (abs(movement)//self.size_of_system)]
        #print("x: %d  y: %d" %(xmov,ymov))
        #[xpos,ypos] = [particles[particle_idx]%self.size_of_system, particles[particle_idx]//self.size_of_system]
        [xpos,ypos] = self.getxy_position(particles[particle_idx])
        if xmov+xpos<0 or xmov+xpos>=self.size_of_system or ymov+ypos<0 or ymov+ypos>=self.size_of_system:
            return False
        for p in np.delete(particles, particle_idx):
            #if particles[particle_idx] + movement == p:
            #    return False
            if self.Apply_movement(particles[particle_idx], movement) == p:
                return False
        return True
    
    def getxy_position(self,particle):
        """
        Get xy-position of the given particle
        Parameters
        ----------
        particles : 
            One particle in the particles format given in particles lists
        Returns
        -------
            Array of the form [x,y] with x and y coordinate
        """
        return [particle%self.size_of_system, particle//self.size_of_system]
    
    def getparticle_position(self,xyparticle):
        """
        Get particle in particles format for given particle in xy format
        The xy coordinates will used as modulo size of the system.
        Parameters
        ----------
        xyparticles :
            One particle in the [x,y] format
        Returns :
            One particle in the particles format
        """
        return xyparticle[0]%self.size_of_system + (xyparticle[1]%self.size_of_system)*self.size_of_system

    def Calculate_energy (self, particles):
        """
        Calculates the energy of the given system.
        Parameters 
        ----------
        particles :
            The state in the particles format whose energy shall be computed
        """
        energy = 0
        if self.energy_mode == Energy_Modes.rectangular: #rectangular
            for p in range(len(particles)):
                for k in range(p+1,len(particles)):
                    energy += self.coupling_constant * (1/self.dist_between_grid_points(particles[p], particles[k]))
        elif self.energy_mode == Energy_Modes.neighbouring: #neighbouring
            for p in range(len(particles)):
                for k in range(p+1,len(particles)):
                    if self.is_neighboured(particles[p], particles[k]):
                        energy += self.coupling_constant 
        return energy
    
    def is_neighboured (self, point_1, point_2):
        """
        Gives back a boolean wheater the two given points are neighboured. 
        (diagonal is not neighboured)
        Parameters
        ----------
        point_1 :
            First point in particles format that is looked at
        point_2 :
            Second point in particles format that is looked at
        """
        p1 = np.array([point_1%self.size_of_system, point_1//self.size_of_system])
        p2 = np.array([point_2%self.size_of_system, point_2//self.size_of_system])
        diff = abs(p1 - p2)
        if (diff[0] + diff[1]) == 1:
            return True
        return False

    def dist_between_grid_points (self, point_1, point_2):
        """
        Calculates the number of grid points that are between the two given points.
        For counting only directly connected grid points are used (no diagonals).
        Parameters
        ----------
        point_1 :
            First point in particles format that is looked at
        point_2 :
            Second point in particles format that is looked at
        """
        p1 = np.array([point_1%self.size_of_system, point_1//self.size_of_system])
        p2 = np.array([point_2%self.size_of_system, point_2//self.size_of_system])
        diff = abs(p1 - p2)
        dist = diff[0] + diff[1]
        return dist

class DataCreator2DGridPeriodic (DataCreator2DGrid):
    """
    Child class of the DataCreator2DGrid class. Provides functions to simulate a physical
    system of particles on a 2D quadratic grid with a periodic boundary condition identifing
    the opposite borders with each other. States are normally handled in the 'particles' format
    meaning a particle that is located at the position [x,y] is in particles format at integer 
    position x+y*size_of_system. A state in particles format lists all current particles in that
    position format.
    """
    def Is_move_possible(self, particles, movement, particle_idx):
        """
        Tests if a given movement is valid for the given state. So it tests if the movement
        moves one particle into another particle.
        Parameters
        ----------
        particles :
            The current state on which one time evolution step shall be applied
        movement :
            The chosen movement from the get_possibilities method
        particle_idx :
            The index of the particle that shall be moved
        """
        for p in np.delete(particles, particle_idx):
            if self.Apply_movement(particles[particle_idx], movement) == p:
                return False
        return True

    def Apply_movement(self, particle, movement):
        """
        Applies the given movement onto the particle with given index.
        Parameters
        ----------
        particle :
            The particle in particles format on which the movement shall be applied
        movement :
            The movement direction that shall be applied
        """
        [xmov,ymov] = [np.sign(movement) * (abs(movement)%self.size_of_system),
                       np.sign(movement) * (abs(movement)//self.size_of_system)]
        [xpos,ypos] = self.getxy_position(particle)
        [x,y] = [(xpos+xmov)%self.size_of_system,(ypos+ymov)%self.size_of_system]
        return self.getparticle_position([x,y])

    def is_neighboured(self, point_1, point_2):
        """
        Gives back a boolean wheater the two given points are neighboured. 
        (diagonal is not neighboured, but periodic boundary condition is used.)
        Parameters
        ----------
        point_1 :
            First point in particles format that is looked at
        point_2 :
            Second point in particles format that is looked at
        """
        p1 = np.array([point_1%self.size_of_system, point_1//self.size_of_system])
        p2 = np.array([point_2%self.size_of_system, point_2//self.size_of_system])
        xdist=min(abs(p1[0]-p2[0]),abs(abs(p1[0]-p2[0])-self.size_of_system))
        ydist=min(abs(p1[1]-p2[1]),abs(abs(p1[1]-p2[1])-self.size_of_system))
        if xdist + ydist == 1:
            return True
        return False
    
    def dist_between_grid_points(self,point_1,point_2):
        """
        Calculates the number of grid points that are between the two given points.
        The periodic boundary condition is used.
        Parameters
        ----------
        point_1 :
            First point in particles format that is looked at
        point_2 :
            Second point in particles format that is looked at
        """
        p1 = np.array([point_1%self.size_of_system, point_1//self.size_of_system])
        p2 = np.array([point_2%self.size_of_system, point_2//self.size_of_system])
        xdist=min(abs(p1[0]-p2[0]),abs(abs(p1[0]-p2[0])-self.size_of_system))
        ydist=min(abs(p1[1]-p2[1]),abs(abs(p1[1]-p2[1])-self.size_of_system))
        return xdist + ydist

class DataCreator2DIsingModel (DataCreator2DGridPeriodic):
    """
    Child class of the DataCreator2DGridPeriodic class. Provides functions to simulate a
    physical ising model on a 2D quadratic grid with periodic boundary condition identifing
    the opposite borders with each other. A state the 'particles' format in the ising model 
    consists of all grid points with a positive spin on it. Such a grid point that is located
    at the position [x,y] is in particles format at integer position x+y*size_of_system.
    """
    def __init__(self,file , size_of_system=7, coupling_constant=1, beta=0.7, energy_mode=Energy_Modes.neighbouring, overwrite_file=False):
        """
        Parameters
        ----------
        coupling_constant :
            strength of coupling, will be used in child classes for calculating the energy of the system
        beta :
            inverse of temperature for the system. Beta specifies the probability to change the state of 
            the system into a state with higher energy
        energy_mode :
            Determines the way the energy is calculated. For the mode the enum class Energy_Modes is used
        size_of_system :
            The size of the system that will be simulated
        file :
            Path of type string to which the calculated states will be saved
        overwrite_file :
            Boolean which determines wether the given file will be cleared or not
        """
        super().__init__(coupling_constant=coupling_constant, beta=beta, energy_mode=energy_mode, 
                            number_of_particles=None, size_of_system=size_of_system, file=file, overwrite_file=overwrite_file)
        self._decay_constant_vis = 0

    def Get_properties(self):
        """
        Returns a dictionary with properties of the data creator object.
        Properties:
        - file
        - coupling constant
        - beta
        - energy_mode
        - size_of_system
        """
        properties = {}
        properties["file"] = self.file
        properties["coupling_constant"] = self.coupling_constant
        properties["beta"] = self.beta
        properties["energy_mode"] = self.energy_mode
        properties["size_of_system"] = self.size_of_system
        return properties
    
    def Load_properties(self, properties):
        """
        Loads the given properties dicitonary into the data creator object.
        Parameters
        ----------
        properties :
            dictionary of properties of a data creator object as recieved from
            the Get_properties method. Alternatively a string with the location of
            the properties dictionary on the harddrive saved with the tools module.
        """
        if type(properties) == str:
            properties = tools.load_data(properties)
        self.file = properties["file"]
        self.coupling_constant = properties["coupling_constant"]
        self.beta = properties["beta"]
        self.energy_mode = properties["energy_mode"]
        self.size_of_system = properties["size_of_system"]
        return properties

    def Generate_random_state(self):
        """
        Generates a random 2D state in the particles format. 
        The positive spins are set randomly onto the grid. 
        """
        places = [k for k in range(self.size_of_system**2)]
        random.shuffle(places)
        particles = np.array(places[:random.randint(0,self.size_of_system**2-1)], dtype=int)
        particles.sort()
        return particles
    
    def get_neighbours (self, particle):
        """
        Returns a list with all positions in the particles format which
        are neighboured to the given particle position.
        Parameters
        ----------
        particle :
            integer which specifies the position of one spin on the grid
        """
        [x,y] = self.getxy_position(particle)
        return [self.getparticle_position([x+1,y]),
                self.getparticle_position([x-1,y]),
                self.getparticle_position([x,y+1]),
                self.getparticle_position([x,y-1])]
               
    def Calculate_energy(self, particles):
        """
        Calculates the energy of the given system.
        Parameters 
        ----------
        particles :
            The state in the particles format whose energy shall be computed
        """
        energy = 0
        if self.energy_mode == Energy_Modes.neighbouring:
            for p in [k for k in range(self.size_of_system**2)]:
                for q in self.get_neighbours(p):
                    energy += 2 * (int(p in particles)-0.5) * (int(q in particles)-0.5) * self.coupling_constant                    
        elif self.energy_mode== Energy_Modes.rectangular:
            for p in [k for k in range(self.size_of_system**2)]:
                for q in [k for k in range(p+1,self.size_of_system**2)]:
                    energy += (4 * (int(p in particles)-0.5) * (int(q in particles)-0.5) *
                                self.coupling_constant * (1/self.dist_between_grid_points(p, q)))
        return energy
    def Next_step(self, particles):
        """
        Simulates one time step for the given systems (orientation of spins).
        For that it uses the methods of markov-chain monte carlo to calculate a possible
        next state and accepting it with a propability that is related to the difference
        in energy between the new and the old state. With the variable beta one can modify 
        the temperature of the system.
        Parameters
        ----------
        particles :
            Current state given in particles format from which the time evolved state shall
            be calculated.
        """
        place = random.randint(0,self.size_of_system**2-1)
        if place in particles:
            new_particles = np.setdiff1d(particles, np.array([place]))
        else:
            new_particles = np.append(particles, np.array([place]))
        new_particles.sort()
        [e_old, e_new] = [self.Calculate_energy(particles), self.Calculate_energy(new_particles)]
        prob = min(math.exp(- self.beta * (e_new - e_old)),1)
        if prob > random.random():
            return new_particles
        return particles
