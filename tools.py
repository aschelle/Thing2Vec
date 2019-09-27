"""
This modules gives some tools for handling the created states of datacreation.py and other small tools
"""
import numpy as np
import sys
import time
import os
from math import factorial
import pickle
import zipfile
import tensorflow as tf
from enum import Enum

class LogTypes(Enum):
    LoadedSettings = 1
    
log = {}

def tic():
    """
    starts a timer that can be stoped with the function toc
    """
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    """
    ends a timer that was started with the function tic and prints the
    elapsed time in seconds.
    """
    elapsed = time.time() - startTime_for_tictoc
    print("Elapsed time: %4.1f s" % elapsed)

def state_to_statearray(state):
    """
    Transforms the given 2d state array into an 1d array of the same state
    """
    statearray = np.array([])
    for s in state:
        statearray = np.append(statearray, s)
    return statearray

def statearray_to_state(array):
    """
    Transforms the given 1d state array into a 2d array of the same state
    """
    length = int(round(np.sqrt(len(array))))
    state = np.array([[array[k*length + l] for l in range(length)] for k in range(length)])
    return state

def statearray_to_particle(array):
    """
    Transforms the given 1d state array into the state representation with 
    the positions of all particles given in an array
    """
    particles = np.array([],dtype = int)
    for k in range(len(array)):
        if array[k] == 1:
            particles = np.append(particles, k)
    return particles

def particle_to_statearray(particles, size_of_system):
    """
    Transforms the given state - represented as positions of all particles - 
    into an 1d state array
    """
    array = np.zeros(size_of_system**2)
    for p in particles:
        array[p] = 1
    return array

def particle_to_state(particles, size_of_system):
    """
    Transforms the given state - represented as positions of all particles - 
    into a 2d state array
    """
    state = np.array([[0 for l in range(size_of_system)] for k in range(size_of_system)])
    for p in particles:
        state[p//size_of_system][p%size_of_system] = 1
    return state

def binomialCoefficient(n, k):
    """
    Calculates the binomial coefficient k out of n
    """
    return factorial(n) // (factorial(k) * factorial(n - k))

def dist_between_grid_points (point_1, point_2, size_of_system):
    """
    Calculates the number of grid points that are between the two given points.
    For counting only directly connected grid points are used (no diagonals).
    """
    p1 = np.array([point_1%size_of_system, point_1//size_of_system])
    p2 = np.array([point_2%size_of_system, point_2//size_of_system])
    diff = abs(p1 - p2)
    dist = diff[0] + diff[1]
    return dist

def is_neighboured (point_1, point_2, size_of_system):
    """
    Gives back a boolean wheater the two given points are neighboured. 
    (diagonal is not neighboured)
    """
    p1 = np.array([point_1%size_of_system, point_1//size_of_system])
    p2 = np.array([point_2%size_of_system, point_2//size_of_system])
    diff = abs(p1 - p2)
    if (diff[0] + diff[1]) == 1:
        return True
    return False

#TODO: Beim Speichern werden neben den Daten auch weitere Infos vom System abgespeichert, damit die Daten immer zugeordnet werden können
def save_data(data, filename, properties = {}):
    """
    Saves the given data to the given path using the pickle package
    """
    #https://www.thoughtco.com/using-pickle-to-save-objects-2813661
    filehandler = open(filename, 'wb')
    if type(properties) == dict:
        pickle.dump([data,properties], filehandler)
    else:
        pickle.dump(data, filehandler)
    filehandler.close()
    
def load_data(filename):
    """
    Loads the data at the given path that was saved with the pickle
    package via the save_data function
    """
    filehandler = open(filename, 'rb')
    filedata = pickle.load(filehandler)
    if len(filedata) == 2 and type(filedata[1]) == dict:
        data = filedata[0]
        log[LogTypes.LoadedSettings] = filedata[1]
    else:
        data = filedata
        log[LogTypes.LoadedSettings] = {}
    filehandler.close()
    return data

def read_data_from_zip(filename):
    """
    Extract the first file enclosed in a zip file as a list of words.
    Used for text procession for neuralnet.
    """
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

class saver_class():
    """
    This class is for handling files located on different places in one ore several
    computers. The class gets the current homedirectory at initialisation.
    """
    def __init__(self, homedir):
        """
        Parameters
        ----------
        homedir:
            root folder of all data that will be handled. This given string will be
            added beforehead using the filename of the data to be saved.
        """
        self.homedir = homedir
    def save_data(self, data, filename, properties = {}):
        """
        Saves data to the given homedir + filename directory.
        Parameters
        ----------
        data:
            Data that will be saved
        filename:
            Path of the location to which the file shall be saved. The filename
            is relative to the homedir location.
        properties:
            Optional dictionary that will be saved in addition to the data.
            When loading this saved data the properties will be loaded to the log
            of the module.
        """
        save_data(data,os.path.join(self.homedir,filename), properties)
    def load_data(self, filename):
        """
        Loads data from the given homedir + filename directory.
        Parameters
        ----------
        filename:
            Path of the location from which the file shall be loaded. The filename
            is relative to the homedir location.
        """
        load_data(os.path.join(self.homedir,filename))
    def get_full_path (self, filename):
        """
        Returns the absolute path of the given filename by appending the homedir
        Parameters
        ----------
        filename :
            Path of the file that shall be given as an absolute path.
        """
        return os.path.join(self.homedir,filename)

def sizeof_fmt(num, suffix='B'):
    """ 
    By Fred Cirera, after https://stackoverflow.com/a/1094933/1870254
    """
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def listsizeof(localitems, number_of_items = 10):
    """
    Prints the size of the largest variables. Localitems is the list
    of loaded variables given e.g. by locals().items()
    """
    #for name, size in sorted(((name, sys.getsizeof(value)) for name,value in locals().items()),key= lambda x: -x[1])[:10]:
    for name, size in sorted(((name, sys.getsizeof(value)) for name,value in localitems),key= lambda x: -x[1])[:number_of_items]:
        print("{:>30}: {:>8}".format(name,sizeof_fmt(size)))

def get_unitvector(n,k):
    """
    Returns a unitvector of size n with an entry equal 1 at index k
    """
    temp = np.zeros(n)
    temp[k] = 1
    return temp

def Create_dic_from_file(file, vocab_size, seperator = ' '):
    """
    Takes the given file and creates a dictionary, containing the vocab_size
    first words which were seperated with the given seperator sign.
    """
    stream = open(file, 'r')
    count = {}
    for line in stream:
        for element in line.replace("\n","").split(seperator):
            if element in count:
                count[element] += 1
            else:
                count[element] = 1
    count = sorted(count.items(), key=lambda kv: kv[1],reverse=True)
    unk_count=0
    for c in count[vocab_size:]:
        unk_count += c[1]
    count = [('UNK', unk_count)] + count
    count = count[:vocab_size]
    dictionary = dict()
    for element, c in count:
        dictionary[element] = len(dictionary)
    count[0] = list(count[0])
    count[0][1] = unk_count
    count[0] = tuple(count[0])
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return count, dictionary, reversed_dictionary

def takeSecond(element):
    """
    Returns the second entry of the given list.
    """
    return element[1]

def exp_func(x,a,b,c):
    """
    Returns the value of the exponential function at position x with 
    the given parameters. 
    f(x)=-a * np.exp(-b * x) + c
    """
    return -a * np.exp(-b * x) + c

def rect_func(x,a,b,c):
    """
    Returns the value of the rectangular function at position x with 
    the given parameters. 
    -a * np.abs(x-b) + c
    """
    return -a * np.abs(x-b) + c

def mostSimilarToLatex(origin,most_similar):
    """
    Returns the result of the most_similar words to a given origin word
    into a latex table.
    """
    finaltext = "\hline \multicolumn{2}{|c|}{%s} \\\ \n\hline\n"%(origin)
    for [t,s] in most_similar:
        finaltext += "%s & %.2f \\\ \n\hline\n"%(t,s)
    return finaltext

def _categories_legend(number_of_lines):
    return ["$C_{\mathrm{trans}}(S_%i)$"%k for k in range(1,number_of_lines+1)]

def fig_to_pdf(figure, path, hide_legend=False, hide_axis=False,xaxis=None,yaxis=None, sav_type="pdf", own_legend=None):
    """
    Saves the given figure to a pdf file at the given path.
    Before eventually some operations are done.
    """
    if hide_legend:
        figure.axes[0].legend().remove()
    if own_legend != None:
        #figure.axes[0].legend().remove()
        figure.axes[0].legend(own_legend)
    if hide_axis:
        figure.axes[0].get_xaxis().set_visible(False)
        figure.axes[0].get_yaxis().set_visible(False)
    if xaxis!=None:
        figure.axes[0].xaxis.set_label_text(xaxis)
    if yaxis!=None:
        figure.axes[0].yaxis.set_label_text(yaxis)
    figure.savefig(path,format=sav_type, bbox_inches='tight')

def _make_particle_visible_svg(text,particles,plidx):
    """
    Takes svg file and makes particle visible at specified file
    location.
    """
    for particle in particles:
        lidx = text.find("label=\"%s\""%str(particle+1),plidx)
        text = text[:lidx]+text[lidx:].replace("display:none","display:inline",1)
    return text

def _set_spins_svg(text,particles,plidx):
    """
    Sets the spins according to the given particles at the
    specified file location.
    """
    for particle in particles:
        upidx = text.find("label=\"up\"",plidx)
        downidx = text.find("label=\"down\"",plidx)
        lupidx = text.find("label=\"%s\""%str(particle+1),upidx)
        lupidx = _find_beginning_svg(lupidx, text)
        ldownidx = text.find("label=\"%s\""%str(particle+1),downidx)
        ldownidx = _find_beginning_svg(ldownidx, text)
        text = text[:lupidx]+text[lupidx:].replace("display:none","display:inline",1)
        text = text[:ldownidx]+text[ldownidx:].replace("display:inline","display:none",1)
    return text
def _find_beginning_svg(idx, text, beginning="<"):
    """
    Gives back the file location of the next beginning string 
    starting with the given index.
    """
    akt_idx = idx
    while text[akt_idx] != beginning:
        akt_idx -= 1
    return akt_idx


def most_similar_to_svg(templatepath, resultpath, str_to_particles, most_similar=[], positive=[], negative=[], answer=[], set_function=_make_particle_visible_svg, positive_label=[], negative_label=[], answer_label=[]):
    """
    Takes a most_similar result of the neural net module and exports it to an svg,
     using the given template.
    """
    stream = open(templatepath, 'r')
    text = stream.read()
    stream.close()
    answer = [answer]
    for idx,item in enumerate(most_similar):
        plidx = text.find("label=\"place_%s\""%str(idx+1))
        text = text[:plidx]+text[plidx:].replace(">label<",">%.2f<"%item[1],1)
        text = set_function(text,str_to_particles(item[0]),plidx)
    for idx,particles in enumerate(positive):
        plidx = text.find("label=\"positive_%s\""%str(idx+1))
        if len(positive_label) > idx:
            text = text[:plidx]+text[plidx:].replace(">label<",">%s<"%positive_label[idx],1)
        text = set_function(text,str_to_particles(particles),plidx)
    for idx,particles in enumerate(negative):
        plidx = text.find("label=\"negative_%s\""%str(idx+1))
        if len(negative_label) > idx:
            text = text[:plidx]+text[plidx:].replace(">label<",">%s<"%negative_label[idx],1)
        text = set_function(text,str_to_particles(particles),plidx)
    for idx,particles in enumerate(answer):
        plidx = text.find("label=\"answer_%s\""%str(idx+1))
        if len(answer_label) > idx:
            text = text[:plidx]+text[plidx:].replace(">label<",">%s<"%answer_label[idx],1)
        text = set_function(text,str_to_particles(particles),plidx)
    wr_stream = open(resultpath,'w')
    wr_stream.write(text)
    wr_stream.close()

def most_occuring_to_svg(templatepath, resultpath, str_to_particles, states,titles, set_function=_make_particle_visible_svg):
    """
    Takes a most_occuring result of the neural net module and exports it to an svg,
    using the given template.
    """
    stream = open(templatepath, 'r')
    text = stream.read()
    stream.close()
    for idx,particles in enumerate(states):
        plidx = text.find("label=\"place_%s\""%str(idx+1))
        text = text[:plidx]+text[plidx:].replace(">label<",">%s<"%str(titles[idx]),1)
        text = set_function(text,str_to_particles(particles),plidx)
    wr_stream = open(resultpath,'w')
    wr_stream.write(text)
    wr_stream.close()

class progress_log:
    old_s = 0
    goal = 0
    def __init__(self, goal):
        self.goal = goal
    def update_progress (self,progress):
        status = (progress*100)/self.goal
        if self.old_s != status:
            sys.stdout.write("\r%4.1f%%" %status)
            self.old_s = status
    def finished(self,message = "Finished!"):
        sys.stdout.write("\r")
        print(message)