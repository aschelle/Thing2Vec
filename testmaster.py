#This module is for creating test series in which certain parameters will be taken as variable
import wordvectors.analysis as analysis
import wordvectors.physicaldata.tools as tools
from wordvectors.physicaldata import creation
import wordvectors.thing2vec as t2v
import os
from enum import Enum
import matplotlib.pyplot as plt
import copy
import numpy as np
import scipy.optimize as opt

#TODO: Create class or set of functions that can read all files in a given folder for getting the data out 
#in the way the user wants it (datagrids, plots)
class TestResultType(Enum):
    """
    Specifies the type of result that is saved.
    Translation :
        Test results of translational questions.
        Structure: [individual_exact, individual_near, collective_exact, collective_near]
    Rotation :
        Test results of rotational questions
        Structure: [exact, near]
    Translational_quality :
        Test result of testing the translational quality
    Data :
        Paths to created data 
        Structure: [physical system path, neural net]
    Inversion :
        Test results of inversion questions
        Structure: [exact, near]
    """
    Translation = 1
    Rotation = 2
    Translational_quality = 3
    Data = 4
    Inversion = 5

class PropertyCreator(Enum):
    """
    Properties that can be set for the creator of physical systems. 
    All these properties can be used as variables that can be modified
    by the test environment class.
    """
    number_of_particles="number_of_particles"
    size_of_system="size_of_system"
    beta="beta"
    energy_mode="energy_mode"
    iterations="iterations"

class PropertyNet(Enum):
    """
    Properties that can be set for the neural net. 
    All these properties can be used as variables that can be modified
    by the test environment class.
    """
    vocab_size="vocab_size"
    window_size="window_size"
    vector_dim="vector_dim"
    negative_samples="negative_samples"
    min_count="min_count"
    epochs="epochs"
    neural_mode="neural_mode"

class PropertyStatic(Enum):
    """
    Properties of the testenvironment that can not be changed.
    """
    folder="folder"
    tests="tests"
    variables="variables"

class Variable():
    """
    A variable can be used for modifing a parameter of the datacreation or the neural net.
    """
    def __init__(self, property,type, start, stop, number_of_steps):
        """
        Parameters
        ----------
        property :
            Element of one of the PropertyEnums to specifiy the property of the variable
        type :
            The type of the variable e.g. str, int etc.
        start :
            The starting value for the variable
        stop :
            The last value for the variable
        number_of_steps :
            Number of discrete steps starting from start and ending on stop
        """
        self.property=property
        self.type=type
        self.start=start
        self.stop=stop
        self.number_of_steps=number_of_steps
        self.build_value_array()
        self._akt_idx=-1

    def build_value_array(self):
        """
        Called by __init__ method. Generates an array with all values the property will take.
        """
        self.values = [self.type(self.start + (self.stop-self.start)*k/(max(self.number_of_steps-1,1))) for k in range(self.number_of_steps)]
    
    def get_next_value(self):
        """
        Returns the next value that the property will take.
        """
        self._akt_idx += 1
        if self._akt_idx >= self.number_of_steps:
            self._akt_idx=-1
            return None
        return self.values[self._akt_idx]

    def get_progress(self):
        """
        Returns the progress of trained values for this particular variable
        """
        return self._akt_idx / self.number_of_steps

def _getkey(variable:Variable):
    """
    Is used for sorting an array of variables in that way that properties of the
    net will be in front of all other properties to minimize the computing time.
    """
    if isinstance(variable.property,PropertyNet):
        return 0
    return 1

class TestSelection():
    """
    Class that saves the selection of tests that will be done by the Test Environment.
    """
    def __init__(self, translation=True, rotation=True, vis_translation=True, vis_vec_space=True, translational_quality=True, data=True, inverting=True):
        """

        """
        self.translation=translation
        self.rotation=rotation
        self.vis_translation=vis_translation
        self.vis_vec_space = vis_vec_space
        self.translational_quality = translational_quality
        self.data = data
        self.inverting = inverting

class base_data_handler():
    def __init__ (self, folder):
        self.folder=folder
        self.filename = "data_setup_"
        self.imagename = "image_"
        self.dataname = "data_"
        if not os.path.exists(folder):
            os.mkdir(folder)

    def get_number_of_test(self):
        """
        Gets the actual number of test that is performed at the moment in order to get the
        next index that can be used for the next file that will be saved.
        """
        idx = 1
        while os.path.exists(self.folder + "\\" + self.filename + str(idx) + ".pkl"):
            idx += 1
        return idx

    def get_saving_path(self,idx=None,name=None, datatype=".pkl"): #One could save a dictionary file with the idx : properties to get access easily to the propterties of one file
        """
        Returns the saving path that can be used for the next file.
        Parameters
        ----------
        idx :
            optional parameter. If set, the index value will be used to give an index to the
            file name. If not set, the next free index in the folder will be identified and set.
        name :
            optional parameter. Name of the file that will be created. If not set, the standard
            filename will be used.
        datatype :
            type of the file that will be saved.
        """
        if idx == None:
            idx = self.get_number_of_test()
        if name == None:
            name = self.filename
        return  self.folder + "\\" + name + str(idx) + datatype

class base_TestEnvironment(base_data_handler):
    """
    Base class for testing a bunch of physical systems with different parameters.
    """
    def __init__(self, folder, tests=TestSelection(), number_of_particles=None, size_of_system=None, beta=None,
                     energy_mode=None, iterations=None, vocab_size=None, window_size=None, vector_dim=None,
                     negative_samples=None,min_count=None, epochs=None, neural_mode=None, variables =None):
        """
        Parameters
        ----------
        folder :
            Path to folder where all test results and temporal files will be saved to.
        tests :
            Object of type TestSelection that specifies the tests that will be performed by the 
            test environment.
        number_of_particles :
            The number of particles that will be simulated
        size_of_system :
            The size of the system that will be simulated
        beta :
            inverse of temperature for the system. Beta specifies the probability to change the state of 
            the system into a state with higher energy
        iterations :
            Number of time steps for applying the monte carlo method. A time step of no change
            in the state is counted as one iteration, too.
        vocab_size : 
            size of vocabularity which defines the dimensionality of the one hot encoding
        window_size : 
            maximum distance of a current and context word for creating skipgrams
        vector_dim : 
            dimensionality of the embedded layer that will define the word vectors
        negative_samples : 
            how many negative samples are created per positive sample
        min_count : 
            minimal number of occurances for a word to be taken into account for the dictionary
        epochs : 
            number of epochs/iterations over the whole corpus of text
        neural_mode : 
            defines the input for the neural net
        variables :
            list of objects of type Variable that specify the parameters that will be modified
            while testing
        """
        super().__init__(folder)
        properties = {}
        self.tests = tests
        self.variables = variables
        properties[PropertyCreator.number_of_particles] = number_of_particles
        properties[PropertyCreator.size_of_system] = size_of_system
        properties[PropertyCreator.beta] = beta
        properties[PropertyCreator.energy_mode] = energy_mode
        properties[PropertyCreator.iterations] = iterations
        properties[PropertyNet.vocab_size] = vocab_size
        properties[PropertyNet.window_size] = window_size
        properties[PropertyNet.vector_dim] = vector_dim
        properties[PropertyNet.negative_samples] = negative_samples
        properties[PropertyNet.min_count] = min_count
        properties[PropertyNet.epochs] = epochs
        properties[PropertyNet.neural_mode] = neural_mode #TODO: Not needed, because its clear, when using child class?
        self.properties = properties

        self.number_of_questions = 1000
        


    def get_all_properties(self):
        """
        Returns all properties of the TestEnvironment to find corresponding saved data.
        """
        all_props = dict(self.properties)
        all_props[PropertyStatic.folder]=self.folder
        all_props[PropertyStatic.tests]=self.tests
        all_props[PropertyStatic.variables]=self.variables
        return all_props

    def setup_variables(self):
        """
        Sets the first value of all variables that were defined in the __init__ method.
        """
        #sort variables in that way that variables for neural net are before all other
        self.variables.sort(key=_getkey)
        for v in self.variables:
            self.properties[v.property] = v.get_next_value()
        self.variables[0]._akt_idx=-1 # in order to start with the first variable combination into testing

    def start_testing(self):
        """
        Main function of the class. Starts the testing process by creating physical systems and
        corresponding neural nets. Between each iterations one variable is changed to the next value.
        Every combination of variable values is calculated and saved into the given folder in __init__
        function.
        """
        self.setup_variables()
        finished = False
        variables_idx = 0
        datacreator, thing2vec = None, None
        while not finished:
            while True:
                if variables_idx == len(self.variables):
                    finished = True
                    break
                next_value = self.variables[variables_idx].get_next_value()
                if next_value == None:
                    variables_idx+=1
                    continue
                if variables_idx == (len(self.variables)-1):
                    print("Progress %d %%"%(self.variables[variables_idx].get_progress()*100))
                self.properties[self.variables[variables_idx].property] = next_value
                thing2vec=None
                if isinstance(self.variables[variables_idx].property, PropertyCreator):
                    datacreator=None
                variables_idx -=1
                if variables_idx < 0:
                    variables_idx=0
                    break
            if not finished:
                [datacreator, thing2vec] = self.execute_test(datacreator, thing2vec)
        
    #TODO: Find options to get % of progress of test
    def generate_datacreator(self, filename) -> creation.base_DataCreator:
        raise NotImplementedError()
    def generate_thing2vec(self, datacreatorproperties) ->t2v.base_Thing2vec:
        raise NotImplementedError()
    def generate_analyzer(self, datacreator, thing2vec) ->analysis.base_analyzer:
        raise NotImplementedError()
    def save_results(self, data_to_save, idx=None, name=None, datatype=".pkl"):
        """
        Saves the results that were calculated by the class into the folder given in the
        __init__ function.
        Parameters
        ----------
        data_to_save :
            Data that will be saved to a pickle file
        idx :
            optional parameter. If set, the index value will be used to give an index to the
            file name. If not set, the next free index in the folder will be identified and set.
        name :
            optional parameter. Name of the file that will be created. If not set, the standard
            filename will be used.
        datatype :
            type of the file that will be saved.
        """
        tools.save_data(data_to_save,self.get_saving_path(idx,name,datatype),self.get_all_properties())

    
    def execute_test(self, datacreator=None, thing2vec=None):
        """
        Executes one testing cycle with the parameters set before. If needed the function will
        first create a new physical system with the given number of iterations and a corresponding 
        neural net. Afterwards, tests will be performed on the word embedding and saved to files.
        Parameters
        ----------
        datacreator :
            optional parameter. If set, the given datacreator will be used. If not set, a new datacreator
            will be calculated using the parameters set at the moment.
        thing2vec :
            optional parameter. If set, the given thing2vec neural net will be used. If not set, a 
            new neural net will be trained using the parameters set at the moment.
        """
        data_to_save = {}
        idx = self.get_number_of_test()
        new_datacreator = (datacreator == None)
        new_thing2vec = (thing2vec == None) or new_datacreator
        if new_datacreator:
            if self.tests.data:
                self.saving_path = self.get_saving_path(idx = idx, name= "datacreation_data_", datatype=".txt")
            else:
                self.saving_path =  self.get_saving_path(idx = "", name= "datacreation_data_temp", datatype=".txt")
            
            if not os.path.exists(self.saving_path) or not self.tests.data: #TODO: Test if this works
                datacreator = self.generate_datacreator(self.saving_path)
                print("Simulate System")
                datacreator.Simulate_System(self.properties[PropertyCreator.iterations])
            else:
                datacreator = self.generate_datacreator(self.saving_path)
                print("Use simulated system")
        if new_thing2vec: #If one needed a new datacreator also the neural net has to be reseted
            thing2vec = self.generate_thing2vec(datacreator.Get_properties())
            print("Train Network")
            thing2vec.Train(epochs=self.properties[PropertyNet.epochs])
        analyzer = self.generate_analyzer(datacreator, thing2vec)
        print("Perform Tests")
        if self.tests.translation:
            questions_indiv = analyzer.get_translational_questions(self.number_of_questions, translational_mode=analysis.Translational_Mode.Individual)
            questions_collec = analyzer.get_translational_questions(self.number_of_questions, translational_mode=analysis.Translational_Mode.Collective)
            report_indiv_exact = analyzer.test_neural_net(copy.deepcopy(questions_indiv), question_mode=analysis.Question_Mode.Exact)
            report_indiv_near = analyzer.test_neural_net(questions_indiv, question_mode=analysis.Question_Mode.Near)
            report_collec_exact = analyzer.test_neural_net(copy.deepcopy(questions_collec), question_mode=analysis.Question_Mode.Exact)
            report_collec_near = analyzer.test_neural_net(questions_collec, question_mode=analysis.Question_Mode.Near)
            data_to_save[TestResultType.Translation] = [report_indiv_exact, report_indiv_near, report_collec_exact, report_collec_near]
        
        if self.tests.rotation:
            questions = analyzer.get_rotational_questions(number_of_questions=self.number_of_questions)
            report_exact = analyzer.test_neural_net(copy.deepcopy(questions), question_mode=analysis.Question_Mode.Exact)
            report_near = analyzer.test_neural_net(questions, question_mode=analysis.Question_Mode.Near)
            data_to_save[TestResultType.Rotation] = [report_exact, report_near]
        
        if self.tests.translational_quality:
            report = analyzer.test_translational_quality(iterations=self.number_of_questions)
            data_to_save[TestResultType.Translational_quality] = [report]
        
        if self.tests.inverting and self.properties[PropertyNet.neural_mode] == t2v.Neural_Mode.physical2DIsing:
            questions = analyzer.get_inversion_questions(self.number_of_questions)
            report_exact = analyzer.test_neural_net(copy.deepcopy(questions), question_mode=analysis.Question_Mode.Exact)
            report_near = analyzer.test_neural_net(questions, question_mode=analysis.Question_Mode.Near)
            data_to_save[TestResultType.Inversion] = [report_exact, report_near]

        if self.tests.vis_translation:
            figPCA = analyzer.visualize_translational_categories(labeling=True, method=t2v.Dimension_Reduction_Method.PCA, hide_plot=True)
            figtsne = analyzer.visualize_translational_categories(labeling=True, method=t2v.Dimension_Reduction_Method.tsne, hide_plot=True)
            figPCA.savefig(self.get_saving_path(idx=idx,name=(self.imagename+"translation_PCA_"),datatype=".png"))
            figtsne.savefig(self.get_saving_path(idx=idx,name=(self.imagename+"translation_tsne_"),datatype=".png"))

        if self.tests.vis_vec_space:
            vec_s_figPCA = analyzer.thing2vec.visualize_vector_space(method=t2v.Dimension_Reduction_Method.PCA, hide_plot=True)
            vec_s_figtsne = analyzer.thing2vec.visualize_vector_space(method=t2v.Dimension_Reduction_Method.tsne, hide_plot=True)
            vec_s_figPCA.savefig(self.get_saving_path(idx=idx,name=(self.imagename+"vec_space_PCA_"),datatype=".png"))
            vec_s_figtsne.savefig(self.get_saving_path(idx=idx,name=(self.imagename+"vec_space_tsne_"),datatype=".png"))

        if self.tests.data:
            if new_datacreator:
                self.properties_path = self.get_saving_path(idx=idx, name="datacreation_properties_", datatype=".pkl")
                datacreator.Save_properties(self.properties_path)
                
            if new_thing2vec:
                self.thing2vec_path = self.get_saving_path(idx=idx, name="thing2vec_weights_", datatype=".h5")
                thing2vec.save_model(self.thing2vec_path)
            data_to_save[TestResultType.Data] = [self.saving_path, self.properties_path, self.thing2vec_path]
        
        self.save_results(data_to_save,idx=idx)
        return [datacreator, thing2vec]

class TestEnvironment_2DGrid(base_TestEnvironment):
    """
    Child class of _base_TestEnvironment which can be used for testing 2D grid physical systems.
    """
    def generate_datacreator(self, file_path):
        """
        Generates a new datagenerator with the parameters set at the moment.
        """
        datacreator = creation.DataCreator2DGrid(file=file_path, 
                            number_of_particles= self.properties[PropertyCreator.number_of_particles],
                            size_of_system= self.properties[PropertyCreator.size_of_system], 
                            beta= self.properties[PropertyCreator.beta],
                            energy_mode= self.properties[PropertyCreator.energy_mode],
                            overwrite_file=False)
        return datacreator

    def generate_thing2vec(self, datacreatorproperties):
        """
        Generates a new neural net thing2vec with the parameters set at the moment.
        Parameters
        ----------
        datacreatorproperties :
            Properties of the datacreator that are needed for the thing2vec object to connect
            to the physical data and some parameters of it.
        """
        thing2vec = t2v.Thing2VecGensim(neural_mode=self.properties[PropertyNet.neural_mode],
                            vocab_size=self.properties[PropertyNet.vocab_size],
                            window_size=self.properties[PropertyNet.window_size],
                            vector_dim=self.properties[PropertyNet.vector_dim],
                            negative_samples=self.properties[PropertyNet.negative_samples],
                            min_count=self.properties[PropertyNet.min_count],
                            properties=datacreatorproperties)
        return thing2vec
    
    def generate_analyzer(self, datacreator, thing2vec):
        """
        Generates an analyzer for testing the trained neural net.
        Parameters
        ----------
        datacreator :
            datacreator that shall be used for analyzing
        thing2vec :
            thing2vec neural net that shall be used for analyzing
        """
        analyzer = analysis.Analyzer2DGrid(datacreator,thing2vec)
        return analyzer

class TestEnvironment_2DGridPeriodic(TestEnvironment_2DGrid):
    """
    Child class of TestEnvironment_2DGrid which can be used for testing 2D grid physical systems
    with periodic boundary conditions.
    """
    def generate_datacreator(self, file_path):
        """
        Generates a new datagenerator with the parameters set at the moment.
        """
        datacreator = creation.DataCreator2DGridPeriodic(file=file_path, 
                            number_of_particles= self.properties[PropertyCreator.number_of_particles],
                            size_of_system= self.properties[PropertyCreator.size_of_system], 
                            beta= self.properties[PropertyCreator.beta],
                            energy_mode= self.properties[PropertyCreator.energy_mode],
                            overwrite_file=False)
        return datacreator
    
    def generate_analyzer(self, datacreator, thing2vec):
        """
        Generates an analyzer for testing the trained neural net.
        Parameters
        ----------
        datacreator :
            datacreator that shall be used for analyzing
        thing2vec :
            thing2vec neural net that shall be used for analyzing
        """
        analyzer = analysis.Analyzer2DGridPeriodic(datacreator,thing2vec)
        return analyzer

class TestEnvironment_2DIsingModel (TestEnvironment_2DGrid):

    def __init__(self, folder, tests=TestSelection(), size_of_system=None, beta=None,
                     energy_mode=None, iterations=None, vocab_size=None, window_size=None, vector_dim=None,
                     negative_samples=None,min_count=None, epochs=None, neural_mode=None, variables =None):
        """
        Parameters
        ----------
        folder :
            Path to folder where all test results and temporal files will be saved to.
        tests :
            Object of type TestSelection that specifies the tests that will be performed by the 
            test environment.
        size_of_system :
            The size of the system that will be simulated
        beta :
            inverse of temperature for the system. Beta specifies the probability to change the state of 
            the system into a state with higher energy
        iterations :
            Number of time steps for applying the monte carlo method. A time step of no change
            in the state is counted as one iteration, too.
        vocab_size : 
            size of vocabularity which defines the dimensionality of the one hot encoding
        window_size : 
            maximum distance of a current and context word for creating skipgrams
        vector_dim : 
            dimensionality of the embedded layer that will define the word vectors
        negative_samples : 
            how many negative samples are created per positive sample
        min_count : 
            minimal number of occurances for a word to be taken into account for the dictionary
        epochs : 
            number of epochs/iterations over the whole corpus of text
        neural_mode : 
            defines the input for the neural net
        variables :
            list of objects of type Variable that specify the parameters that will be modified
            while testing
        """
        super().__init__(folder, tests=tests, number_of_particles=None, size_of_system=size_of_system, beta=beta,
                     energy_mode=energy_mode, iterations=iterations, vocab_size=vocab_size, window_size=window_size, vector_dim=vector_dim,
                     negative_samples=negative_samples,min_count=min_count, epochs=epochs, neural_mode=neural_mode, variables =variables)
        self.properties.pop(PropertyCreator.number_of_particles)

    def generate_datacreator(self, file_path):
        """
        Generates a new datagenerator with the parameters set at the moment.
        """
        datacreator = creation.DataCreator2DIsingModel(file=file_path, 
                            size_of_system= self.properties[PropertyCreator.size_of_system], 
                            beta= self.properties[PropertyCreator.beta],
                            energy_mode= self.properties[PropertyCreator.energy_mode],
                            overwrite_file=False)
        return datacreator

    def generate_thing2vec(self, datacreatorproperties):
        """
        Generates a new neural net thing2vec with the parameters set at the moment.
        Parameters
        ----------
        datacreatorproperties :
            Properties of the datacreator that are needed for the thing2vec object to connect
            to the physical data and some parameters of it.
        """
        thing2vec = t2v.Thing2VecGensim(neural_mode=self.properties[PropertyNet.neural_mode],
                            vocab_size=self.properties[PropertyNet.vocab_size],
                            window_size=self.properties[PropertyNet.window_size],
                            vector_dim=self.properties[PropertyNet.vector_dim],
                            negative_samples=self.properties[PropertyNet.negative_samples],
                            min_count=self.properties[PropertyNet.min_count],
                            properties=datacreatorproperties)
        return thing2vec

    def generate_analyzer(self, datacreator, thing2vec):
        """
        Generates an analyzer for testing the trained neural net.
        Parameters
        ----------
        datacreator :
            datacreator that shall be used for analyzing
        thing2vec :
            thing2vec neural net that shall be used for analyzing
        """
        analyzer = analysis.Analyzer2DIsingModel(datacreator,thing2vec)
        return analyzer

class PresentingHandler(base_data_handler):
    """
    Base class for loading and presenting the calculated results in test environment
    """
    def __init__(self, folder, reduced_loading=False):
        """
        Parameters
        ----------
        folder :
            path to folder where the data was saved by the test environment
        reduced_loading :
            specifies if all data will be loaded or the results are calculated 
            and the questions are deleted
        """
        super().__init__(folder)
        self.file_infos = self.load_files(reduced_loading)

    #TODO: Add option to only load accuracy results to minimize RAM-comsumption
    def load_files(self, reduced_loading=False):
        """
        Loads data from all files in the folder
        Parameters
        ----------
        reduced_loading :
            specifies if all data will be loaded or the results are calculated 
            and the questions are deleted
        """
        file_infos = []
        self.reduced_loading = reduced_loading
        for k in range(1,self.get_number_of_test()):
            data = tools.load_data(self.get_saving_path(idx=k))
            if reduced_loading:
                taccuracies,raccuracies, iaccuracies =[], [], []
                if TestResultType.Translation in data:
                    for result in data[TestResultType.Translation]:
                        taccuracies.append(result.get_accuracy())
                    data[TestResultType.Translation] = taccuracies
                if TestResultType.Rotation in data:
                    for result in data[TestResultType.Rotation]:
                        raccuracies.append(result.get_accuracy())
                    data[TestResultType.Rotation] = raccuracies
                if TestResultType.Inversion in data:
                    for result in data[TestResultType.Inversion]:
                        iaccuracies.append(result.get_accuracy())
                    data[TestResultType.Inversion] = iaccuracies
            infos = tools.log[tools.LogTypes.LoadedSettings]
            file_infos.append([data,infos])
        return file_infos
    
    def sort_data_for_plot(self, xaxisproperty):
        """
        Sorts the data into lists of results with constant parameters apart from
        the parameter that shall be on the x-axis of the plot.
        Parameters
        ----------
        xaxisproperty :
            Property that will be on the x-axis variated. Needs to be a Property Enum
            of the testmaster module.
        """
        tests = []
        for f in self.file_infos:
            found = False
            for [idx,t] in enumerate(tests):
                f_copy, t_copy = f[1].copy(), t[0][1].copy()
                f_copy.pop(xaxisproperty), t_copy.pop(xaxisproperty)
                for prop_st in PropertyStatic: #Static properties are not compared
                    if prop_st in f_copy:
                        f_copy.pop(prop_st), t_copy.pop(prop_st)
                if f_copy == t_copy:
                    tests[idx].append(f)
                    found=True
                    break
            if not found:
                tests.append([f])
        return tests
    

    def _find_constant_variables(self, test, xaxisproperty):
        """
        Finds the variables that stay constant in the test array created by
        sort_data_for_plot and gives back a string with the current values of 
        the fixed variables.
        Parameters
        ----------
        test :
            An element of the output of sort_data_for_plot that is a list of results
            with constant parameters apart from one parameter.
        xaxisproperty :
            The parameter that is not kept constant and will be plotted on the x-axis
        """
        const_var = []
        text = ""
        if PropertyStatic.variables in test[0][1]:
            for v in test[0][1][PropertyStatic.variables]:
                if v.property != xaxisproperty:
                    const_var.append(v.property)
            for p in const_var:
                text += p.name + " = " + str(test[0][1][p])+"\n"
            text = text[:-1] #remove last \n
        return text

    def plot_test(self, test, xaxisproperty, testresulttype, question_mode=analysis.Question_Mode.Exact, 
                    translational_mode=analysis.Translational_Mode.Individual, polynomial_fit=-1, 
                    exponential_fit=False, arbitrary_fit=None, arbitrary_p0=None):
        """
        Plots results of the given test on the y-axis and the property given on the x-axis 
        with the given test data.
        Parameters
        ----------
        test :
            An element of the return of sort_data_for_plot method giving data with constant
            parameters apart from the parameters on x-axis.
        xaxisproperty :
            Property that will be on the x-axis variated. Needs to be a Property Enum
            of the testmaster module.
        testresulttype :
            Test that shall be used for the accuracy on the y-axis. Must be an element of the
            TestResult Enum in the testmaster module.
        question_mode :
            If implemented for testtype, specifies the question mode
        translational_mode :
            When using the Translation test, specifies the translational mode 
        polynomial_fit :
            Optional polynomial fit. If < 0 no fit is applied. If >=0 a fit is applied and plotted
            with polynomial_fig being the degree of the fit polynomial.
        exponential_fit :
            Optional exponential fit. If True, an exponential fit is applied and plotted. 
        arbitrary_fit :
            Optional fit. Set arbitrary to the fit function, which will be used for fitting and will
            be plotted.
        arbitrary_p0 :
            Optional starting parameter for the arbitrary_fit function. 
        """
        x = [t[1][xaxisproperty] for t in test]
        idx = 0
        if testresulttype in [TestResultType.Translation, TestResultType.Rotation, TestResultType.Inversion]:
            if question_mode == analysis.Question_Mode.Near:
                idx += 1
            if testresulttype == TestResultType.Translation:
                if translational_mode == analysis.Translational_Mode.Collective:
                    idx += 2
            if not self.reduced_loading:
                y = [t[0][testresulttype][idx].get_accuracy() for t in test]
            else :
                y = [t[0][testresulttype][idx] for t in test]
        elif testresulttype == TestResultType.Translational_quality:
            y = [t[0][testresulttype][0][0] for t in test]
        label = self._find_constant_variables(test,xaxisproperty)
        if polynomial_fit > -1:
            f = np.polynomial.polynomial.Polynomial.fit(x,y,polynomial_fit)
            x_fit = np.linspace(x[0], x[-1], 100)
            y_fit = f(x_fit)    
        if exponential_fit:
            optimizedParameters, pcov = opt.curve_fit(tools.exp_func, x, y)
            x_fit_2 = np.linspace(x[0], x[-1], 100)
            y_fit_2 = tools.exp_func(x_fit_2, *optimizedParameters)
        if arbitrary_fit != None:
            optimizedParameters, pcov = opt.curve_fit(arbitrary_fit, x, y, p0=arbitrary_p0)
            x_fit_3 = np.linspace(x[0], x[-1], 100)
            y_fit_3 = arbitrary_fit(x_fit_3, *optimizedParameters)
        fig = plt.figure()
        plt.plot(x,y,'.',label=label)
        if polynomial_fit > -1:
            plt.plot(x_fit,y_fit,'-')
        if exponential_fit:
            plt.plot(x_fit_2,y_fit_2,'-')
        if arbitrary_fit:
            plt.plot(x_fit_3,y_fit_3,'-')
        plt.xlabel(xaxisproperty.name)
        plt.ylabel(testresulttype.name + " accuracy")
        if label != "":
            plt.legend()
        return [x,y,fig]

    def load_datacreator_net(self, file_infos_idx):
        """
        Loads, if saved from the test environment, the datacreator and the thing2vec 
        object.
        Parameters
        ----------
        file_infos_idx :
            The index of the data in the file_infos object whose datacreator and
            thing2vec object shall be loaded.
        """
        if TestResultType.Data in self.file_infos[file_infos_idx][0]:
            data_file, properties_file, weights = self.file_infos[file_infos_idx][0][TestResultType.Data]
            datacreator = self.create_datacreator(data_file)
            datacreator.Load_properties(properties_file)
            thing2vec = self.create_net(properties_file, self.file_infos[file_infos_idx][1])
            thing2vec.load_model(weights)
            return [datacreator,thing2vec]
        return None

    def create_datacreator(self,file)->creation.base_DataCreator:
        raise NotImplementedError()
    
    def create_net(self,properties_creator, properties_testEnvironment)->t2v.base_Thing2vec:
        raise NotImplementedError()

class PresentingHandler_2DGrid(PresentingHandler):
    """
    Child class of PresentingHandler that implements function needed for specifics
    of 2D-Grids.
    """
    def create_datacreator(self,file):
        """
        Creates a datacreator that is connected to the datafile given.
        Parameters
        ----------
        file :
            path to the file with the simulated states.
        """
        return creation.DataCreator2DGrid(file=file,overwrite_file=False)
    def create_net(self,properties_creator, properties_testEnvironment):
        """
        Creates a thing2vec neural net object using the given properties for the 
        neural net and the datacreator.
        Parameters
        ----------
        properties_creator :
            Properties of the data creator
        properties_testEnvironment :
            Properties of the test Environment to get the properties for the 
            neural net that is created.
        """
        return t2v.Thing2VecGensim(neural_mode=t2v.Neural_Mode.physical2D,
                                    vocab_size=properties_testEnvironment[PropertyNet.vocab_size],
                                    window_size=properties_testEnvironment[PropertyNet.window_size],
                                    vector_dim=properties_testEnvironment[PropertyNet.vector_dim],
                                    negative_samples=properties_testEnvironment[PropertyNet.negative_samples],
                                    min_count=properties_testEnvironment[PropertyNet.min_count],
                                    properties=properties_creator)


class PresentingHandler_2DGridPeriodic(PresentingHandler_2DGrid):
    """
    Child class of PresentingHandler_2DGrid that implements function needed for specifics
    of 2D-Grids with periodic boundary conditions.
    """
    def create_datacreator(self,file):
        """
        Creates a datacreator that is connected to the datafile given.
        Parameters
        ----------
        file :
            path to the file with the simulated states.
        """
        return creation.DataCreator2DGridPeriodic(file=file,overwrite_file=False)
    def create_net(self,properties_creator, properties_testEnvironment):
        """
        Creates a thing2vec neural net object using the given properties for the 
        neural net and the datacreator.
        Parameters
        ----------
        properties_creator :
            Properties of the data creator
        properties_testEnvironment :
            Properties of the test Environment to get the properties for the 
            neural net that is created.
        """
        return t2v.Thing2VecGensim(neural_mode=t2v.Neural_Mode.physical2Dperiodic,
                                    vocab_size=properties_testEnvironment[PropertyNet.vocab_size],
                                    window_size=properties_testEnvironment[PropertyNet.window_size],
                                    vector_dim=properties_testEnvironment[PropertyNet.vector_dim],
                                    negative_samples=properties_testEnvironment[PropertyNet.negative_samples],
                                    min_count=properties_testEnvironment[PropertyNet.min_count],
                                    properties=properties_creator)
class PresentingHandler_2DGridIsingModel(PresentingHandler):
    def create_datacreator(self,file):
        """
        Creates a datacreator that is connected to the datafile given.
        Parameters
        ----------
        file :
            path to the file with the simulated states.
        """
        return creation.DataCreator2DIsingModel(file=file,overwrite_file=False)
    def create_net(self,properties_creator, properties_testEnvironment):
        """
        Creates a thing2vec neural net object using the given properties for the 
        neural net and the datacreator.
        Parameters
        ----------
        properties_creator :
            Properties of the data creator
        properties_testEnvironment :
            Properties of the test Environment to get the properties for the 
            neural net that is created.
        """
        return t2v.Thing2VecGensim(neural_mode=t2v.Neural_Mode.physical2DIsing,
                                    vocab_size=properties_testEnvironment[PropertyNet.vocab_size],
                                    window_size=properties_testEnvironment[PropertyNet.window_size],
                                    vector_dim=properties_testEnvironment[PropertyNet.vector_dim],
                                    negative_samples=properties_testEnvironment[PropertyNet.negative_samples],
                                    min_count=properties_testEnvironment[PropertyNet.min_count],
                                    properties=properties_creator)
