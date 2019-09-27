import wordvectors.physicaldata.tools as tools
import wordvectors.physicaldata.creation as creation
import wordvectors.thing2vec as t2v

from enum import Enum
import numpy as np
import collections
import random

class Question_Mode (Enum):
    """
    Specifies how the answer given by the neural net is interpreted.
    Exact : 
        The question is answered correctly only if the first answer of the neural net
        is the correct one.
    Near :
        The question is answered correctly if the first n answers of the neural net contain
        the correct one. n is specified in the analyzer class by _number_of_near.
    """
    Exact = 1
    Near = 2

class Translational_Mode (Enum):
    """
    Specifies how the translational questions are generated.
    Collective : 
        For all translational operations all particles of the system are moved. So from the start
        state all particles are moved randomly for one positive state and one negative state. Then 
        the translations are added in the way positive_movement - negative_movement. The resulting 
        movement is added to the start state for the correct answer state.
    Individual :
        Some translational operations only move single particles that were randomly chosen. First all
        particles are moved from a start state (original state is negative and moved is positive). 
        Then a single particle is moved from the moved state and from the original 
        (for both states the same)(first one is the correct answer, second is a positive state).
    """
    Collective = 1
    Individual = 2

class Question ():
    """
    Class that bundles a question that is asked to the neural net. Objects of that class
    are used by the ReportClass and its child classes.
    """
    def __init__(self, answer, positive, negative=[], vectors = False):
        """
        Attention
        ---------
        The string of the parameters has to be a member of the dictionary in the thing2vec neural net!
        Answer has to be a string parameter always. positive and negative members can be vectors in the
        word vector space, too.
        Parameters
        ----------
        answer :
            object that will be the correct answer of that question.
        positive :
            list of objects that will be positively (+) contribute to the question that is asked.
        negative :
            list of objects that will be negatively (-) contribute to the question that is asked.
        vectors :
            determines if the positive and negative objects are already vectors (True) or words 
            that have to be translated into vectors (False) 
        """
        self.answer = answer
        self.positive = positive
        self.negative = negative
        self.vectors = vectors
    def set_answer(self, correct, accuracy=None, infos=None):
        """
        Adds information about the answer that was given by the neural net
        Parameters
        ----------
        correct :
            Set to True if question was answered correctly, False if not
        accuracy :
            Ranking of the right answer in the answer given by the neural net
        infos :
            Further infos about the answer that was given by the neural net
        """
        self.correct = correct
        self.accuracy = accuracy
        self.infos = infos

class ReportClass():
    """
    Class that bundles all information of a test that was performed on a neural net.
    """
    def __init__(self, questions):
        """
        Parameters
        ----------
        questions :
            list of Question objects that were asked to the neural net.
        """
        self.questions = questions

    def get_correct(self):
        """
        Returns a list of all questions that were answered correctly.
        """
        correct = []
        for q in self.questions:
            if q.correct:
                correct.append(q)
        return correct

    def get_incorrect(self):
        """
        Returns a list of all questions that were answered wrongly.
        """
        incorrect = []
        for q in self.questions:
            if not q.correct:
                incorrect.append(q)
        return incorrect        

    def get_accuracy(self):
        """
        Returns the accuracy of the neural net on that particular test.
        """
        accuracy = 0
        for q in self.questions:
            if q.correct:
                accuracy += 1
        accuracy /= len(self.questions)
        return accuracy

class base_analyzer ():
    """
    Base class for analyzing a Thing2Vec neural net. Introduces general functions an variables to generate
    and ask questions to the neural net.
    """
    #When using Question_Mode.Near this variable specifies the maximal distance (number of wrong candidates
    #before the right one) of the right answer and the answer the neural net has given.
    _number_of_near = 10
    #Number of attempts when generating questions to find a allowed state before skipping that particular base state
    _max_it = 1000
    def __init__(self, creator: creation.base_DataCreator, thing2vec: t2v.base_Thing2vec):
        """
        Parameters
        ----------
        creator : base_DataCreator
            Object of class base_DataCreator that was used for generating the data that was used by
            the neural net.
        thing2vec : base_Thing2vec
            The trained neural net which used the data created by the creator object.
        """
        self.creator = creator
        self.thing2vec = thing2vec

    
    def get_translational_questions(self):
        raise NotImplementedError()

    def get_rotational_questions(self):
        raise NotImplementedError()

    def get_inversion_questions(self, number_of_questions=100):
        raise NotImplementedError()
    #Idea of enivronmental question: Take a state and its environment (nearest states), then translate the original state
    #and its environment by a vector and look at the nearest states of the translated original state. Now compare the shifted
    #environment of the original state and the nearest states of the shifted original state
    def get_environmental_questions(self):
        raise NotImplementedError()

    def shift_state(self, particles, particle_idx, movement):
        raise NotImplementedError()

    def visualize_translational_categories(self, labeling=False,number_of_categories=5):
        raise NotImplementedError()

    def get_translational_error(self):
        raise NotImplementedError()

    def test_translational_quality(self, iterations=100, question_mode = Question_Mode.Near):
        raise NotImplementedError()

    def ask_question(self, question, question_mode = Question_Mode.Exact, plot = False):
        """
        Asks question to the neural net and returns wheater the question was anwered correctly or not.
        Parameters
        ----------
        question :
            Question of type Question that shall be asked.
        question_mode :
            How the answer of the question by the neural net shall be interpreted.
        plot :
            If =True the answer by the neural net will be visualized and plotted
        Returns
        -------
        Array of following form: 
        [Question was answered right: bool, ranking of the right answer: int, result of neural net: list]
        """
        plotsettings=t2v.PlotMode.noPlotting
        if plot and not question.vectors:
            self.visualize_question(question)
            plotsettings = t2v.PlotMode.answerOnly
        
        if not question.vectors:
            result = self.thing2vec.most_similar(positive= [str(q) for q in question.positive], negative=[str(q) for q in question.negative],number = self._number_of_near, plot=plotsettings)
        else:
            if question.negative != []:
                result = self.thing2vec.similar_by_vector(vector= np.sum(question.positive,axis=0) - np.sum(question.negative,axis=1), number=self._number_of_near, plot=plot)
            else:
                result = self.thing2vec.similar_by_vector(vector= np.sum(question.positive,axis=0), number=self._number_of_near, plot=plot)

        if question_mode == Question_Mode.Exact:
            if result[0][0] == str(question.answer):
                return [True,0,result]
            return [False,-1,result]
        elif question_mode == Question_Mode.Near:
            for i,r in enumerate(result):
                if r[0] == str(question.answer):
                    return [True,i,result]
            return [False,-1,result]
        print("No suitable question mode selected!")
        return [False,-1,result]

    def test_neural_net (self, questions, question_mode = Question_Mode.Exact):
        """
        Tests the neural net with the given bunch of questions.
        Parameters
        ----------
        questions :
            List of questions that will be asked to the neural net.
        question_mode :
            How the answer of the question by the neural net shall be interpreted.
        """
        report = ReportClass(questions)
        for q in questions:
            answer = self.ask_question(q, question_mode=question_mode)
            q.set_answer(answer[0],answer[1],answer[2])
        return report

    def visualize_question (self, question):
        """
        Visualizes the question by plotting all contributing states.
        Parameters
        ----------
        question :
            Question that shall be plotted.
        """
        titles = ["positive"] * len(question.positive) + ["negative"] * len(question.negative) + ["answer"]
        self.creator.plot_states(question.positive + question.negative + [question.answer], titles=titles)

class Analyzer2DGrid (base_analyzer):
    """
    Child class of base_analyzer that is used for analyzing neural nets that were trained with elements
    of the DataCreator2DGrid
    """

    def __init__(self, creator: creation.DataCreator2DGrid, thing2vec: t2v.base_Thing2vec):
        """
        Parameters
        ----------
        creator : DataCreator2DGrid
            Object of class DataCreator2DGrid that was used for generating the data that was used by
            the neural net.
        thing2vec : base_Thing2vec
            The trained neural net which used the data created by the creator object.
        """
        super().__init__(creator=creator, thing2vec=thing2vec)
    def _get_number_of_particles_in_state(self, particles):
        return self.creator.number_of_particles
    def get_translational_questions(self, number_of_questions=100, translational_mode = Translational_Mode.Individual):
        """
        Get a randomly generated list of questions that have translational variances, meaning the particles 
        have been only translational moved. 
        Parameters
        ----------
        number_of_questions :
            The maximal number of questions that will be generated
        translational_mode :
            Settings how the translational questions are created
        """
        #TODO: Here one is limited by the number of states in the dictionary. One could implement a non-uniform random number generator
        #which takes more frequently more frequently occuring states.
        starting_states, translated_states, movement = [], [], []
        changed_states, changed_translated_states, movement_change, particle_change = [],[],[],[]
        #TODO: Only maximal number_of_questions will be generated. Change for to while loop to be sure 
        #to generate number_of_questions questions
        for n in range(number_of_questions):
            starting = self.creator.str_to_particles(self.thing2vec.reverse_dictionary[n+1])
            if translational_mode == Translational_Mode.Individual:
                p_c = [random.randint(0,self._get_number_of_particles_in_state(starting)-1)]
            if translational_mode == Translational_Mode.Collective:
                p_c = [k for k in range(self._get_number_of_particles_in_state(starting))]
            it = 0
            found = False
            while True:
                it +=1
                if it > self._max_it:
                    break #if maximal number of iterations is reached, break while loop
                [start, stop] = self._get_movement_range(starting)
                mov = random.randint(start, stop)
                shifted = self.shift_state(starting, [k for k in range(self._get_number_of_particles_in_state(starting))],mov)
                shifted_not_sorted = self.shift_state(starting, [k for k in range(self._get_number_of_particles_in_state(starting))],mov, sorted_p=False)
                if shifted != starting:
                    found = True
                    break
            if not found:
                continue #if while loop ended without finding a shifted state, this combination will be skipped
            it = 0
            while True:
                it +=1
                if it > self._max_it:
                    break #if maximal number of iterations is reached, break while loop
                [start, stop] = self._get_movement_range(starting[p_c[0]:(p_c[-1]+1)])
                mov_c = random.randint(start,stop)
                #mov_c = random.randint(-starting[p_c[0]], self.creator.size_of_system**2 - starting[len(p_c)-1])
                changed_starting = self.shift_state(starting,p_c,mov_c)
                changed_translated = self.shift_state(shifted_not_sorted,p_c,mov_c) #The not sorted particles are used so that the same particle is moved as before
                #when suitable states for the question were found, add them to the list.
                if changed_starting != starting and changed_translated != shifted:
                    if ((self.thing2vec.is_in_dictionary(str(shifted))) and
                            (self.thing2vec.is_in_dictionary(str(starting))) and
                            (self.thing2vec.is_in_dictionary(str(changed_starting))) and 
                            (self.thing2vec.is_in_dictionary(str(changed_translated)))):
                        translated_states.append(shifted)
                        starting_states.append(starting)
                        movement.append(mov)
                        changed_states.append(changed_starting)
                        changed_translated_states.append(changed_translated)
                        movement_change.append(mov_c)
                        particle_change.append(p_c)
                    break

        #Now sum up results for questions
        questions = [Question(changed_translated_states[k], [changed_states[k], translated_states[k]],[starting_states[k]]) for k in range(len(starting_states))]
        return questions

    def _get_movement_range(self, particles):
        """
        Get maximal possible movement range of the given particles.
        Parameters
        ----------
        particles :
            bunch of particles in particles format that shall be moved
        Returns
        -------
        [minmal movement, maximal movement]
        """
        return [-particles[0], self.creator.size_of_system**2 - particles[-1]]
    def get_rotational_questions(self, number_of_questions=100):
        """
        Get randomly generated list of questions that have rotational variances. 
        Two semirandom start configurations are rotated with the same angle. The rotation
        of of two states is taken as "difference" (by substracting) and is added to the other start
        state.
        Parameters
        ----------
        number_of_questions :
            The maximal number of questions that will be generated
        """
        first_states = [self.creator.str_to_particles(self.thing2vec.reverse_dictionary[n+1]) for n in range(number_of_questions)]
        second_states = [self.creator.str_to_particles(self.thing2vec.reverse_dictionary[random.randint(1,number_of_questions)]) for n in range(number_of_questions)]
        rotation_steps = [random.randint(1,3) for n in range(number_of_questions)]
        rotated_first_states = [self._rotate_particles(first_states[k],rotation_steps=rotation_steps[k]) for k in range(number_of_questions)]
        rotated_second_states = [self._rotate_particles(second_states[k], rotation_steps=rotation_steps[k]) for k in range(number_of_questions)]
        questions = [Question(rotated_second_states[k],[rotated_first_states[k], second_states[k]], [first_states[k]]) for k in range(number_of_questions) if
                        self.thing2vec.is_in_dictionary(str(rotated_second_states[k])) and
                        self.thing2vec.is_in_dictionary(str(rotated_first_states[k])) and
                        self.thing2vec.is_in_dictionary(str(second_states[k])) and
                        self.thing2vec.is_in_dictionary(str(first_states[k]))]
        return questions
    
    def _rotate_particles(self, particles,direction=+1, rotation_steps =1):
        """
        Rotates the given particles state clockwise in 90° steps around the center
        of the system.
        Parameters
        ----------
        particles :
            state in particles format that will be rotated
        direction :
            direction of the rotation. +1 is clockwise, -1 is anticlockwise
        rotation_steps :
            number of 90° rotations in the given direction
        """
        xypositions = [self.creator.getxy_position(p) for p in particles]
        middle = (self.creator.size_of_system-1) / 2
        for t in range(rotation_steps):
            for k in range(len(xypositions)):
                xypositions[k] = [int((-1)*direction*(xypositions[k][1]-middle)+middle), int(direction*(xypositions[k][0]-middle)+middle)]
        particles = [self.creator.getparticle_position(p) for p in xypositions]
        particles.sort()
        return particles
        

        

    #TODO: Implement function to show context of the given categories too. Meaning, plotting the most common states, too
    def visualize_translational_categories(self, labeling=False,number_of_categories=7, method=t2v.Dimension_Reduction_Method.tsne, hide_plot=False, starting=None, movement=None):
        """
        Visualizes random states that were translated in x or y direction to build a translation
        line with different colours using the tsne method.
        Parameters
        ----------
        labeling :
            if set to True all states will be labeled by their state name.
        number_of_categories :
            number of translation lines/categories that will be generated and visualized.
        method :
            Method that is used for dimension reduction
        hide_plot :
            if True the plot will not be shown (can be useful if the plot shall be saved directly)
        starting_state :
            starting state of creating the translational categories. If set to None the starting state
            will be chosen randomly.
        movement :
            array of the two movements that will be used for creating all starts for the categories (index 0)
            and the direction of movement of one category (index 1). If set to None the starting state
            will be chosen randomly.
        """
        categories = []
        if starting == None:
            starting = self.creator.str_to_particles(self.thing2vec.reverse_dictionary[random.randint(1,100)])
        if movement == None:
            movement = sorted(self.creator.get_possibilities(), reverse=True)[:2]
            random.shuffle(movement)
        starts_for_categories = self._generate_translational_category(starting,movement[0])
        #random.shuffle(starts_for_categories)
        for k in range(min(number_of_categories,len(starts_for_categories))):
            categories.append(self._generate_translational_category(starts_for_categories[k],movement[1]))
        categories = [[str(c) for c in ies] for ies in categories]
        return self.thing2vec.visualize_categories(categories, [str(c[0]) for c in categories], labeling=labeling, method=method, hide_plot=hide_plot)

    def _generate_translational_category(self,start,movement):
        """
        Generates starting from the start state a translational line/category by shifting the start
        state in movement and -movement direction.
        Parameters
        ----------
        start :
            starting state that will be shifted
        movement :
            direction of the shift of the start state
        """
        category = [start]
        while True:
            new = self.shift_state(category[-1], [k for k in range(len(start))], movement)
            if new == category [-1]:
                break
            category.append(new)
        while True:
            new = self.shift_state(category[0], [k for k in range(len(start))], -movement)
            if new == category [0]:
                break
            category.insert(0,new)
        return category

    

    def shift_state(self, particles, particle_idx, movement, sorted_p=True):
        """
        Shifts all specified particles in the given state by the given movement direction.
        Parameters
        ----------
        particles :
            The state that shall be shifted in the particles format
        particle_idx :
            A list of indexes of the particles in the state that shall be moved.
        movement :
            Movement that shall be applied to all particles that were specified in
            particle_idx.
        sorted_p :
            If True, the particles are sorted due their indices (needed for the dictionary)
        """
        shifted_particles = particles[:] #copy particles
        for idx in particle_idx:
            if self.creator.Is_move_possible([shifted_particles[idx]], movement, 0): #With these parameters only borders will be checked
                #shifted_particles[idx] = shifted_particles[idx] + movement
                shifted_particles[idx] = self.creator.Apply_movement(shifted_particles[idx], movement)
            else:
                #print("Not possible to move state!")
                return particles
        counter = collections.Counter(shifted_particles)
        if counter.most_common(n=1)[0][1] > 1:
            #print("Not possible to move state!")
            return particles
        if sorted_p:
            shifted_particles.sort()
        return shifted_particles


    def test_translational_quality(self, iterations=100, question_mode = Question_Mode.Near):
        """
        Tests the quality of a translational motion. For that a random state is translated in x or
        y direction to get a "line of states". Now the normalized distance between the first and the 
        last state is taken to get a difference vector. Parts of that vector are now added to the first state
        vector. The results are compared to the real word vector of the translated state given in 
        the "line of states". Additionaly a test is performed if the true word vector is comparable near
        to the calculated vectors in contrast to all other vectors.
        Parameters
        ----------
        iterations :
            How much iterations of the above procedure are performed to calculate a mean error
        question_mode :
            question mode that will be used for the test if the true word vector is near the calculated vector
        Returns
        -------
        [errors, accuracies] :
            errors :
                distance between the real word vector and the calculated vector divided by the length of the
                added vector.
            accuracies :
                ratio of correct answered tests
        """
        errors, accuracies = [], []
        key_errors = 0
        for k in range(iterations):
            try:
                result = self._get_translational_error(question_mode=question_mode)
                errors.append(result[0])
                accuracies.append(result[1])
            except KeyError: #for the case that some states do not exist in dictionary
                key_errors += 1
        errors, accuracies = np.mean(errors), np.mean(accuracies)
        return [errors, accuracies]
        
    def _get_translational_error(self, question_mode = Question_Mode.Near):
        """
        Is used by the test_translational_quality method to perform the translational test as described
        in the documentation of that function.
        Parameters
        ----------
        question_mode :
            question mode that will be used for the test if the true word vector is near the calculated vector
        """
        translation_line = []
        #Find random translation line that has a length of minimum 3 to be able to perform the test 
        while len(translation_line) < 3:
            starting = self.creator.str_to_particles(self.thing2vec.reverse_dictionary[random.randint(1,100)])
            movement = sorted(self.creator.get_possibilities(), reverse=True)[:2]
            random.shuffle(movement)
            translation_line = self._generate_translational_category(starting,movement[0])
        #Use normalized vectors because for most_similar methods normalized vectors are used too.
        start_vec, stop_vec = self.thing2vec.get_word_vector(str(translation_line[0]), normalized=True), self.thing2vec.get_word_vector(str(translation_line[-1]), normalized=True)
        diff_vec = stop_vec-start_vec
        error = 0
        questions = []

        for k in range(1,len(translation_line)-1):
            #Calculate vector starting from the start word vector and normalize it
            vector = start_vec + diff_vec * (k/(len(translation_line)-1))
            vector /= np.linalg.norm(vector)
            questions.append(Question(answer=translation_line[k], positive=[vector],vectors=True))
            dist_btw_pnts = np.linalg.norm(self.thing2vec.get_word_vector(str(translation_line[0]), normalized=True) - 
                            self.thing2vec.get_word_vector(str(translation_line[k]), normalized=True))
            translational_error = np.linalg.norm(vector - self.thing2vec.get_word_vector(str(translation_line[k]), normalized=True))
            error += translational_error/dist_btw_pnts
            #print("dist: %f error: %f"%(dist_btw_pnts, translational_error))
            
        report = self.test_neural_net(questions,question_mode=question_mode)
        error /= len(translation_line)-2
        return [error, report.get_accuracy()]

class Analyzer2DGridPeriodic (Analyzer2DGrid):
    """
    Child class of Analyzer2DGrid that is used for analyzing neural nets that were trained with elements
    of the DataCreator2DGridPeriodic
    """
    def __init__(self, creator: creation.DataCreator2DGridPeriodic, thing2vec: t2v.base_Thing2vec):
        """
        Parameters
        ----------
        creator : DataCreator2DGrid
            Object of class DataCreator2DGrid that was used for generating the data that was used by
            the neural net.
        thing2vec : base_Thing2vec
            The trained neural net which used the data created by the creator object.
        """
        super().__init__(creator=creator, thing2vec=thing2vec)

    def _get_movement_range(self,particles):
        """
        Get maximal possible movement range of the given particles.
        Parameters
        ----------
        particles :
            bunch of particles in particles format that shall be moved
        Returns
        -------
        [minmal movement, maximal movement]
        """
        return [0, self.creator.size_of_system**2]

    def _generate_translational_category(self,start,movement):
        """
        Generates starting from the start state a translational line/category by shifting the start
        state in movement direction until it reaches start again.
        Parameters
        ----------
        start :
            starting state that will be shifted
        movement :
            direction of the shift of the start state
        """
        category = [start]
        while True:
            new = self.shift_state(category[-1], [k for k in range(len(start))], movement)
            if new == category [0]:
                break
            category.append(new)
        return category
    
class Analyzer2DIsingModel(Analyzer2DGridPeriodic):
    def __init__(self, creator: creation.DataCreator2DIsingModel, thing2vec: t2v.base_Thing2vec):
        """
        Parameters
        ----------
        creator : DataCreator2DIsingModel
            Object of class DataCreator2DIsingModel that was used for generating the data that 
            was used by the neural net.
        thing2vec : base_Thing2vec
            The trained neural net which used the data created by the creator object.
        """
        super().__init__(creator=creator, thing2vec=thing2vec)

    def _get_number_of_particles_in_state(self, particles):
        return len(particles)

    def _invert_particles(self, particles):
        all_places = [k for k in range(self.creator.size_of_system**2)]
        for p in particles:
            all_places.remove(p)
        return all_places

    #TODO: Add option to only invert a certain area of the grid
    def get_inversion_questions(self, number_of_questions=100):
        idx_1 = [k for k in range(1,number_of_questions*2)]
        idx_2 = idx_1[:]
        random.shuffle(idx_1)
        random.shuffle(idx_2)
        first_states = [self.creator.str_to_particles(self.thing2vec.reverse_dictionary[idx_1[n]]) 
                        for n in range(number_of_questions)]
        second_states = [self.creator.str_to_particles(self.thing2vec.reverse_dictionary[idx_2[n]]) 
                        for n in range(number_of_questions)]
        first_states_inv = [self._invert_particles(s) for s in first_states]
        second_states_inv = [self._invert_particles(s) for s in second_states]
        questions = [Question(second_states_inv[k],[second_states[k],first_states_inv[k]],[first_states[k]])
                        for k in range(number_of_questions) if
                        self.thing2vec.is_in_dictionary(str(first_states[k])) and
                        self.thing2vec.is_in_dictionary(str(second_states[k])) and
                        self.thing2vec.is_in_dictionary(str(first_states_inv[k])) and
                        self.thing2vec.is_in_dictionary(str(second_states_inv[k]))]
        return questions
    
