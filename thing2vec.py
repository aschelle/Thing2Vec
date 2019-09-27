from keras.models import Model
from keras.layers import Input, Reshape, merge, dot, Activation
from keras.layers.embeddings import Embedding
import keras.initializers
from keras.utils import Sequence
from keras.preprocessing import sequence

import wordvectors.physicaldata.tools as tools
import wordvectors.physicaldata.creation as creation
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from enum import Enum
import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random

import os


class Neural_Mode(Enum):
    physical2D = 1
    text = 2
    physical2Dperiodic = 3
    physical2DIsing = 4

class Dimension_Reduction_Method(Enum):
    tsne = 1
    PCA = 2

class PlotMode(Enum):
    noPlotting =1
    answerOnly =2
    complete=3

#TODO: Introduce option to load data into RAM to get faster calculations if the file is not to big
"""
class Skipgram_Generator(Sequence):
    
    def __init__(self, file, batch_size):
        stream = open(file,mode='r')
        self.batch_size = batch_size
    
    def __getitem__(self, idx):
"""

class base_Thing2vec():
    """
    Base class for thing2vec child classes. Introduces general functions that will be used or be
    overwritten by the child classes.
    """
    #TODO: Introduce standard place for saving models and loading data
    def __init__(self, vocab_size, window_size, vector_dim, negative_samples, neural_mode, file=None, properties=None):
        """
        Parameters
        ----------
        vocab_size : 
            size of vocabularity which defines the dimensionality of the one hot encoding
        window_size : 
            maximum distance of a current and context word for creating skipgrams
        vector_dim : 
            dimensionality of the embedded layer that will define the word vectors
        negative_samples : 
            how many negative samples are created per positive sample
        neural_mode : 
            defines the input for the neural net
        file : 
            string which points to the file with the corpus of text
            This is only needed if neural_mode is set to text
        properties: 
            Properties for the data creator that will be loaded. 
            This is only needed if neural_mode is set to physical2D or subclass
        """
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.vector_dim = vector_dim
        self.negative_samples = negative_samples
        self.neural_mode = neural_mode
        self.file = file
        if (self.neural_mode == Neural_Mode.physical2D or 
                self.neural_mode == Neural_Mode.physical2Dperiodic or 
                self.neural_mode == Neural_Mode.physical2DIsing):
            if self.neural_mode == Neural_Mode.physical2D:
                self.datacreator = creation.DataCreator2DGrid(file=None)
            elif self.neural_mode == Neural_Mode.physical2Dperiodic:
                self.datacreator = creation.DataCreator2DGridPeriodic(file=None)
            elif self.neural_mode == Neural_Mode.physical2DIsing:
                self.datacreator = creation.DataCreator2DIsingModel(file=None)
            self.datacreator.Load_properties(properties)
            if self.file == None:
                self.file = self.datacreator.file
            self.seperator = self.datacreator._seperator
        elif self.neural_mode == Neural_Mode.text:
            self.seperator = ' '
        if self.file != None:
            count, dictionary, reverse_dictionary = tools.Create_dic_from_file(self.file, vocab_size, seperator=self.seperator)
            self.count = count
            self.dictionary = dictionary
            self.reverse_dictionary = reverse_dictionary
        self.vocab_size = min(self.vocab_size, len(self.count)) #if there are less words than given vocab size
        

    #TODO: For optimization a second iterable would be usefull that does not need a dictionary and just
    # uses the tokens from the file without conversion (more infos jupyter "setting up Thing2Vec")
    class Thing2String(object):
        """
        Class that reads the given file via an iterator and translates the read words via the
        dictionary for the underlying neuralnet.
        """
        def __init__(self, file, dictionary, sep=" ", transformation=str):
            """
            Parameters
            ----------
            file : 
                string which points to the file with the corpus of text
            dictionary : 
                dictionary which is used for conversion of the words in the file text
            sep:
                token which is used to seperate the lines in single words
            transformation:
                type of the yielded output
            """
            self.file = file
            self.dictionary = dictionary
            self.sep = sep
            self.transformation = transformation
        def __iter__ (self):
            stream = open(self.file, 'r')
            for line in stream:
                sentence = []
                for element in line.split(sep=self.sep):
                    if element in self.dictionary:
                        sentence.append(self.transformation(self.dictionary[element]))
                    else:
                        sentence.append(self.transformation("0")) #The UNK word
                yield sentence
            stream.close()

        def len (self):
            stream = open(self.file, 'r')
            k = 0
            for s in stream:
                k += 1
            stream.close()
            return k

    def get_word_vector (self, word, normalized = True):
        raise NotImplementedError()
    
    def get_word_vectors(self, normalized = False):
        raise NotImplementedError()

    def Train(self, file, epochs, batch_size, initial_epoch):
        raise NotImplementedError()

    def index2word(self, idx):
        raise NotImplementedError()

    def save_model(self, path):
        raise NotImplementedError()

    def load_model(self,path):
        raise NotImplementedError()

    def similarity(self, w1, w2):
        raise NotImplementedError()

    def most_similar(self, positive, negative, number):
        raise NotImplementedError()

    def most_similar_cosmul(self, positive, negative, number):
        raise NotImplementedError()
    
    def visualize_most_similar(self, positive = [], negative=[], max_number=200, perplexity=40, n_iter=2500):
        """
        Visualizes the vector space using the tsne method. The space is restricted around the
        resulting vector by the given positive and negative words.
        Parameters
        ----------
        positive : 
            list of positive words which will be given to the most_similar method
        negative : 
            list of negative words which will be given to the most_similar method
        max_number : 
            maximal number of visualized words
        perplexity : 
            parameter for the tsne method
        n_iter : 
            parameter for the tsne method
        """
        most_similar = self.most_similar(positive=positive, negative=negative, number=max_number-1)
        custom_words = list(map(lambda x: x[0], most_similar))
        custom_points = []
        if len(positive) == 1 and len(negative) == 0:
            custom_points.append([self.get_word_vector(positive[0],normalized=False),positive[0].upper()])
        return self.visualize_vector_space(custom_words=custom_words, custom_points=custom_points, max_number=max_number, perplexity=40, n_iter=2500)

    def visualize_most_similar_cosmul(self, positive = [], negative=[], max_number=200, perplexity=40, n_iter=2500):
        """
        Visualizes the vector space using the tsne method. The space is restricted around the
        resulting vector by the given positive and negative words.
        Parameters
        ----------
        positive : 
            list of positive words which will be given to the most_similar_cosmul method
        negative : 
            list of negative words which will be given to the most_similar_cosmul method
        max_number : 
            maximal number of visualized words
        perplexity : 
            parameter for the tsne method
        n_iter : 
            parameter for the tsne method
        """
        most_similar = self.most_similar_cosmul(positive=positive, negative=negative, number=max_number)
        custom_words = list(map(lambda x: x[0], most_similar))
        custom_points = []
        if len(positive) == 1 and len(negative) == 0:
            custom_points.append([self.get_word_vector(positive[0],normalized=False),positive[0].upper()])
        return self.visualize_vector_space(custom_words=custom_words, custom_points=custom_points, max_number=max_number, perplexity=40, n_iter=2500)

    
     #TODO: For better results one should introduce an PCA before applying tsne
    def visualize_vector_space(self, custom_words = [], custom_points = [], method=Dimension_Reduction_Method.tsne, max_number = 200, perplexity=40, n_iter=2500, hide_plot=False):
        """
        Visualizes the vector space using the tsne method. 
        Parameters
        ----------
        custom_words : 
            optional list of words which are taken into account for visualization
        custom_points : 
            optional list of lists of the form [point, label] which will be added in the
            visualization
        method :
            Specifies which method form dimension reduction is used
        max_number : 
            maximal number of visualized words (including custom_words)
        hide_plot :
            if True the plot will not be shown (can be useful if the plot shall be saved directly)
        """
        [new_values,labels] = self._apply_dimension_reduction(custom_words, custom_points,method = method,number_to_train=max_number, perplexity=perplexity, n_iter=n_iter)
        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])
            
        fig = plt.figure(figsize=(16, 16)) 
        for i in range(len(x)):
            plt.scatter(x[i],y[i], color='blue')
            plt.annotate(labels[i],xy=(x[i], y[i]),xytext=(5, 2),textcoords='offset points',ha='right',va='bottom')
        if not hide_plot:
            plt.show()
        else:
            plt.close()
        return fig
    
    def _apply_dimension_reduction(self,trainables, custom_points= [], number_to_train = 0,number_of_learning=500,method=Dimension_Reduction_Method.tsne, perplexity=40, n_iter=2500):
        """
        Applies a dimension reduction, using the chosen method, onto the given data.
        Parameters
        ----------
        trainables :
            list of words (or general things) which are contained in the vocabulary and
            whose embedding vectors shall be lowered in dimension
        custom_points :
            list of custom points which shall be lowered in dimension
        number_to_train :
            number of words (or general things) which will be outputted after dimension 
            reduction
        number_of_learning :
            number of points which will minimal be used for dimension reduction. A higher 
            number improves the results when using tsne
        method :
            method that will be used for dimension reduction
        perplexity :
            parameter for tsne (see documentation)
        n_iter :
            parameter for tsne (see documentation)
        """
        labels = []
        tokens = []
        _skipped = 0
        for word in trainables:
            try:
                tokens.append(self.get_word_vector(word,normalized=False))
                labels.append(word)
            except KeyError:
                _skipped += 1
        
        for point,label in custom_points:
            tokens.append(point)
            labels.append(label)
        vectors = self.get_word_vectors(normalized = False)
        vectors_labels = [[vectors[k], self.index2word(k)] for k in range(len(vectors)) if self.index2word(k) not in trainables]
        vectors_labels = vectors_labels[:max(0,number_of_learning-len(trainables)-len(custom_points))]
        for v in vectors_labels:
            tokens.append(v[0])
            labels.append(v[1])
        if method == Dimension_Reduction_Method.tsne:
            tsne_model = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=n_iter, random_state=23)
            new_values = tsne_model.fit_transform(tokens)
        else :
            pca_model = PCA(n_components=2)
            new_values = pca_model.fit(tokens).transform(tokens)
        
        #final = [new_values[k] for k in range(len(labels)) if labels[k] is in trainables]
        return [new_values[:max(number_to_train, len(trainables)+ len(custom_points))], labels[:max(number_to_train, len(trainables)+len(custom_points))]]  

    def visualize_categories(self, categories, label_categories, method = Dimension_Reduction_Method.tsne, perplexity=40, n_iter=2500, labeling=False, hide_plot=False):
        """
        Visualizes the given categories in a two-dimensional space by reducing the dimension
        of the embedding space.
        Parameters
        ----------
        categories :
            A list of categories which shall be visualized
        label_categories :
            List of strings which label the given categories
        method :
            method that will be used for dimension reduction
        perplexity :
            parameter for tsne (see documentation)
        n_iter :
            parameter for tsne (see documentation)
        labeling :
            if False the labels of the categories will be omitted
        hide_plot :
            if True the plot of the visualization will not be shown
        """
        #tokens = []
        labels = []
        for category in categories:
            for element in category:
                #tokens.append(self.get_word_vector(element,normalized=False))
                labels.append(element)
        #tsne_model = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=n_iter, random_state=23)
        #new_values = tsne_model.fit_transform(tokens)
        [new_values,labels] = self._apply_dimension_reduction(labels,method=method, perplexity=perplexity, n_iter=n_iter)
        x_cat,y_cat = [],[]
        akt_idx = 0
        for k in range(len(categories)):
            new_idx = akt_idx + len(categories[k])
            x_cat.append([new_values[k][0] for k in range(akt_idx,new_idx)])
            y_cat.append([new_values[k][1] for k in range(akt_idx,new_idx)])
            akt_idx = new_idx
        fig = plt.figure(figsize=(16,16))
        for i in range(len(categories)):
            plt.scatter(x_cat[i],y_cat[i])
            plt.plot(x_cat[i],y_cat[i])
        plt.legend(label_categories)
        if labeling:
            for k in range(len(new_values)):
                plt.annotate(labels[k],xy=(new_values[k][0],new_values[k][1]),xytext=(5, 2),textcoords='offset points',ha='right',va='bottom')
        if not hide_plot:
            plt.show()
        else:
            plt.close()
        return fig
            
class Thing2VecGensim(base_Thing2vec):
    """
    Child class of base_Thing2vec which uses the gensim module for calculating word vectors.
    """
    def __init__(self, neural_mode, vocab_size=10000, window_size=5, vector_dim=100, negative_samples=5,  file=None, min_count=5, workers=3, properties=None):
        """
        Parameters
        ----------
        neural_mode : 
            defines the input for the neural net
        vocab_size : 
            size of vocabularity which defines the dimensionality of the one hot encoding
        window_size : 
            maximum distance of a current and context word for creating skipgrams
        vector_dim : 
            dimensionality of the embedded layer that will define the word vectors
        negative_samples : 
            how many negative samples are created per positive sample
        file : 
            string which points to the file with the corpus of text
        min_count : 
            minimal number of occurances for a word to be taken into account for the dictionary
        workers : 
            number of workers that will be created for calculating the word vectors
        properties: 
            Properties for the data creator that will be loaded. 
            This is only needed if neural_mode is set to physical2D
        """
        super().__init__(vocab_size,window_size, vector_dim, negative_samples, neural_mode,  file, properties)
        self.min_count = min_count
        self.workers = workers
        iterator = self.Thing2String(self.file, self.dictionary, self.seperator)
        self.model = Word2Vec(iterator, size = self.vector_dim, window=self.window_size, 
                                min_count= self.min_count, workers=self.workers, negative=self.negative_samples,sg=0, iter=0,
                                max_vocab_size=self.vocab_size)

    #TODO: One could introduce a system that saves the progress of training, meaning the actual epoch for recalculation
    # the needed alpha value. With that the user would give the desired number of iterations in the initialization
    # process and with Train method how long the training should be for the moment.
    def Train(self, file=None, epochs=10):
        """
        Trains the neural net for deriving word vectors using the gensim module.
        Parameters
        ----------
        file : 
            string which points to the file with the corpus of text, if None the 
            given file in the init function is used
        epochs : 
            number of epochs/iterations over the whole corpus of text
        """
        callbacks = [self.EpochLogger(epochs)]
        if file == None:
            file = self.file
        iterator = self.Thing2String(file, self.dictionary,self.seperator)
        self.model.train(iterator, total_examples=self.model.corpus_count, epochs=epochs, callbacks=callbacks)

    def get_word_vector(self, word, normalized = True):
        """
        Returns the word vector that corresponds to the given word.
        Parameters
        ----------
        word : 
            The word which vector is wanted
        normalized : 
            if =True the word vector will be normalized 
        """
        self.model.wv.init_sims() #in order to compute normalized matrix
        return self.model.wv.word_vec(str(self.dictionary[word]),use_norm=normalized)

    def get_word_vectors(self, normalized = False):
        """
        Returns a list of all word vectors trained by the neural net.
        Attention
        ---------
        The list has not to be in any order that corresponds to the internal dictionary
        Parameters
        ----------
        normalized : 
            if =True the word vectors will be normalized
        """
        self.model.wv.init_sims() #in order to compute normalized matrix
        if normalized:
            return self.model.wv.vectors_norm
        else:
            return self.model.wv.vectors

    def similarity(self, w1, w2):
        """
        Gives back the cosine similarity between two words.
        Parameters
        ----------
        w1 :
            Input word
        w2 :
            Input word
        """
        return self.model.wv.similarity(str(self.dictionary[w1]), str(self.dictionary[w2]))
    
    def similar_by_vector(self, vector, number=10, plot=True):
        """
        Gives back the words with the most similar word vectors to the given vector.
        Parameters
        ----------
        vector :
            origin, from which the search after similar word vectors is started
        number : 
            number of most similar words that is given back
        plot :
            if True the results will be plotted if possible
        """
        result =self.model.wv.similar_by_vector(vector,topn=number)
        result_final = list(map(lambda x: [self.reverse_dictionary[int(x[0])],x[1]], result))
        if (self.neural_mode == Neural_Mode.physical2D or self.neural_mode == Neural_Mode.physical2Dperiodic) and plot:
            particles = list(map(lambda x: self.reverse_dictionary[int(x[0])], result))
            titles = list(map(lambda x: x[1], result))
            self.datacreator.plot_states(particles, titles=titles)
        return result_final

    #TODO: Expand function in order to get plots of all positive and negative states if wanted from the user
    #with short title like "positive1", "negative2"
    def most_similar(self, positive=[], negative=[], number=10, plot = PlotMode.complete):
        """
        Gives back the most similar words. Positive words contribute positivly, negative words
        negativly. For measuring similarity cosine similarity is used as described in the original
        paper.
        Parameters
        ----------
        positive : 
            list of positive words which will be given to the most_similar method
        negative : 
            list of negative words which will be given to the most_similar method
        number : 
            number of most similar words that is given back
        plot :
            mode of plotting, wheather and how will be plotted
        """
        positive_dic = list(map(lambda x: str(self.dictionary[x]), positive))
        negative_dic = list(map(lambda x: str(self.dictionary[x]), negative))
        result = self.model.wv.most_similar(positive=positive_dic, negative=negative_dic, topn=number)
        result_final = list(map(lambda x: [self.reverse_dictionary[int(x[0])],x[1]], result))
        if (self.neural_mode in [Neural_Mode.physical2D, Neural_Mode.physical2Dperiodic, Neural_Mode.physical2DIsing]) and plot!=PlotMode.noPlotting:
            if plot==PlotMode.complete:
                titles_input = ["positive"] * len(positive) + ["negative"] * len(negative)
                self.datacreator.plot_states(positive + negative, titles=titles_input)
            particles = list(map(lambda x: self.reverse_dictionary[int(x[0])], result))
            titles = list(map(lambda x: x[1], result))
            self.datacreator.plot_states(particles, titles=titles)
        return result_final

    def most_similar_cosmul(self, positive=[], negative=[], number=10, plot = PlotMode.noPlotting):
        """
        Gives back the most similar words. Positive words contribute positivly, negative words
        negativly. For measuring similarity the multiplicative combination objective is used,
        see <http://www.aclweb.org/anthology/W14-1618>.
        Parameters
        ----------
        positive : 
            list of positive words which will be given to the most_similar method
        negative : 
            list of negative words which will be given to the most_similar method
        number : 
            number of most similar words that is given back
        plot :
            if True the results will be plotted if possible
        """
        positive_dic = list(map(lambda x: str(self.dictionary[x]), positive))
        negative_dic = list(map(lambda x: str(self.dictionary[x]), negative))
        result = self.model.wv.most_similar_cosmul(positive=positive_dic, negative=negative_dic, topn=number)
        result_final = list(map(lambda x: [self.reverse_dictionary[int(x[0])],x[1]], result))
        if (self.neural_mode in [Neural_Mode.physical2D, Neural_Mode.physical2Dperiodic, Neural_Mode.physical2DIsing]) and plot!=PlotMode.noPlotting:
            if PlotMode.complete:
                titles_input = ["positive"] * len(positive) + ["negative"] * len(negative)
                self.datacreator.plot_states(positive + negative, titles=titles_input)
            particles = list(map(lambda x: self.reverse_dictionary[int(x[0])], result))
            titles = list(map(lambda x: x[1], result))
            self.datacreator.plot_states(particles, titles=titles)
        return result_final

    def index2word(self, idx):
        """
        Returns the word to the given index.
        Parameters
        ----------
        idx : index of a word in the internal gensim implementation
        """
        return self.reverse_dictionary[int(self.model.wv.index2word[idx])]

    def save_model(self, path):
        """
        Saves the model at the given path.
        """
        self.model.save(path)

    def load_model(self, path):
        """
        Loads the model with the data given at path.
        """
        self.model = Word2Vec.load(path)

    def is_in_dictionary(self, word):
        """
        Gives back wheater the given word is in dictionary or not.
        Parameters
        ----------
        word :
            word that shall be tested
        """
        if word in self.dictionary:
            #this has to be tested for small datasets as can be seen in jupyter notebook thing2vec with physical data #small dataset
            if str(self.dictionary[word]) in self.model.wv.vocab: 
                return True
        return False
    class EpochLogger(CallbackAny2Vec):
        """
        Logs the status of the gensim learning process by using Callback methods.
        The status is update at the end of every epoch. 
        """
        def __init__(self, epochs):
            self.akt_epoch = 0
            self.logger = tools.progress_log(epochs)

        def on_epoch_end(self, model):
            self.akt_epoch += 1
            self.logger.update_progress(self.akt_epoch)

        def on_train_end(self, model):
            self.logger.finished()



class Thing2VecKeras(base_Thing2vec):
    """
    Child class of base_Thing2vec that uses the keras backend for calculating word vectors.
    Attention
    ---------
    The class is not finished yet!
    """

    def __init__(self, vocab_size, window_size, vector_dim, negative_samples, neural_mode, file, sg_file, optimizer, properties=None):
        """
        Parameters
        ----------
        vocab_size : 
            size of vocabularity which defines the dimensionality of the one hot encoding
        window_size : 
            maximum distance of a current and context word for creating skipgrams
        vector_dim : 
            dimensionality of the embedded layer that will define the word vectors
        negative_samples : 
            how many negative samples are created per positive sample
        neural_mode : 
            defines the input for the neural net
        file : 
            string which points to the file with the corpus of text
        sg_file :
            string which points to the location with the skipgrams created with the corpus of text
        optimizer :
            optimizer which shall be used for the keras neural net
        properties: 
            Properties for the data creator that will be loaded. 
            This is only needed if neural_mode is set to physical2D

        """
        super().__init__(vocab_size,window_size, vector_dim, negative_samples, neural_mode, file, properties)
        self.optimizer = optimizer
        self.file = file
        self.sg_file = sg_file
        self.SetupNeuralnet()

    def SetupNeuralnet(self):
        """
        Creates the neural network by using the keras module.
        """
        #print("Setup neural net...")
        # create some input variables
        input_target = Input((1,))
        input_context = Input((1,))

        #initialization values are originated in the gensim code
        embedding = Embedding(self.vocab_size, self.vector_dim, input_length=1, name='embedding_word') #Create embedding layer
        embedding_context = Embedding(self.vocab_size, self.vector_dim, input_length=1, name='embedding_context', embeddings_initializer=keras.initializers.RandomUniform(minval=-0.5/self.vector_dim,maxval=0.5/self.vector_dim)) #extra embedding layer for context
        target = embedding(input_target) #calculate the word vector of the target word
        target = Reshape((self.vector_dim, 1))(target)
        context = embedding_context(input_context) #calculate the word vector of the possible context word
        context = Reshape((self.vector_dim, 1))(context)

        # now perform the dot product operation to get a similarity measure
        dot_product = dot([target, context], axes = 1, normalize = False)
        dot_product = Reshape((1,))(dot_product)

        output = Activation('sigmoid')(dot_product) #With that approach there is no additional parameter that can be learned
        # create the primary training model
        model = Model(inputs=[input_target, input_context], outputs=output)
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer) #optimizer='SGD' #optimizer='rmsprop'

        #create a model which gives back the word_vector representation of the context words
        word_vector_model = Model(inputs=[input_context],outputs=context)
        self.model = model
        self.word_vector_model = word_vector_model


    def batchgenerator(self, batch_size):
        """
        Generates batch from the skip gram file given at the initialization.
        Parameters
        ----------
        batch_size :
            Number of skipgram pairs in one batch that is given to the neural net for training
        """
        def set_to_zero():
            return [np.zeros(batch_size, dtype='int32'), np.zeros(batch_size, dtype='int32'), np.zeros(batch_size, dtype='int32')]
        stream = open(self.sg_file, mode='r')
        while True:
            word_target, word_context, labels = set_to_zero()
            act_idx = 0 
            for idx, line in enumerate(stream):
                k = idx - act_idx
                word_target[k], word_context[k], labels[k] = line.replace("\n","").split(" ")
                if k == batch_size - 1:
                    yield ([word_target, word_context], labels)
                    word_target, word_context, labels = set_to_zero()
                    act_idx = idx
            stream.seek(0)


    def __number_of_skipgrams_in_file(self):
        """
        Counts the number of lines in the skip gram file to determine the number of skipgrams.
        """
        stream = open(self.sg_file, mode='r')
        length = 0
        while stream.readline() != "":
            length += 1
        stream.close()
        return length

    def Train(self, epochs, batch_size, initial_epoch=0):
        """
        Trains the model by using the keras api.
        Parameters
        ----------
        epochs :
            Number of final epoch that will be trained.
        batch_size :
            Number of skipgrams that will be given as one batch for training the neural net
        initial_epoch :
            Last learned epoch. So for starting learning the value is 0.
        """
        number = self.__number_of_skipgrams_in_file()
        self.model.fit_generator(self.batchgenerator(batch_size), epochs=epochs,steps_per_epoch = number//batch_size, verbose=1, initial_epoch=initial_epoch)

    def get_word_vector (self, word, normalized = True):
        """
        Returns the word vector that corresponds to the given word.
        Parameters
        ----------
        word : 
            The word which vector is wanted
        normalized : 
            if =True the word vector will be normalized 
        """
        in_word = np.zeros(1)
        if type(word) == int:
            in_word[0] = word
        else:
            in_word[0] = self.dictionary[word]
        vector = np.ndarray.flatten(self.word_vector_model.predict_on_batch(in_word)[0])
        if normalized:
            vector /= np.linalg.norm(vector)
        return vector
    
    def get_word_vectors(self, normalized = False):
        """
        Returns a list of all word vectors trained by the neural net.
        Attention
        ---------
        The list has not to be in any order that corresponds to the internal dictionary
        Parameters
        ----------
        normalized : 
            if =True the word vectors will be normalized
        """        
        in_batch = np.array([k for k in range (self.vocab_size)])
        vectors = self.word_vector_model.predict_on_batch(in_batch)
        return np.squeeze(vectors, axis=2)

    def index2word(self, idx):
        """
        Returns the word to the given index.
        Parameters
        ----------
        idx : index of a word in the internal gensim implementation
        """
        return self.reverse_dictionary[idx]

    #TODO: Add visalization for 2d systems as in gensim class
    def most_similar(self, positive=[], negative=[], number=10):
        """
        Gives back the most similar words. Positive words contribute positivly, negative words
        negativly. For measuring similarity cosine similarity is used as described in the original
        paper.
        Parameters
        ----------
        positive : 
            list of positive words which will be given to the most_similar method
        negative : 
            list of negative words which will be given to the most_similar method
        number : 
            number of most similar words that is given back
        """
        vectors = [] 
        for i in positive:
            vectors.append(self.get_word_vector(i,normalized=True))
        for i in negative:
            vectors.append((-1) * self.get_word_vector(i,normalized=True))
        if vectors == []:
            raise ValueError("cannot compute nearest words with no input")
        final_vec = np.mean(np.array(vectors),axis=0)
        norm_vec = final_vec / np.linalg.norm(final_vec)
        in_batch = np.array([k for k in range (self.vocab_size)])
        vectors = self.word_vector_model.predict_on_batch(in_batch)
        for v in vectors:
            v /= np.linalg.norm(v)
        similarity = [[self.reverse_dictionary[k],(np.transpose(vectors[k])@norm_vec)[0]] for k in range(len(vectors)) if self.reverse_dictionary[k] not in positive+negative]
        return sorted(similarity,reverse=True, key=tools.takeSecond)[:number]

    def most_similar_cosmul(self, positive=[],negative=[],number=10):
        """
        Gives back the most similar words. Positive words contribute positivly, negative words
        negativly. For measuring similarity the multiplicative combination objective is used,
        see <http://www.aclweb.org/anthology/W14-1618>.
        Parameters
        ----------
        positive : 
            list of positive words which will be given to the most_similar method
        negative : 
            list of negative words which will be given to the most_similar method
        number : 
            number of most similar words that is given back
        """
        in_batch = np.array([k for k in range (self.vocab_size)])
        vectors = self.word_vector_model.predict_on_batch(in_batch)
        for v in vectors:
            v /= np.linalg.norm(v)
        
        pos_dist, neg_dist = [], []
        for i in positive:
            pos_dist.append((1+np.dot(np.squeeze(vectors,axis=2), self.get_word_vector(i,normalized=True)))/2)    
        for i in negative:
            neg_dist.append((1+np.dot(np.squeeze(vectors,axis=2), self.get_word_vector(i,normalized=True)))/2)
        dist = np.prod(pos_dist,axis=0) / (np.prod(neg_dist, axis=0) + 0.000001)
        similarity = [[self.reverse_dictionary[k],dist[k]] for k in range(len(dist)) if self.reverse_dictionary[k] not in positive+negative]
        return sorted(similarity,reverse=True, key=tools.takeSecond)[:number]
    
    #TODO: Save the whole model? In order to just go on without again initialisation
    def save_model(self, path):
        """
        Saves the model at the given path.
        """
        self.model.save_weights(path)

    def load_model(self, path):
        """
        Loads the model with the data given at path.
        """
        self.model.load_weights(path)

    def make_cum_table(self, domain=2**31 - 1):
        """
        Calculates the noise distribution that is used for sampling of negative samples.
        The code is adopted from the gensim library. The distribution follows the stated
        one in the original paper.
        """
        cum_table = np.zeros(self.vocab_size-1, dtype=np.uint32) #in order to ignore UNK -1
        # compute sum of all power (Z in paper)
        train_words_pow = 0.0
        for word_index in range(1,self.vocab_size): #To ignore the UNK start with 1
            train_words_pow += self.count[word_index][1]**(0.75)
        cumulative = 0.0
        for word_index in range(1,self.vocab_size):
            cumulative += self.count[word_index][1]**(0.75)
            cum_table[word_index-1] = round(cumulative / train_words_pow * domain)
        return cum_table

    def load_embedding_matrix_from_gensim(self, thing2vecgensim):
        """
        Loads the embedding vectors from the gensim model word2vec into the 
        own model. 
        """
        def get_matrix(wvmatrix):
            wv_matrix = (np.random.rand(self.vocab_size, self.vector_dim) - 0.5) / 5.0
            for i in self.reverse_dictionary:
                if i >= self.vocab_size:
                    continue
                try:
                    index = thing2vecgensim.model.wv.vocab[str(thing2vecgensim.dictionary[self.reverse_dictionary[i]])].index
                    embedding_vector = wvmatrix[index] 
                    # words not found in embedding index will be all-zeros.
                    wv_matrix[i] = embedding_vector
                except:
                    pass
            return wv_matrix
        syn1neg = get_matrix(thing2vecgensim.model.trainables.syn1neg)
        syn0 = get_matrix(thing2vecgensim.model.wv.vectors)
        self.model.set_weights([syn1neg,syn0])

    def Generate_skipgrams(self, replace_context=False):
        if os.path.isfile(self.sg_file):
            print("Skipgram file already exists!")
            return None
        cum_table = self.make_cum_table()
        sentences = self.Thing2String(self.file, self.dictionary, transformation=int)
        sampling_table = sequence.make_sampling_table(self.vocab_size)
        self.skipgrams_sampled(sentences, self.sg_file, sampling_table=sampling_table, replace_context=replace_context, cum_table=cum_table)

    #TODO: This function has to be changed due just taking a file "sequence" which is not data_idx so
    # dictionary has to be applied.
    def skipgrams_sampled_old(self, sequence, vocabulary_size,
                window_size=4, negative_samples=1., shuffle=True,
                categorical=False, sampling_table=None, seed=None,
                unigram_distribution = True, replace_context=False, count = []):
        """
        Generates skipgram word pairs.
        Function originally from keras package with added functionality
        sampling words with different distances to the original word.
        With unigram_distribution the negative samples are sampled due
        to a unigram distribution.
        replace_context defines wheater the negtive samples are created
        by replacing the context word or the "goal" word.
        """
        couples = []
        labels = []
        for i, wi in enumerate(sequence):
            if not wi:
                continue
            if sampling_table is not None:
                if sampling_table[wi] < random.random():
                    continue
            
            reduced_window = random.randint(0,window_size) #Added code
            window_start = max(0, i - window_size + reduced_window)
            window_end = min(len(sequence), i + window_size + 1 - reduced_window)
            for j in range(window_start, window_end):
                if j != i:
                    wj = sequence[j]
                    if not wj:
                        continue
                    couples.append([wi, wj])
                    if categorical:
                        labels.append([0, 1])
                    else:
                        labels.append(1)

        if negative_samples > 0:
            num_negative_samples = int(len(labels) * negative_samples)
            if replace_context:
                words = [c[0] for c in couples]
            else:
                words = [c[1] for c in couples]
            if shuffle: 
                random.shuffle(words)
            if unigram_distribution:
                if count == []:
                    raise ValueError("Need count variable to create unigram distribution")
                cum_table = self.make_cum_table(count)
                if replace_context:
                    couples += [[words[i % len(words)],
                                int(cum_table.searchsorted(random.randint(0,cum_table[-1]))+1)] #+1 because of ignoring UNK, the int lowers memory consumption when saving variable with pickle
                                for i in range(num_negative_samples)]
                else:
                    couples += [[int(cum_table.searchsorted(random.randint(0,cum_table[-1]))+1), #+1 because of ignoring UNK, the int lowers memory consumption when saving variable with pickle
                                words[i % len(words)]] 
                                for i in range(num_negative_samples)]
            else:
                if replace_context:
                    couples += [[words[i % len(words)],
                                random.randint(1, vocabulary_size - 1)]
                                for i in range(num_negative_samples)]
                else:
                    couples += [[random.randint(1, vocabulary_size - 1), 
                                words[i % len(words)]]
                                for i in range(num_negative_samples)]
            if categorical:
                labels += [[1, 0]] * num_negative_samples
            else:
                labels += [0] * num_negative_samples

        if shuffle:
            if seed is None:
                seed = random.randint(0, 10e6)
            random.seed(seed)
            random.shuffle(couples)
            random.seed(seed)
            random.shuffle(labels)

        return couples, labels

    #TODO: One could speedup this procedure by implementing several threads that do the same thing with different sentences 
    #TODO: Problem: created file needs way too much space on hard drive (around ~40x more than the original file(depending on parameters))
    #TODO: For optimizing runtime the Cython compiler could be used https://cython.org/
    def skipgrams_sampled(self, sentences, result_file, sampling_table=None, replace_context=False, cum_table = None):
        """
        Generates skipgram word pairs and saves them to a file.
        Parameters
        ----------
        sentences : 
            iterable that gives a list of words of the individual sentences
        result_file :
            the file to which the resulting skipgrams shall be saved
        sampling_table :
            table for sampling occuring words in sentences. So more frequent words 
            will be downsampled. Use keras function for creating sampling_table
        replace_context :
            if true for negative samples the context word will be replaced by a negative sampled word
        cum_table :
            table for sampling negative samples. Use make_cum_table for generation.
        """
        stream = open(result_file, 'w')
        logger = None
        if type(sentences) == self.Thing2String:
            logger = tools.progress_log(sentences.len())
        for idx,sentence in enumerate(sentences):
            if logger != None:
                logger.update_progress(idx)
            if sampling_table is not None:
                idx_vocabs = [word for word in sentence if word != 0 and sampling_table[word] < random.random()]
            else:
                idx_vocabs = [word for word in sentence if word != 0]
            
            for pos, word in enumerate(idx_vocabs):
                reduced_window = random.randint(0,self.window_size) #Added code
                window_start = max(0, pos - self.window_size + reduced_window)
                window_end = min(len(idx_vocabs), pos + self.window_size + 1 - reduced_window)
                for pos2, word2 in enumerate(idx_vocabs[window_start:window_end],window_start):
                    if pos2 != pos:
                        stream.write(str(word) + " " + str(word2) + " 1\n")
                    for i in range(self.negative_samples):
                        if type(cum_table) != type(None):
                            if replace_context:
                                stream.write(str(word) + " " + str(cum_table.searchsorted(random.randint(0,cum_table[-1]))+1) + " 0\n")
                            else:
                                stream.write(str(cum_table.searchsorted(random.randint(0,cum_table[-1]))+1) + " " + str(word2) + " 0\n")
                        else:
                            if replace_context:
                                stream.write(str(word) + " " + str(random.randint(1, self.vocab_size - 1)) + " 0\n")
                            else:
                                stream.write(str(random.randint(1, self.vocab_size - 1)) + " " + str(word2) + " 0\n")
        stream.close()
        if logger != None:
            logger.finished()

