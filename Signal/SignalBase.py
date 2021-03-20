from abc import abstractmethod, ABCMeta, abstractstaticmethod


class SignalBase(object,metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def initialize(self):
        # Do some prepare to generate signals
        # load the factors 
        # update the factors
        # prepare some preprocess metetials 
        pass

    @abstractmethod
    def generate_signals(self):
        # the main main func of this class
        # iter through all time periods and get the signals
        # for each iteration: call train_test_slice, preprocessing, get_signal
        pass

    @abstractstaticmethod
    def train_test_slice(factors, dependents, trainStart, trainEnd, testStart, testEnd):
        # split all the factors and dependents to train part and test part according to input,
        # if end part isn't passed in, slice one period as default, 
        # if the test start isn't passed in,
        # take the very next time period of trainEnd,
        # the input of factors could be a list of factors or just one Factor
        pass

    @abstractmethod
    def preprocessing(self):
        # apply preprocess in here including 
        # clean up nans and 停牌 ST ect,
        # deal with extreme points
        # and other stuff
        # use np.ma module technic here should be suitable 
        # please make it modulized and easy to maintain (take cleanUpRules as inputs ect.)
        pass

    @abstractmethod
    def get_signal(self):
        # define how we get signal for one interation
        # the obviuos version will be use feature selection and models 
        # to predict crossSectional expected returns of next period
        pass

    @abstractmethod
    def smoothing(self):
        # smoothing methods defind at the end
        # typicaly is the moving average of n days
        # use partial function technic here will be suitable 
        pass