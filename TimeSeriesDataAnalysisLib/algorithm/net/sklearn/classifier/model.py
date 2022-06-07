
from TimeSeriesDataAnalysisLib.algorithm.net.sklearn.base import BaseSKModelBuilder
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import TimeSeriesDataAnalysisLib.algorithm.batch_algorithm as ba
class KNN(BaseSKModelBuilder):
    def __init__(self,inputs_shape:list,class_num:int):
        super().__init__(inputs_shape,class_num)
    def create_model(self,*args, **kwargs):
        ## build your model here ##
        self.model = KNeighborsClassifier()
        return self.isModelExist()

class KNN_Scan(BaseSKModelBuilder):
    def __init__(self,inputs_shape:list,class_num:int):
        super().__init__(inputs_shape,class_num)
    def create_model(self,*args, **kwargs):
        classifier=KNeighborsClassifier()
        k_range = list(range(2,self.class_num))
        weight_options = ['uniform','distance']
        algorithm_options = ['auto','ball_tree','kd_tree','brute']
        leaf_range = list(range(1,2))
        param_grid = dict(n_neighbors = k_range,weights = weight_options,algorithm=algorithm_options,leaf_size=leaf_range)
        self.model = GridSearchCV(classifier,param_grid,scoring='accuracy',verbose=1)
        return self.isModelExist()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
class LDA(BaseSKModelBuilder):
    def __init__(self,inputs_shape:list,class_num:int):
        super().__init__(inputs_shape,class_num)
    def create_model(self,*args, **kwargs):
        ## build your model here ##
        self.model = LinearDiscriminantAnalysis(n_components=self.class_num-1)
        return self.isModelExist()

class LDA_Scan(BaseSKModelBuilder):
    def __init__(self,inputs_shape:list,class_num:int):
        super().__init__(inputs_shape,class_num)
    def create_model(self,*args, **kwargs):
        classifier=LinearDiscriminantAnalysis()
        param_grid = dict(n_components=[self.class_num-1])
        self.model = GridSearchCV(classifier,param_grid,scoring='accuracy',verbose=1)
        return self.isModelExist()

from sklearn.linear_model import LogisticRegression as LR
class LogisticRegression(BaseSKModelBuilder):
    def __init__(self,inputs_shape:list,class_num:int):
        super().__init__(inputs_shape,class_num)
    def create_model(self,*args, **kwargs):
        self.model = LR(penalty='l2',random_state=0, solver='lbfgs', multi_class='multinomial',)
        return self.isModelExist()

from sklearn.tree import DecisionTreeClassifier
class DecisionTree(BaseSKModelBuilder):
    def __init__(self,inputs_shape:list,class_num:int):
        super().__init__(inputs_shape,class_num)
    def create_model(self,*args, **kwargs):
        self.model = DecisionTreeClassifier()
        return self.isModelExist()
    
from sklearn.ensemble import RandomForestClassifier
class RandomForest(BaseSKModelBuilder):
    def __init__(self,inputs_shape:list,class_num:int):
        super().__init__(inputs_shape,class_num)
    def create_model(self,*args, **kwargs):
        self.model = RandomForestClassifier()
        return self.isModelExist()

from sklearn.svm import SVC as skSVC
class SVC(BaseSKModelBuilder):
    def __init__(self,inputs_shape:list,class_num:int):
        super().__init__(inputs_shape,class_num)
    def create_model(self,*args, **kwargs):
        self.model = skSVC()
        return self.isModelExist()

from sklearn.svm import NuSVC as NSVC
class NuSVC(BaseSKModelBuilder):
    def __init__(self,inputs_shape:list,class_num:int):
        super().__init__(inputs_shape,class_num)
    def create_model(self,*args, **kwargs):
        self.model = NSVC()
        return self.isModelExist()

from sklearn.svm import LinearSVC as LSVC
class LinearSVC(BaseSKModelBuilder):
    def __init__(self,inputs_shape:list,class_num:int):
        super().__init__(inputs_shape,class_num)
    def create_model(self,*args, **kwargs):
        self.model = LSVC()
        return self.isModelExist()

from sklearn.naive_bayes import GaussianNB as GNB
class GaussianNB(BaseSKModelBuilder):
    
    def __init__(self,inputs_shape:list,class_num:int):
        super().__init__(inputs_shape,class_num)
    def create_model(self,*args, **kwargs):
        self.model = GNB()
        return self.isModelExist()

from sklearn.naive_bayes import MultinomialNB as MNB
class MultinomialNB(BaseSKModelBuilder):
    def __init__(self,inputs_shape:list,class_num:int):
        super().__init__(inputs_shape,class_num)
    def create_model(self,*args, **kwargs):
        self.model = MNB()
        return self.isModelExist()

from sklearn.naive_bayes import BernoulliNB as BNB
class BernoulliNB(BaseSKModelBuilder):
    def __init__(self,inputs_shape:list,class_num:int):
        super().__init__(inputs_shape,class_num)
    def create_model(self,*args, **kwargs):
        self.model = BNB()
        return self.isModelExist()

from sklearn.naive_bayes import ComplementNB as CNB
class ComplementNB(BaseSKModelBuilder):
    def __init__(self,inputs_shape:list,class_num:int):
        super().__init__(inputs_shape,class_num)
    def create_model(self,*args, **kwargs):
        self.model = CNB()
        return self.isModelExist()
