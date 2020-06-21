import pandas
import numpy
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import ShuffleSplit

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor	
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor


def relative_error(y_target, y_predicted): 
	y_target, y_predicted = numpy.array(y_target), numpy.array(y_predicted)
	change=[]
	for i in range(len(y_target)):
		if y_target[i]==0 and y_predicted[i]==0:
			change.append(0)
		else:
			change.append(abs(y_target[i]-y_predicted[i])/max([abs(y_target[i]),abs(y_predicted[i])]))
	return numpy.mean(change)


def evaluate_algorithms(features, targets):
	
	cv = ShuffleSplit() #shuffle for crossval. n_splits=10, test_size='default', train_size=None, random_state=None
	#cv=10
	
	print('Method\tMeanRelativeError\tMeanAbsoluteError')
	print
	
	regLR=LinearRegression()
	predicted = cross_val_predict(regLR, features, targets, cv=10)
	#print("%0.3f" % relative_error(targets,predicted))
	crossValScore=cross_val_score(regLR, features, targets, cv=cv, scoring='neg_mean_absolute_error')
	print('LinearRegression\t%0.3f\t%0.3f'% (relative_error(targets,predicted), abs(crossValScore.mean())))
	#print("Mean of mean absolute errors: %0.3f (+/- %0.3f)" % (crossValScore.mean(), crossValScore.std() * 2))
	regLR.fit(features, targets)
	print('coefficients',regLR.coef_)
	
	print
	
	regL=Lasso()
	predicted = cross_val_predict(regL, features, targets, cv=10)
	crossValScore=cross_val_score(regL, features, targets, cv=cv, scoring='neg_mean_absolute_error')
	print('Lasso\t%0.3f\t%0.3f'% (relative_error(targets,predicted), abs(crossValScore.mean())))
	
	print
	
	regR=Ridge()
	predicted = cross_val_predict(regR, features, targets, cv=10)
	crossValScore=cross_val_score(regR, features, targets, cv=cv, scoring='neg_mean_absolute_error')
	print('Ridge\t%0.3f\t%0.3f'% (relative_error(targets,predicted), abs(crossValScore.mean())))
	
	print
	
	regKR=KernelRidge()
	predicted = cross_val_predict(regKR, features, targets, cv=10)
	crossValScore=cross_val_score(regKR, features, targets, cv=cv, scoring='neg_mean_absolute_error')
	print('KernelRidge\t%0.3f\t%0.3f'% (relative_error(targets,predicted), abs(crossValScore.mean())))
	
	print
	
	regSVR_Lin=SVR(kernel='linear')
	predicted = cross_val_predict(regSVR_Lin, features, targets, cv=10)
	crossValScore=cross_val_score(regSVR_Lin, features, targets, cv=cv, scoring='neg_mean_absolute_error')
	print('SVR_Lin\t%0.3f\t%0.3f'% (relative_error(targets,predicted), abs(crossValScore.mean())))
	
	print
	
	regSVR_Poly=SVR(kernel='poly')
	predicted = cross_val_predict(regSVR_Poly, features, targets, cv=10)
	crossValScore=cross_val_score(regSVR_Poly, features, targets, cv=cv, scoring='neg_mean_absolute_error')
	print('SVR_Poly\t%0.3f\t%0.3f'% (relative_error(targets,predicted), abs(crossValScore.mean())))
	
	print
	
	regSVR_RBF=SVR(kernel='rbf')
	predicted = cross_val_predict(regSVR_RBF, features, targets, cv=10)
	crossValScore=cross_val_score(regSVR_RBF, features, targets, cv=cv, scoring='neg_mean_absolute_error')
	print('SVR_RBF\t%0.3f\t%0.3f'% (relative_error(targets,predicted), abs(crossValScore.mean())))
	
	print
	
	regKNR_U=KNeighborsRegressor()
	predicted = cross_val_predict(regKNR_U, features, targets, cv=10)
	crossValScore=cross_val_score(regKNR_U, features, targets, cv=cv, scoring='neg_mean_absolute_error')
	print('KNeighborsRegressor, weight uniform\t%0.3f\t%0.3f'% (relative_error(targets,predicted), abs(crossValScore.mean())))
	
	print
	
	regKNR_D=KNeighborsRegressor(weights='distance')
	predicted = cross_val_predict(regKNR_D, features, targets, cv=10)
	crossValScore=cross_val_score(regKNR_D, features, targets, cv=cv, scoring='neg_mean_absolute_error')
	print('KNeighborsRegressor, weight inversely proportional to distance\t%0.3f\t%0.3f'% (relative_error(targets,predicted), abs(crossValScore.mean())))
	
	print
	
	regGPR=GaussianProcessRegressor()
	predicted = cross_val_predict(regGPR, features, targets, cv=10)
	crossValScore=cross_val_score(regGPR, features, targets, cv=cv, scoring='neg_mean_absolute_error')
	print('GaussianProcessRegressor\t%0.3f\t%0.3f'% (relative_error(targets,predicted), abs(crossValScore.mean())))
	
	print
	
	regMLP=MLPRegressor()
	predicted = cross_val_predict(regMLP, features, targets, cv=10)
	crossValScore=cross_val_score(regMLP, features, targets, cv=cv, scoring='neg_mean_absolute_error')
	print('MLPRegressor\t%0.3f\t%0.3f'% (relative_error(targets,predicted), abs(crossValScore.mean())))
	
	print
	
	regDTR=DecisionTreeRegressor()
	predicted = cross_val_predict(regDTR, features, targets, cv=10)
	crossValScore=cross_val_score(regDTR, features, targets, cv=cv, scoring='neg_mean_absolute_error')
	print('DecisionTreeRegressor\t%0.3f\t%0.3f'% (relative_error(targets,predicted), abs(crossValScore.mean())))
	
	print
	
	regRFR=RandomForestRegressor()
	predicted = cross_val_predict(regRFR, features, targets, cv=10)
	crossValScore=cross_val_score(regRFR, features, targets, cv=cv, scoring='neg_mean_absolute_error')
	print('RandomForestRegressor\t%0.3f\t%0.3f'% (relative_error(targets,predicted), abs(crossValScore.mean())))
	
	print
	
	regB_RF=BaggingRegressor(RandomForestRegressor())
	predicted = cross_val_predict(regB_RF, features, targets, cv=10)
	crossValScore=cross_val_score(regB_RF, features, targets, cv=cv, scoring='neg_mean_absolute_error')
	print('BaggingRegressor with RandomForestRegressor\t%0.3f\t%0.3f'% (relative_error(targets,predicted), abs(crossValScore.mean())))
	
	print
	
	regB_DTR=BaggingRegressor(DecisionTreeRegressor())
	predicted = cross_val_predict(regB_DTR, features, targets, cv=10)
	crossValScore=cross_val_score(regB_DTR, features, targets, cv=cv, scoring='neg_mean_absolute_error')
	print('BaggingRegressor with DecisionTreeRegressor\t%0.3f\t%0.3f'% (relative_error(targets,predicted), abs(crossValScore.mean())))
	
	print
	
	regB_Lin=BaggingRegressor(LinearRegression())
	predicted = cross_val_predict(regB_Lin, features, targets, cv=10)
	crossValScore=cross_val_score(regB_Lin, features, targets, cv=cv, scoring='neg_mean_absolute_error')
	print('BaggingRegressor with LinearRegression\t%0.3f\t%0.3f'% (relative_error(targets,predicted), abs(crossValScore.mean())))
	
	print

	regGBR=GradientBoostingRegressor()
	predicted = cross_val_predict(regGBR, features, targets, cv=10)
	crossValScore=cross_val_score(regGBR, features, targets, cv=cv, scoring='neg_mean_absolute_error')
	print('GradientBoostingRegressor\t%0.3f\t%0.3f'% (relative_error(targets,predicted), abs(crossValScore.mean())))
	
	print
		
		
	print
	print


	

datafr=pandas.read_csv('SLN trialMean.csv', ',')	
#print(datafr.shape)
#print(datafr.iloc[0:10,:])

print('Pearson correlation')
print(datafr.corr('pearson'))
print
print('Spearman correlation')
print(datafr.corr('spearman'))

datafr=pandas.DataFrame(MinMaxScaler().fit_transform(datafr),columns=datafr.columns)
#print(datafr.iloc[0:10,:])

datafr = datafr.sample(frac=1).reset_index(drop=True) #shuffles datafr

featuresS=datafr.iloc[:,0:3].copy()
targetsArr=[]
targetsArr.append(datafr.iloc[:,4].copy())
targetsArr.append(datafr.iloc[:,5].copy())
targetsArr.append(datafr.iloc[:,6].copy())

targetNameArr=[]
targetNameArr.append(datafr.columns[4])
targetNameArr.append(datafr.columns[5])
targetNameArr.append(datafr.columns[6])

print
print

print('Predicting particle size using only time')
print
features=pandas.DataFrame(featuresS.loc[:,'Time'])
targets=targetsArr[0]
evaluate_algorithms(features, targets)


print('Predicting particle size using all features')
print

for i in range(len(targetsArr)):
	
	print(targetNameArr[i])
	targets=targetsArr[i]
	
	'''
	regr = linear_model.LinearRegression()
	regr.fit(featuresS, targets)
	print(regr.coef_)
	predictions = regr.predict(featuresS)
	print('Mean squared error: %.2f' % mean_squared_error(targets, predictions))
	print('Variance score: %.3f' % r2_score(targets, predictions))
	'''
	
	'''
	#rfecv = RFECV(estimator=RandomForestRegressor(), step=1, cv=5,scoring='neg_mean_absolute_error')
	rfecv = RFECV(estimator=DecisionTreeRegressor(), step=1, cv=5,scoring='neg_mean_absolute_error')
	features=pandas.DataFrame(rfecv.fit_transform(featuresS, targets))
	features.columns=featuresS.columns[rfecv.support_]
	print('targets',i)
	print(features.columns)
	'''
	
	features=featuresS
	evaluate_algorithms(features, targets)
