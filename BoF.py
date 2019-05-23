import numpy as np
import math
from random import randint
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import accuracy_score

class BoF:
	setPuntos=False
	setFit=False
	def __init__(self, d=None,tipo=3,z=.1, w_min=5, b=10,n_clases=None,ntree=50,njobs=2, clf="RandomForest", mid=0):	
		self.mid=mid
		self.clf=clf
		self.njobs=njobs	
		self.ntree=ntree
		self.n_clases=n_clases
		self.tipo=tipo
		self.d=d   #Numero de intervalos para una subsecuencia			
		self.z=z			
		self.w_min=w_min
		self.b=b

	def fit(self, X, Y):			
		X.astype(np.double)
		if self.d==None:				
				T=X[0].shape[0] # si no se especifica d entonces d se calcula a partir de la longitud de la primera serie
								# asumiendo que todas las series poseen la misma longitud.
								# se podria utilizar una funcion que busque la longitud de la serie mas pequeña y
								# definir d a partir de esta
				#Generando CPE (Class probability estimate)						  
				self.d=math.floor((self.z*T)/self.w_min)			
		#self.f_train=self.vectorizacion(X,Y)
		f_train=self.vectorizacion(X,Y)	
		h_train=self.aHistrogramas(f_train,X.shape[0])				
		tolerance=0.05
		self.h_train_set=h_train
		if self.clf=="RandomForest":
			self.Rf2=RandomForestClassifier(n_estimators=self.ntree, criterion='gini',n_jobs=self.njobs, warm_start=True, min_impurity_decrease=self.mid, oob_score=True)
		elif self.clf=="BaggingClassifier":
			#print("bagging 2")
			base=DecisionTreeClassifier(splitter="random",min_impurity_decrease=self.mid)
			self.Rf2=BaggingClassifier(base_estimator =base,n_estimators=self.ntree,n_jobs=self.njobs, warm_start=True, oob_score=True)	
		elif self.clf=="ExtraTreesClassifier":
			self.Rf2=ExtraTreesClassifier(n_estimators=self.ntree,n_jobs=self.njobs, warm_start=True, min_impurity_decrease=self.mid, oob_score=True, bootstrap=True)			
		else:
			print("Valor de Clasificador equivocado : ", self.clf)
		h_train=h_train.round(decimals=7)
		self.Rf2.fit(h_train, Y)
		prev_OOBerror=1				
		cur_OOBerror=1-self.Rf2.oob_score_ 
		#print(cur_OOBerror)
		itera=0				
		while (itera<20) and cur_OOBerror<(1-tolerance)*prev_OOBerror:			
			prev_OOBerror=cur_OOBerror
			self.Rf2.n_estimators +=self.ntree				
			self.Rf2.fit(h_train, Y)									
			cur_OOBerror=1-self.Rf2.oob_score_ 
			#print(cur_OOBerror)
			itera +=1		
		self.setFit=True	

	def predict(self, X):		
		if self.setFit:
			Y=np.zeros((X.shape[0],1),dtype=int)
			f_test=self.vectorizacion(X, Y)
			h_test=self.aHistrogramas(f_test,X.shape[0],fase="test")			
			prediccion=self.Rf2.predict(h_test)						
			return prediccion
		else:
			print("No se ha entrenado, utilizar funcion fit()")

	def transform(self, X):
		if self.setFit:
			Y=np.zeros((X.shape[0],1),dtype=int)
			f_test=self.vectorizacion(X, Y)
			h_test=self.aHistrogramas(f_test,X.shape[0],fase="test")
			return h_test
		else:
			print("No se ha entrenado, utilizar funcion fit()")
		
	def getHtrain(self):
		return self.h_train_set

	def aHistrogramas(self,Vector,n_series,fase="train"):				
		indice_clase=(self.d*3)+4
		indice_id=(self.d*3)+5
		indice_caracteristicas=(self.d*3)+3		
		arreglo_clases=Vector[:,indice_clase:(indice_clase+1)]
		arreglo_clases=arreglo_clases.astype(int)
		arreglo_id=Vector[:,indice_id:(indice_id+1)]
		arreglo_id=arreglo_id.astype(int)
		arreglo_caracteristicas=Vector[:,0:(indice_caracteristicas+1)]		
		if fase == "train" and self.n_clases==None:
			self.n_clases=np.unique(arreglo_clases).shape[0]		
		arreglo_clases=arreglo_clases.ravel()				
		if fase == "train":			
			tolerance=0.05			
			if self.clf=="RandomForest":
				self.Rf1=RandomForestClassifier(n_estimators=self.ntree, criterion='gini',n_jobs=self.njobs, warm_start=True, min_impurity_decrease=self.mid, oob_score=True)
			elif self.clf=="BaggingClassifier":				
				base=DecisionTreeClassifier(splitter="random",min_impurity_decrease=self.mid)
				self.Rf1=BaggingClassifier(base_estimator =base,n_estimators=self.ntree,n_jobs=self.njobs, warm_start=True, oob_score=True)
			elif self.clf=="ExtraTreesClassifier":
				self.Rf1=ExtraTreesClassifier(n_estimators=self.ntree,n_jobs=self.njobs, warm_start=True, min_impurity_decrease=self.mid, oob_score=True, bootstrap=True)
			else:
				print("Valor de Clasificador equivocado : ", self.clf)			
			arreglo_caracteristicas=arreglo_caracteristicas.round(decimals=7)

			self.Rf1.fit(arreglo_caracteristicas, arreglo_clases)					
			prev_OOBerror=1						
			#cur_OOBerror=1-accuracy_score(arreglo_clases,self.Rf1.predict(arreglo_caracteristicas))			
			"""
			oob_score_ : float
    		Score of the training dataset obtained using an out-of-bag estimate.
			oob_decision_function_ : array of shape = [n_samples, n_classes]

    		Decision function computed with out-of-bag estimate on the training set. If n_estimators is small it might
			be possible that a data point was never left out during the bootstrap. In this case, oob_decision_function_ 
			might contain NaN.
			"""

			cur_OOBerror=1-self.Rf1.oob_score_
			itera=0		
			#print(cur_OOBerror)						
			#for i in self.Rf1.oob_decision_function_:
			#	print(i)
			#print(self.Rf1.oob_decision_function_[0])
			while (itera<20) and (cur_OOBerror<((1-tolerance)*prev_OOBerror)):				
				prev_OOBerror=cur_OOBerror	
				self.Rf1.n_estimators +=self.ntree				
				self.Rf1.fit(arreglo_caracteristicas, arreglo_clases)								
				cur_OOBerror=1-self.Rf1.oob_score_
				itera +=1
				#print(cur_OOBerror)									
				#print(self.Rf1.oob_decision_function_[0])
					
		#CPE=self.Rf1.predict_proba(arreglo_caracteristicas)
		if fase == "train":	
			CPE= np.nan_to_num(self.Rf1.oob_decision_function_)
		else:
			CPE=self.Rf1.predict_proba(arreglo_caracteristicas)
		#Reduciendo dimensionalidad (C-1) ya que la sumatoria de los valores de las clases es igual a 1 se descarta una clase (la ultima por comodidad)	
		#indice_cpe=CPE.shape[1]-1		
		#CPE=CPE[:,0:indice_cpe]			*Se eliminimo por que el codigo de Baydogan no elimina clases (aunque en el paper dice que si)		
		#La cantidad de palabras en el codebook es igual a (C-1)*b + C, si se sigue el paper al pie de la letra
		#o  (C-1)*b + C, si se sigue el codigo, siendo b el numero de contenedores (para discretizar los valores de las probabilidades)
		#esto quiere decir que con b=10 la palabra C1-1 contiene los valores de la clase 1 comprendidos de 0 a 0.1, los valores de C1-2 los valores de la clase 1 entre
		#.1 y .2 y asi consecutivamente. Las  ultimas C casillas de los vectores almacenan la frecuencia relativa de cada clase
		#es decir que se agrega una palabra para cada clase y se le suma 1 cada que esta es la que tiene mayor probabilidad de ser eelegida. 
		h_series=np.zeros((n_series,self.n_clases*(self.b+1)),dtype=int)	
		arreglo_id=arreglo_id.transpose()[0]
		get_indices = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]								
		for i in range (n_series):			
			indices_serie=get_indices(i,arreglo_id)									
			for j in indices_serie:							
				val=-1   #almacener el promedio mayor
				for k in range(self.n_clases):																
					conj_palabra=self.getPalabra(CPE[j][k],i,j,k) 
					palabra=int(k*self.b+conj_palabra)					
					h_series[i][palabra] += 1	
					if CPE[j][k]>val:
						f_rel=k
						val=CPE[j][k]
				h_series[i][self.n_clases*self.b + f_rel] += 1				
		return h_series			
	
	def getPalabra(self,valor,i,j,k):	
		temp=math.floor(valor*self.b)
		if temp== self.b:			
			temp-=1			
		return temp

	def vectorizacion(self,X,Y):		
		vectorizados=[]						
		for i in range (X.shape[0]):
			T=X[i].shape[0]
			l_min=math.floor(T*self.z)			
			n_sub=math.floor((T/self.w_min) -self.d)			
			if l_min < (self.d*self.w_min):
				print("longitud minima de la subsecuencia demasiado pequeña")								
				break			
			if i==0 and self.setPuntos==False:
				self.puntos=self.generarPuntos(X[i].shape[0],l_min,n_sub)
				self.setPuntos=True			
			puntos=self.puntos		
			if self.tipo==3:		
				puntos=self.generarPuntos(X[i].shape[0],l_min,n_sub)  
			vectores_serie=self.f_extraccion(Y[i],X[i],puntos,i)			
			vectorizados.append(vectores_serie)				
		vectorizados=np.concatenate( vectorizados, axis=0 )		
		return vectorizados

	def test(self,Test):
		#vectorizar y convertir en histogramas... y eso 
		print("en construccion")

	def f_extraccion(self,clase,serie,puntos,id_serie):
		resp=np.zeros((puntos.shape[0],(self.d*3)+6), dtype=np.double)					
		for x in range (puntos.shape[0]):			
			s=0			
			for y in range (self.d):
				punto_inicial=int(math.floor(y*(((puntos[x][1])-puntos[x][0])/self.d)+puntos[x][0]))
				punto_final=int(math.floor((y+1)*(((puntos[x][1])-puntos[x][0])/self.d)+puntos[x][0]))										
				resp[x][s]=self.Slope(serie[punto_inicial:punto_final],punto_inicial,punto_final)
				resp[x,s+1]=np.mean(serie[punto_inicial:punto_final],axis=0)
				resp[x][s+2]=np.var(serie[punto_inicial:punto_final],axis=0)				
				s=s+3				
			resp[x][s]=np.mean(serie[puntos[x][0]:puntos[x][1]+1],axis=0)
			resp[x][s+1]=np.var(serie[puntos[x][0]:puntos[x][1]+1],axis=0)				
			resp[x][s+2]=puntos[x][0]
			resp[x][s+3]=puntos[x][1]
			resp[x][s+4]=clase
			resp[x][s+5]=id_serie
		return resp

	def generarPuntos(self,c_puntos_serie,l_min,n_sub):			
		resp=np.zeros((n_sub,2),dtype=int)
		if self.tipo==2:     			
			#se generan ventanas seguidas una de otra			
			for i in range (n_sub):
				resp[i][0]=self.w_min*i
				resp[i][1]=self.w_min*i+l_min						
		else:			
			for i in range (n_sub):
				resp[i][0]=randint(0,(c_puntos_serie-l_min))
				final=math.floor((c_puntos_serie-resp[i][0])/self.d)
				
				final=randint(self.w_min,final)
				final=final*self.d
				resp[i][1]=resp[i][0]+final		
		return resp

	def Slope(self,serie,punto_inicial,punto_final):
		x=list(range(punto_inicial,punto_final))
		ymean = np.mean(serie,axis=0)
		x=np.asarray(x)			
		xmean=np.mean(x,axis=0)
		sx=0
		sxy=0
		for i in range(x.shape[0]):			
			sx=sx+math.pow((x[i]-xmean),2)			
			sxy=sxy+(x[i]-xmean)*(serie[i]-ymean)				
		if sx==0 and sxy==0:
			print ("pi:",punto_inicial," pf:",punto_final)		
		return sxy/sx