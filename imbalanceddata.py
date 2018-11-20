from sklearn.datasets import make_classification
from sklearn.neural_network import multilayer_perceptron
from matplotlib import pyplot as plt

X1,Y1 = make_classification(n_samples=1000,n_classes=3,n_features=2,n_informative=2,n_redundant=0,n_clusters_per_class=1,shuffle=True,
                            weights=(0.01,0.01,0.98),class_sep=0.6)




plt.scatter(X1[:,0],X1[:,1],marker="o",c = Y1,s=25,edgecolors="k")

plt.show()




trainingx = X1[:700]
trainingy = Y1[:700]
valx = X1[700:]
valy = Y1[700:]

model = multilayer_perceptron.MLPClassifier()
model.fit(trainingx,trainingy)
print("----Accuracy during training-----")
print(model.score(trainingx,trainingy))
print(model.score(valx,valy))
print("----Test the Minority Classes-----")
print("Training")
print(model.score(valx[valy==0],valy[valy==0]))
print(model.score(valx[valy==1],valy[valy==1]))
print(model.score(valx[valy==2],valy[valy==2]))

print("Validation")
print(model.score(trainingx[trainingy==0],trainingy[trainingy==0]))
print(model.score(trainingx[trainingy==1],trainingy[trainingy==1]))
print(model.score(trainingx[trainingy==2],trainingy[trainingy==2]))