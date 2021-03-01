'''
    Most of this code is based on the code of the original author:
    Michał Dobrzański, 2016
    dobrzanski.michal.daniel@gmail.com
    Nevertheless, many changes and adjustments were made as described in the report.
'''
import mnist_loader
import network2

training_data, validation_data, test_data, input_dimension = mnist_loader.load_data_wrapper()

#this is done to save time in training
#mnist_loader.save(training_data, validation_data, test_data, input_dimension, 'PCA_data') #save dim_reduced data
#training_data, validation_data, test_data, input_dimension = mnist_loader.load('PCA_data') #load the data

training_data = list(training_data)
validation_data = list(validation_data)

input_layer = input_dimension
second_layer = 30
third_layer = 30
fourth_layer = 16
fifth_layer = 16
last_layer = 10
epochs = 5
batch_size = 1
eta = 0.1
lmbda = 5.0
n = len(training_data[:50000])
evaluation_data = validation_data[:10000] #could also be test_data
monitor_evaluation_cost = True
monitor_evaluation_accuracy = True
monitor_training_cost = True
monitor_training_accuracy = True
print('Network: [{}, {}, {}], cost function = CrossEntropyCost'.format(input_layer,
                    second_layer, last_layer))
print('training_data: {}, evaluation_data: 10000'.format(n))
print('learning rate = {}, lmbda = {}'.format(eta, lmbda))
print('epochs = {}, batch size = {}'.format(epochs, batch_size))
print('monitor_evaluation_accuracy={}'.format(monitor_evaluation_accuracy))
print('monitor_training_accuracy={}'.format(monitor_training_accuracy))
print('monitor_evaluation_cost={}'.format(monitor_evaluation_cost))
if input_layer < 784:
    print('Reducing data dimensionality using PCA to {} dimensions'.format(input_layer))
net = network2.Network([input_layer, second_layer, last_layer],
                    cost=network2.CrossEntropyCost)
net.SGD(training_data, epochs, batch_size, eta, lmbda, evaluation_data,
    monitor_evaluation_cost, monitor_evaluation_accuracy,
    monitor_training_cost, monitor_training_accuracy)
